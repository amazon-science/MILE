import math
import sys

import numpy as np
from dino import utils
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from logger_config import logger

def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    optimizer.zero_grad()

    for it, (images, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        if args.view == "multi-view":
            images = [[im.cuda() for im in view] for view in images]
            # teacher and student forward passes + compute dino loss
            with torch.cuda.amp.autocast(fp16_scaler is not None):
    
                teacher_output = teacher([view[:2] for view in images])  # only the 2 global views pass through the teacher
                student_output = student(images)
                loss = dino_loss(student_output, teacher_output, epoch)
        else:
            images = [im.cuda(non_blocking=True) for im in images]
            # teacher and student forward passes + compute dino loss
            with torch.cuda.amp.autocast(fp16_scaler is not None):
                teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
                student_output = student(images)
                loss = dino_loss(student_output, teacher_output, epoch)

        if not math.isfinite(loss.item()):
            logger.info("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)
            
        param_norms = None
        update_ema = False
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            if (it+1) % args.gradient_acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                update_ema = True
        else:
            assert False, "loss scaler not expected"
 

        if update_ema and not args.distil_frozen_teacher: # if distilation, no need to update teacher
            with torch.no_grad():
                if args.num_gpus > 1:
                    student_params = student.module.named_parameters()
                else:
                    student_params = student.named_parameters()
                teacher_params = teacher_without_ddp.named_parameters()
                m = momentum_schedule[it]  # momentum parameter
                backbone_m = m
 

                for (name_q, param_q), (name_k, param_k) in zip(student_params, teacher_params):
                    adaptive_m = m
                    assert name_q == name_k
                    if "backbone" in name_q:
                        adaptive_m = backbone_m
                    assert not args.distil_frozen_teacher, "Distilling frozen teacher therefore ema should be disabled."
                    param_k.data.mul_(adaptive_m).add_((1 - adaptive_m) * param_q.detach().data)
                metric_logger.update(momentum_teacher=m)
                metric_logger.update(momentum_backbone=backbone_m)

        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        if args.num_gpus > 1:
            if args.gate_tanh and args.view == "multi-view":
                metric_logger.update(student_tanh=nn.Tanh()(student.module.gate_alpha))
                metric_logger.update(teacher_tanh=nn.Tanh()(teacher.gate_alpha))

            if args.dual_gate_tanh and args == "multi-view":
                metric_logger.update(student_dual_tanh=nn.Tanh()(student.module.dual_gate_alpha))
                metric_logger.update(teacher_dual_tanh=nn.Tanh()(teacher.dual_gate_alpha))
            
        else:
            if args.gate_tanh and args.view == "multi-view":
                metric_logger.update(student_tanh=nn.Tanh()(student.gate_alpha))
                metric_logger.update(teacher_tanh=nn.Tanh()(teacher.gate_alpha))

            if args.dual_gate_tanh and args.view == "multi-view":
                metric_logger.update(student_dual_tanh=nn.Tanh()(student.dual_gate_alpha))
                metric_logger.update(teacher_dual_tanh=nn.Tanh()(teacher.dual_gate_alpha))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if utils.get_world_size() != 1:
            dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * utils.get_world_size())
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)