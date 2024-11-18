import argparse
import datetime
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import models as torchvision_models

from data.datasets import MILESampler, MVDataset
from dino.data import DataAugmentationDINO
from dino.dino_train import DINOLoss, train_one_epoch
from dino.dino_args import get_dino_args, get_mile_args
from sagemaker.sagemaker_args import get_sagemaker_args

from logger_config import logger
from model.train_stud_teacher_arch import apply_peft,build_teacher, wrap_models
from dino.utils import LARS, cosine_scheduler, fix_random_seeds, get_params_groups, get_world_size, has_batchnorms, is_main_process, save_on_master

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def parse_arguments():
    parser = argparse.ArgumentParser('DINO', parents=[get_dino_args(), get_mile_args(), get_sagemaker_args()])
    return parser.parse_args()

def setup_environment(args):
    fix_random_seeds(args.seed)
    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True


def prepare_data(args):
    logger.info("Preparing data...")
    transform = DataAugmentationDINO(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
        args.patch_size,
    )
    if args.view == "single":
        dataset = datasets.ImageFolder(args.data_path, transform=transform)
    elif args.view == "multi-view":
        random_k_samples = MILESampler(
            args.samples_per_class,
            transform,
            stitching=args.stitching
        )
        dataset = MVDataset(args.data_path, random_k_samples)
    else:
        raise ValueError(f"Unknown view mode {args.view}")
    
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True) if args.num_gpus > 1 else torch.utils.data.RandomSampler(dataset)
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    logger.info(f"Data loader created with {len(dataset)} samples")
    return data_loader

def build_models(args):
    logger.info("Building models...")
    student = build_teacher(args)
    teacher = build_teacher(args)
    
    if args.peft:
        logger.info("Applying PEFT...")
        student, teacher = apply_peft(student, teacher)
    
    student, teacher = wrap_models(student, teacher, args)
    
    student, teacher = student.cuda(), teacher.cuda()
    
    if has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)
        if args.num_gpus > 1:
            teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu], find_unused_parameters=True)
    
    if args.num_gpus > 1:
        student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu], find_unused_parameters=True)
    
    for p in teacher.parameters():
        p.requires_grad = False
    
    logger.info("Models built and prepared")
    return student, teacher

def setup_loss_and_optimizer(args, student):
    logger.info("Setting up loss and optimizer...")
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    params_groups = get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)
    elif args.optimizer == "lars":
        optimizer = LARS(params_groups)

    fp16_scaler = torch.cuda.amp.GradScaler() if args.use_fp16 else None

    logger.info(f"Using {args.optimizer} optimizer")
    return dino_loss, optimizer, fp16_scaler

def setup_schedulers(args, data_loader):
    logger.info("Setting up schedulers...")
    lr_schedule = cosine_scheduler(
        args.lr * get_world_size(),
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    momentum_schedule = cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))

    logger.info("Loss, optimizer and schedulers ready.")
    
    return lr_schedule, wd_schedule, momentum_schedule

def train_dino(args):
    setup_environment(args)

    data_loader = prepare_data(args)
    student, teacher = build_models(args)
    dino_loss, optimizer, fp16_scaler = setup_loss_and_optimizer(args, student)
    lr_schedule, wd_schedule, momentum_schedule = setup_schedulers(args, data_loader)

    start_epoch = 0
    start_time = time.time()
    logger.info("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        if args.num_gpus > 1:
            data_loader.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(student, teacher, teacher, dino_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule, 
            epoch, fp16_scaler, args)

        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if is_main_process():
            os.path.makedirs(args.output_dir, exist_ok=True)
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
            if args.saveckp_freq and (epoch < 10 or (epoch % args.saveckp_freq == 0)):
                save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
    
        logger.info(f"Epoch {epoch} completed. Train stats: {train_stats}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f'Training completed. Total time: {total_time_str}')

if __name__ == '__main__':
    args = parse_arguments()

    if not args.distributed_mode:
        train_dino(args)
    else:
        from launch_dist_train import launch

        machine_rank = os.environ["SM_CURRENT_HOST"]
        try:
            machine_rank = int(machine_rank.replace("algo-", "")) - 1
            print(f"machine rank {os.environ['SM_CURRENT_HOST']} -> {machine_rank}")
        except Exception as e:
            assert False, f"FAILED parsing machine rank {machine_rank} {e}"

        print("machines", args.num_machines)
        print("num gpus", args.num_gpus)
        print("dist url", args.dist_url)
        launch(
            train_dino,
            num_gpus_per_machine = args.num_gpus,
            num_machines = args.num_machines,
            machine_rank = machine_rank,
            dist_url = args.dist_url,
            args=(args,),
        )
        print("TRAINING DONE")

    logger.info("TRAINING DONE")