# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

import logging
from datetime import timedelta
import torch
import torch.multiprocessing as mp
import os

import torch.distributed as dist
import mmoc

__all__ = ["DEFAULT_TIMEOUT", "launch"]

DEFAULT_TIMEOUT = timedelta(minutes=60)


def _find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port

from datetime import timedelta


def launch_distributed(
    main_func,
    num_gpus_per_machine,
    num_machines,
    machine_rank,
    dist_url=None,
    args=(),
    timeout=DEFAULT_TIMEOUT,
):
    assert num_machines > 1, "Running distributed on more than one machine"

    # Environment variables set by torch.distributed.launch
    LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    try: 
        WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    except Exception as e:
        print("computing world size manually")
        WORLD_SIZE = num_machines * num_gpus_per_machine
    try:
        WORLD_RANK = int(os.environ['RANK'])
    except Exception as e:
        print("computing world rank manually")
        WORLD_RANK = machine_rank * num_gpus_per_machine + LOCAL_RANK

    print(f"LR={LOCAL_RANK}/WR={WORLD_RANK}/WS={WORLD_SIZE} 0/ parsed local_rank/world_rank/world_size")

    assert torch.cuda.is_available(), "Need GPUs"
    print(f"LR={LOCAL_RANK}/WR={WORLD_RANK}/WS={WORLD_SIZE} 1/ setting GPU device GPU={LOCAL_RANK}")
    torch.cuda.set_device(LOCAL_RANK)

    print(f"LR={LOCAL_RANK}/WR={WORLD_RANK}/WS={WORLD_SIZE} 2/ calling init_process_group")
    dist.init_process_group("NCCL", rank=WORLD_RANK, world_size=WORLD_SIZE)
    
    print(f"LR={LOCAL_RANK}/WR={WORLD_RANK}/WS={WORLD_SIZE} 3/ calling dist.barrier()")
    dist.barrier()

    print(f"LR={LOCAL_RANK}/WR={WORLD_RANK}/WS={WORLD_SIZE} 4/ calling main_func(*args)")

    main_func(*args)

def launch(
    main_func,
    # Should be num_processes_per_machine, but kept for compatibility.
    num_gpus_per_machine,
    num_machines=1,
    machine_rank=0,
    dist_url=None,
    args=(),
    timeout=DEFAULT_TIMEOUT,
):
    """
    Launch multi-process or distributed training.
    This function must be called on all machines involved in the training.
    It will spawn child processes (defined by ``num_gpus_per_machine``) on each machine.

    Args:
        main_func: a function that will be called by `main_func(*args)`
        num_gpus_per_machine (int): number of processes per machine. When
            using GPUs, this should be the number of GPUs.
        num_machines (int): the total number of machines
        machine_rank (int): the rank of this machine
        dist_url (str): url to connect to for distributed jobs, including protocol
                       e.g. "tcp://127.0.0.1:8686".
                       Can be set to "auto" to automatically select a free port on localhost
        timeout (timedelta): timeout of the distributed workers
        args (tuple): arguments passed to main_func
    """

    if num_machines > 1:
        return launch_distributed(main_func, num_gpus_per_machine, num_machines, machine_rank, dist_url, args, timeout)

    mp.start_processes(
        _distributed_local_worker,
        nprocs=world_size,
        args=(
            main_func,
            world_size,
            num_gpus_per_machine,
            machine_rank,
            dist_url,
            args,
            timeout,
        ),
        daemon=False,
    )

def _distributed_local_worker(
    local_rank,
    main_func,
    world_size,
    num_gpus_per_machine,
    machine_rank,
    dist_url,
    args,
    timeout=DEFAULT_TIMEOUT,
):
    has_gpu = torch.cuda.is_available()
    if has_gpu:
        assert num_gpus_per_machine <= torch.cuda.device_count()
    global_rank = machine_rank * num_gpus_per_machine + local_rank
    try:
        dist.init_process_group(
            backend="NCCL" if has_gpu else "GLOO",
            init_method=dist_url,
            world_size=world_size,
            rank=global_rank,
            timeout=timeout,
        )
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error("Process group URL: {}".format(dist_url))
        raise e

    # Setup the local process group.
    mmoc.create_local_process_group(num_gpus_per_machine)
    if has_gpu:
        torch.cuda.set_device(local_rank)
    # synchronize is needed here to prevent a possible timeout after calling init_process_group
    # See: https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
    mmoc.synchronize()

    main_func(*args)
