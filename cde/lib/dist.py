from typing import Callable
import multiprocessing
import os

import torch


def get_world_size() -> int:
    try:
        return torch.distributed.get_world_size()
    except (RuntimeError, ValueError):
        return 1


def get_rank() -> int:
    try:
        return torch.distributed.get_rank()
    except (RuntimeError, ValueError):
        return 0
    
def gather(t: torch.Tensor) -> torch.Tensor:
    # torch.distributed.nn.all_gather scales by world size since the reduce op is SUM
    # https://github.com/pytorch/pytorch/issues/58005
    # only should use torch.distributed.nn.all_gather if we implement a `local_loss`
    # like: https://github.com/mlfoundations/open_clip/issues/616
    world_size = get_world_size()
    if world_size == 1:
        return t

    if t.ndim == 0:
        t = t.unsqueeze(0)

    gathered = [torch.empty_like(t) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, t)
    gathered[get_rank()] = t
    return torch.cat(gathered, dim=0)


def gather_sum(t: torch.Tensor) -> torch.Tensor:
    # torch.distributed.nn.all_gather scales by world size since the reduce op is SUM
    # https://github.com/pytorch/pytorch/issues/58005
    # only should use torch.distributed.nn.all_gather if we implement a `local_loss`
    # like: https://github.com/mlfoundations/open_clip/issues/616
    world_size = get_world_size()
    if world_size == 1:
        return t

    if t.ndim == 0:
        t = t.unsqueeze(0)

    gathered = [torch.empty_like(t) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, t)
    gathered = torch.stack(gathered, dim=0)
    return gathered.sum(dim=0) # Sum across workers


def get_num_proc() -> int:
    world_size: int = get_world_size()
    try:
        # os.sched_getaffinity respects schedulers, unlike cpu_count(), but it's only available
        # on some Unix platforms, so we support both!
        return len(os.sched_getaffinity(0)) // world_size  # type: ignore[attr-defined]
    except AttributeError:
        return multiprocessing.cpu_count() // world_size


def torch_main_worker_finish_first(func: Callable):
    def wrapper(*args, **kwargs):
        # Get local rank (need to support non-DDP).
        try:
            local_rank = torch.distributed.get_rank()
            ddp_enabled = True
        except (RuntimeError, ValueError):
            local_rank = -1
            ddp_enabled = False
        is_main_worker = local_rank <= 0
        # Run on main worker first.
        if is_main_worker:
            result = func(*args, **kwargs)
        # Then everyone waits.
        if ddp_enabled:
            torch.distributed.barrier()
        # Run on other workers now.
        if not is_main_worker:
            result = func(*args, **kwargs)
        # Now everyone waits again.
        if ddp_enabled:
            torch.distributed.barrier()
        return result

    return wrapper


def print0(*args, **kwargs) -> None:
    if get_rank() == 0:
        print(*args, **kwargs)


def verify_ddp_weights_equal(model: torch.nn.Module, atol: float = 1e-5) -> None:
    if hasattr(model, "module"):
        model = model.module
    
    world_size = get_world_size()

    if world_size > 8:
        print0(f"[verify_ddp_weights_equal] Skipping with world_size={world_size} ⚠️")
        return

    for name, param in model.named_parameters():
        if param is None: continue
        if param.grad is None: 
            print0(f"[verify_ddp_weights_equal] Skipping param [{name}] with no grad")
            continue
        gathered_param = gather(param).reshape((world_size, -1))
        absolute_diffs = (gathered_param[None, 0, :] - gathered_param).abs()
        rank_params_eq = (absolute_diffs < atol).all()
        assert rank_params_eq, f"❌ param [{name}] not equal - got max_absolute_diff={absolute_diffs.max()}"
        ###################################################################################################################
        gathered_param_grad = gather(param.grad).reshape((world_size, -1))
        absolute_grad_diffs = (gathered_param_grad[None, 0, :] - gathered_param_grad).abs()
        rank_grad_params_eq = (absolute_grad_diffs < atol).all()
        assert rank_grad_params_eq, f"❌ param [{name}] grad not equal - got max_absolute_diff={absolute_grad_diffs.max()}"
        ###################################################################################################################
        
    
    print0("[verify_ddp_weights_equal] Verified DDP parameter correctness ✅")
    