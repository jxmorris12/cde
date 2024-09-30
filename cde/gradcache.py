from typing import Any, Callable, List, Optional, Union, Tuple
import typing

import functools
import logging
from collections import UserDict
from contextlib import nullcontext
from itertools import repeat

import accelerate
import torch
from torch import Tensor, nn
from torch.utils.checkpoint import get_device_states, set_device_states
from torch.distributed.fsdp import FullyShardedDataParallel

from cde.lib.dist import get_rank, get_world_size
from cde.lib.misc import tqdm_if_main_worker

min_tqdm_inputs = 8

class RandContext:
    def __init__(self, *tensors):
        self.fwd_cpu_state = torch.get_rng_state()
        self.fwd_gpu_devices, self.fwd_gpu_states = get_device_states(*tensors)

    def __enter__(self):
        self._fork = torch.random.fork_rng(devices=self.fwd_gpu_devices, enabled=True)
        self._fork.__enter__()
        torch.set_rng_state(self.fwd_cpu_state)
        set_device_states(self.fwd_gpu_devices, self.fwd_gpu_states)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._fork.__exit__(exc_type, exc_val, exc_tb)
        self._fork = None


logger = logging.getLogger(__name__)


class GradCache:
    """
    Gradient Cache class. Implements input chunking, first graph-less forward pass, Gradient Cache creation, second
    forward & backward gradient computation. Optimizer step is not included. Native torch automatic mixed precision is
    supported. User needs to handle gradient unscaling and scaler update after a gradient cache step.
    """

    def __init__(
        self,
        chunk_sizes: Union[int, List[int]],
        first_stage_chunk_sizes: Union[int, List[int]],
        loss_fn: Callable[..., Tensor],
        split_input_fn: Callable[[Any, int], Any] = None,
        get_rep_fn: Callable[..., Tensor] = None,
        accelerator: Optional[accelerate.Accelerator] = None,
        bf16: bool = False,
    ):
        """
        Initialize the Gradient Cache class instance.
        :param model: Single model with `forward_embedder` func
        :param chunk_sizes: An integer indicating chunk size. Or a list of integers of chunk size for each model.
        :param loss_fn: A loss function that takes arbitrary numbers of representation tensors and
        arbitrary numbers of keyword arguments as input. It should not in any case modify the input tensors' relations
        in the autograd graph, which are later relied upon to create the gradient cache.
        :param split_input_fn: An optional function that split generic model input into chunks. If not provided, this
        class will try its best to split the inputs of supported types. See `split_inputs` function.
        :param get_rep_fn: An optional function that takes generic model output and return representation tensors. If
        not provided, the generic output is assumed to be the representation tensor.
        :param bf16: If True, run mixed precision training
        """
        self.accelerator = accelerator
        self.chunk_sizes = chunk_sizes
        self.first_stage_chunk_sizes = first_stage_chunk_sizes

        self.split_input_fn = split_input_fn
        self.get_rep_fn = get_rep_fn
        self.loss_fn = loss_fn
        self.bf16 = bf16
        self._get_input_tensors_strict = False

    def __call__(self, *args, model_stages: Optional[Tuple[nn.Module, nn.Module]] = None, **kwargs):
        """
        Call the cache_step function.
        :return: Current step loss.
        """
        model = kwargs["model"]
        model_is_ddp = (
            isinstance(model, nn.parallel.DistributedDataParallel)
            or
            isinstance(model, FullyShardedDataParallel)
        )
        if model_is_ddp:
            model_has_two_stages = hasattr(model.module, "second_stage_model")
        else:
            model_has_two_stages = hasattr(model, "second_stage_model")

        if model_has_two_stages:
            return self.cache_step_two_stage(*args, model_stages=model_stages, **kwargs)
        else:
            return self.cache_step_one_stage(*args, **kwargs)

    def split_inputs(self, model_input, chunk_size: int) -> List:
        """
        Split input into chunks. Will call user provided `split_input_fn` if specified. Otherwise,
        it can handle input types of tensor, list of tensors and dictionary of tensors.
        :param model_input: Generic model input.
        :param chunk_size:  Size of each chunk.
        :return: A list of chunked model input.
        """

        # delegate splitting to user provided function
        if self.split_input_fn is not None:
            return self.split_input_fn(model_input, chunk_size)

        if isinstance(model_input, (dict, UserDict)) and all(
            isinstance(x, Tensor) for x in model_input.values()
        ):
            keys = list(model_input.keys())
            chunked_tensors = [model_input[k].split(chunk_size, dim=0) for k in keys]
            return [
                dict(zip(kk, tt)) for kk, tt in zip(repeat(keys), zip(*chunked_tensors))
            ]

        elif isinstance(model_input, list) and all(
            isinstance(x, Tensor) for x in model_input
        ):
            chunked_x = [t.split(chunk_size, dim=0) for t in model_input]
            return [list(s) for s in zip(*chunked_x)]

        elif isinstance(model_input, Tensor):
            return list(model_input.split(chunk_size, dim=0))

        elif isinstance(model_input, tuple) and list(map(type, model_input)) == [
            list,
            dict,
        ]:
            args_chunks = self.split_inputs(model_input[0], chunk_size)
            kwargs_chunks = self.split_inputs(model_input[1], chunk_size)
            return list(zip(args_chunks, kwargs_chunks))

        else:
            raise NotImplementedError(
                f"Model input split not implemented for type {type(model_input)}"
            )

    def get_input_tensors(self, model_input) -> List[Tensor]:
        """
        Recursively go through model input and grab all tensors, which are then used to record current device random
        states. This method will do its best to parse types of Tensor, tuple, list, dict and UserDict. Other types will
        be ignored unless self._get_input_tensors_strict is set to True, in which case an exception will be raised.
        :param model_input: input to model
        :return: all torch tensors in model_input
        """
        if isinstance(model_input, Tensor):
            return [model_input]

        elif isinstance(model_input, (list, tuple)):
            return sum((self.get_input_tensors(x) for x in model_input), [])

        elif isinstance(model_input, (dict, UserDict)):
            return sum((self.get_input_tensors(x) for x in model_input.values()), [])

        elif self._get_input_tensors_strict:
            raise NotImplementedError(
                f"get_input_tensors not implemented for type {type(model_input)}"
            )
        else:
            return []

    def model_call(self, model: nn.Module, model_input):
        """
        Literally call the model's __call__ method.
        :param model: model to be called
        :param model_input: input to the model call
        :return: model output
        """
        with torch.autocast(device_type="cuda") if self.bf16 else nullcontext():
            if isinstance(model_input, Tensor):
                return model(model_input)
            elif isinstance(model_input, list):
                return model(*model_input)
            elif isinstance(model_input, (dict, UserDict)):
                return model(**model_input)
            elif isinstance(model_input, tuple) and list(map(type, model_input)) == [
                list,
                dict,
            ]:
                model_args, model_kwargs = model_input
                return model(*model_args, **model_kwargs)
            else:
                raise NotImplementedError

    def get_reps(self, model_out) -> Tensor:
        """
        Return representation tensor from generic model output
        :param model_out: generic model output
        :return: a single tensor corresponding to the model representation output
        """
        if self.get_rep_fn is not None:
            return self.get_rep_fn(model_out)
        else:
            return model_out

    def forward_no_grad(
        self,
        model: nn.Module,
        model_inputs,
    ) -> Union[Tensor, List[RandContext]]:
        """
        The first forward pass without gradient computation.
        :param model: Encoder model.
        :param model_inputs: Model input already broken into chunks.
        :return: A tuple of a) representations and b) recorded random states.
        """
        rnd_states = []
        model_reps = []

        with torch.no_grad():
            for x in model_inputs:
                rnd_states.append(RandContext(*self.get_input_tensors(x)))
                y = self.model_call(model, x)
                model_reps.append(self.get_reps(y))

        # concatenate all sub-batch representations
        model_reps = torch.cat(model_reps, dim=0)
        return model_reps, rnd_states

    @typing.no_type_check
    def build_cache(self, backward_fn, *reps: Tensor, **loss_kwargs) -> Union[List[Tensor], Tensor]:
        """
        Compute the gradient cache
        :param reps: Computed representations from all encoder models
        :param loss_kwargs: Extra keyword arguments to the loss function
        :return: A tuple of a) gradient cache for each encoder model, and b) loss tensor
        """
        reps = [r.detach().requires_grad_() for r in reps]
        loss = self.loss_fn(*reps, **loss_kwargs)

        backward_fn(loss)

        cache = [r.grad for r in reps]

        return cache, loss.detach()

    @typing.no_type_check
    def forward_backward_one_stage(
        self,
        model: nn.Module,
        backward_fn: Callable,
        model_inputs,
        cached_gradients: List[Tensor],
        random_states: List[RandContext],
        no_sync_except_last: bool = False,
    ) -> None:
        """
        Run the second forward and the backward pass to compute gradient for a model.
        :param model: Encoder model.
        :param model_inputs: Chunked input to the encoder model.
        :param cached_gradients: Chunked gradient cache tensor for each input.
        :param random_states: Each input's device random state during the first forward.
        :param no_sync_except_last: If True, under distributed setup, only trigger gradient reduction across processes
        for the last sub-batch's forward-backward pass.
        """
        if len(model_inputs) > min_tqdm_inputs:
            model_inputs = tqdm_if_main_worker(
                model_inputs,
                desc="computing first stage gradients",
                colour="#C7D3BF",
                leave=False
            )
        
        # https://huggingface.co/docs/accelerate/en/concept_guides/gradient_synchronization
        if (get_world_size() > 0) and no_sync_except_last:
            no_sync_func = model.no_sync if hasattr(model, "no_sync") else self.accelerator.no_sync
            sync_contexts = [
                no_sync_func for _ in range(len(model_inputs) - 1)
            ] + [nullcontext]
        else:
            sync_contexts = [nullcontext for _ in range(len(model_inputs))]

        for x, state, gradient, sync_context in zip(
            model_inputs, random_states, cached_gradients, sync_contexts
        ):
            with state, sync_context():
                y = self.model_call(model, x)
                reps = self.get_reps(y)
                surrogate = torch.dot(reps.flatten(), gradient.flatten())
                backward_fn(surrogate)  # [modified]
    
    def cache_step_one_stage(
        self, *model_inputs, model: nn.Module, no_sync_except_last: bool = False, backward_fn: Callable, run_backward: bool, **loss_kwargs
    ) -> Tensor:
        """
        Run a cached step to compute gradient over the inputs.
        :param model_inputs: Input to each encoder model. Should be in similar order as the class's model.
        :param no_sync_except_last: If True, under distributed setup, for each model, only trigger gradient reduction
        across processes for the last sub-batch's forward-backward pass.
        :param loss_kwargs: Additional keyword arguments to the loss function.
        :return: The current's loss.
        """
        model_inputs = [
            { 
                "input_ids": x["input_ids"], 
                "attention_mask": x["attention_mask"]
            }
            for x in model_inputs
        ]
        model_inputs = [
            self.split_inputs(x, chunk_size)
            for x, chunk_size in zip(model_inputs, self.chunk_sizes)
        ]

        all_reps = []
        all_rnd_states = []
        for _model, x in zip([model, model], model_inputs):
            model_reps, rnd_states = self.forward_no_grad(_model, x)
            all_reps.append(model_reps)
            all_rnd_states.append(rnd_states)

        cache, loss = self.build_cache(backward_fn, *all_reps, **loss_kwargs)
        if not run_backward:
            return loss
        
        cache = [c.split(chunk_size) for c, chunk_size in zip(cache, self.chunk_sizes)]

        for _model, x, model_cache, rnd_states in zip(
            [model, model], model_inputs, cache, all_rnd_states
        ):
            self.forward_backward_one_stage(
                _model,
                backward_fn,
                x,
                model_cache,
                rnd_states,
                no_sync_except_last=no_sync_except_last,
            )

        return loss

    def forward_backward_two_stage(
        self,
        model: nn.Module,
        backward_fn: Callable,
        model_inputs,
        dataset_embeddings: torch.Tensor,
        cached_gradients: List[Tensor],
        random_states: List[RandContext],
        no_sync_except_last: bool = False,
    ) -> None:
        """
        Run the second forward and the backward pass to compute gradient for a model.
        :param model: Encoder model.
        :param model_inputs: Chunked input to the encoder model.
        :param cached_gradients: Chunked gradient cache tensor for each input.
        :param random_states: Each input's device random state during the first forward.
        :param no_sync_except_last: If True, under distributed setup, only trigger gradient reduction across processes
        for the last sub-batch's forward-backward pass.
        """
        if (get_world_size() > 0) and no_sync_except_last:
            no_sync_func = model.no_sync if hasattr(model, "no_sync") else self.accelerator.no_sync
            sync_contexts = [
                no_sync_func for _ in range(len(model_inputs) - 1)
            ] + [nullcontext]
        else:
            sync_contexts = [nullcontext for _ in range(len(model_inputs))]
        if len(model_inputs) > min_tqdm_inputs:
            model_inputs_tqdm = tqdm_if_main_worker(
                model_inputs,
                desc="computing second stage gradients",
                leave=False,
                colour="#807182",
            )
        else:
            model_inputs_tqdm = model_inputs
        
        for x, state, gradient, sync_context in zip(
            model_inputs_tqdm, random_states, cached_gradients, sync_contexts
        ):
            with state, sync_context(), (torch.autocast(device_type="cuda") if self.bf16 else nullcontext()):
                y = model(
                    input_ids=x["input_ids"],
                    attention_mask=x["attention_mask"],
                    dataset_embeddings=dataset_embeddings,
                )
            reps = self.get_reps(y)
            surrogate = torch.dot(reps.flatten(), gradient.flatten())
            backward_fn(surrogate)  # [added]

    def cache_step_two_stage(
        self, 
            *model_inputs, 
            model: nn.Module, no_sync_except_last: bool = False, 
            backward_fn: Callable, run_backward: bool, 
            model_stages: Optional[Tuple[nn.Module, nn.Module]] = None,
            **loss_kwargs
    ) -> Tensor:
        """
        Run a cached step to compute gradient over the inputs.
        :param model_inputs: Input to each encoder model. Should be in similar order as the class's model.
        :param no_sync_except_last: If True, under distributed setup, for each model, only trigger gradient reduction
        across processes for the last sub-batch's forward-backward pass.
        :param loss_kwargs: Additional keyword arguments to the loss function.
        :return: The current's loss.
        """
        # TODO: Pass these keys to constructor as
        # `first_stage_data_keys` and `second_stage_data_keys` or
        # something like that.
        first_chunk_size_scale = 1
        first_stage_model_inputs = [
            { 
                "dataset_input_ids": x["dataset_input_ids"], 
                "dataset_attention_mask": x["dataset_attention_mask"]
            }
            for x in model_inputs
        ]
        # TODO: Argparse for different chunk sizes for first and second stage.
        # For now we just use the heuristic that the first stage can take ~2x
        # the chunk size.
        first_stage_model_inputs = [
            self.split_inputs(x, chunk_size * first_chunk_size_scale)
            for x, chunk_size in zip(first_stage_model_inputs, self.first_stage_chunk_sizes)
        ]
        second_stage_model_inputs = [
            { 
                "input_ids": x["input_ids"], 
                "attention_mask": x["attention_mask"]
            }
            for x in model_inputs
        ]
        second_stage_model_inputs = [
            self.split_inputs(x, chunk_size)
            for x, chunk_size in zip(second_stage_model_inputs, self.chunk_sizes)
        ] # List of dicts where sub-batch size in each dict is at most chunk size

        ##
        ## First stage no grad
        ##
        model_is_ddp = (
            isinstance(model, nn.parallel.DistributedDataParallel)
            or
            isinstance(model, FullyShardedDataParallel)
        )
        assert model_stages is not None
        first_stage_model, second_stage_model = model_stages

        first_stage_input_chunks = first_stage_model_inputs[0]

        first_stage_input_chunks_tqdm = first_stage_input_chunks
        if len(first_stage_input_chunks) > min_tqdm_inputs:
            first_stage_input_chunks_tqdm = tqdm_if_main_worker(
                first_stage_input_chunks_tqdm,
                desc="computing first stage outputs",
                leave=False,
                colour="MAGENTA"
            )

        first_stage_rnd_states = []
        first_stage_embedding = []
        with torch.no_grad():
            for x in first_stage_input_chunks_tqdm:
                first_stage_rnd_states.append(RandContext(*self.get_input_tensors(x)))
                # print(f"calling first_stage_model of type {type(first_stage_model)}")
                with torch.autocast(device_type="cuda") if self.bf16 else nullcontext():
                    y = first_stage_model(
                        input_ids=x['dataset_input_ids'],
                        attention_mask=x['dataset_attention_mask'],
                    )
                first_stage_embedding.append(self.get_reps(y))
        # concatenate all sub-batch representations
        first_stage_embedding = torch.cat(first_stage_embedding, dim=0)
        ##
        ## Second stage no grad
        ##
        # concatenate all sub-batch representations
        output_embeddings = []
        second_stage_rnd_states = []
        for _model, second_stage_input_chunks in zip([second_stage_model, second_stage_model], second_stage_model_inputs):
            second_stage_input_chunks_tqdm = second_stage_input_chunks
            if len(second_stage_input_chunks) > min_tqdm_inputs:
                second_stage_input_chunks_tqdm = tqdm_if_main_worker(
                    second_stage_input_chunks,
                    desc="computing second stage outputs",
                    leave=False,
                )
            model_reps = []
            rnd_states = []
            with torch.no_grad():
                for x in second_stage_input_chunks_tqdm:
                    rnd_states.append(RandContext(*self.get_input_tensors(x)))
                    with torch.autocast(device_type="cuda") if self.bf16 else nullcontext():
                        y = _model(
                            input_ids=x["input_ids"],
                            attention_mask=x["attention_mask"],
                            dataset_embeddings=first_stage_embedding,
                        )
                    model_reps.append(self.get_reps(y))
            # concatenate all sub-batch representations
            output_embeddings.append(torch.cat(model_reps, dim=0))
            second_stage_rnd_states.append(rnd_states)
        
        ################################################
        ###
        ### Using output representations, compute loss :)
        ###
        output_gradient_cache, loss = self.build_cache(
            backward_fn, 
            *output_embeddings, 
            **loss_kwargs
        )
        if not run_backward:
            # We also support a mode now where we don't actually compute gradients, we just
            # compute loss in this multi-stage process, which is used for consistent evaluation.
            return loss

        second_stage_gradients = [
            c.split(chunk_size) for c, chunk_size in 
            zip(output_gradient_cache, self.chunk_sizes)
        ]
        ################################################
        ###
        ### Compute gradients wrt second stage
        ###
        first_stage_embedding = first_stage_embedding.detach().requires_grad_()
        for _model, second_stage_input_chunks, random_states, cached_gradients in zip(
            [second_stage_model, second_stage_model], second_stage_model_inputs, second_stage_rnd_states, second_stage_gradients
        ):
            self.forward_backward_two_stage(
                model=_model,
                backward_fn=backward_fn,
                model_inputs=second_stage_input_chunks,
                dataset_embeddings=first_stage_embedding,
                cached_gradients=cached_gradients,
                random_states=random_states,
                no_sync_except_last=no_sync_except_last
            )
        ###
        ### Compute gradients wrt first stage
        ### 
        if no_sync_except_last and model_is_ddp:
            no_sync_func = model.no_sync if hasattr(model, "no_sync") else self.accelerator.no_sync
            sync_contexts = [
                no_sync_func for _ in range(len(model_inputs) - 1)
            ] + [nullcontext]
        else:
            sync_contexts = [nullcontext for _ in range(len(first_stage_input_chunks))]
        
        # TODO: There's a *2 here too; get that from argparsed value once we add argparse
        # support for "max_batch_size_first_stage" or something like that.
        first_stage_gradients = first_stage_embedding.grad.split(
            self.first_stage_chunk_sizes[0] * first_chunk_size_scale
        )
        first_stage_input_chunks_tqdm = first_stage_input_chunks
        if len(first_stage_input_chunks) > min_tqdm_inputs:
            first_stage_input_chunks_tqdm = tqdm_if_main_worker(
                first_stage_input_chunks,
                desc="computing first stage gradients",
                colour="#C7D3BF",
                leave=False
            )
        for x, state, gradient, sync_context in zip(
            first_stage_input_chunks, first_stage_rnd_states, first_stage_gradients, sync_contexts
        ):
            with state, sync_context(), (torch.autocast(device_type="cuda") if self.bf16 else nullcontext()):
                y = first_stage_model(
                    input_ids=x["dataset_input_ids"],
                    attention_mask=x["dataset_attention_mask"],
                )
            reps = self.get_reps(y)
            surrogate = torch.dot(reps.flatten(), gradient.flatten())
            backward_fn(surrogate)
        
        return loss