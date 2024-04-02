import logging
import typing
from collections import UserDict
from contextlib import nullcontext
from itertools import repeat
from typing import Any, Callable, List, Union

import torch
from torch import Tensor, nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.checkpoint import get_device_states, set_device_states


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
        model: nn.Module,
        chunk_sizes: Union[int, List[int]],
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable[..., Tensor],
        split_input_fn: Callable[[Any, int], Any] = None,
        get_rep_fn: Callable[..., Tensor] = None,
        fp16: bool = False,
        scaler: GradScaler = None,
        backward_fn=None,  # [added]
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
        :param fp16: If True, run mixed precision training, which requires scaler to also be set.
        :param scaler: A GradScaler object for automatic mixed precision training.
        :[added] param backward_fn: The `manual_backward` function of pytorch lightning trainer when automatic_optimization is disabled.
        """
        self.model = model
        self.models = [model.forward, model.forward]
        self.optimizer = optimizer

        if isinstance(chunk_sizes, int):
            self.chunk_sizes = [chunk_sizes for _ in range(len(self.models))]
        else:
            self.chunk_sizes = chunk_sizes

        self.split_input_fn = split_input_fn
        self.get_rep_fn = get_rep_fn
        self.loss_fn = loss_fn

        assert not fp16, "fp16 no longer supported! please use bf16."
        self.scaler = scaler
        # [added]
        self.backward_fn = backward_fn

        self._get_input_tensors_strict = False
        self.model_has_two_stages = hasattr(self.model, "forward_first_stage")

    def __call__(self, *args, **kwargs):
        """
        Call the cache_step function.
        :return: Current step loss.
        """
        if self.model_has_two_stages:
            return self.cache_step_two_stage(*args, **kwargs)
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
    def build_cache(self, *reps: Tensor, **loss_kwargs) -> Union[List[Tensor], Tensor]:
        """
        Compute the gradient cache
        :param reps: Computed representations from all encoder models
        :param loss_kwargs: Extra keyword arguments to the loss function
        :return: A tuple of a) gradient cache for each encoder model, and b) loss tensor
        """
        reps = [r.detach().requires_grad_() for r in reps]
        loss = self.loss_fn(*reps, **loss_kwargs)

        self.backward_fn(loss)
        
        self.optimizer.step()
        self.model.zero_grad()

        cache = [r.grad for r in reps]

        return cache, loss.detach()

    @typing.no_type_check
    def forward_backward_one_stage(
        self,
        model: nn.Module,
        model_inputs,
        cached_gradients: List[Tensor],
        random_states: List[RandContext],
        no_sync_except_last: bool = False,
    ):
        """
        Run the second forward and the backward pass to compute gradient for a model.
        :param model: Encoder model.
        :param model_inputs: Chunked input to the encoder model.
        :param cached_gradients: Chunked gradient cache tensor for each input.
        :param random_states: Each input's device random state during the first forward.
        :param no_sync_except_last: If True, under distributed setup, only trigger gradient reduction across processes
        for the last sub-batch's forward-backward pass.
        """
        if isinstance(
            model, nn.parallel.DistributedDataParallel
        ):  

            if no_sync_except_last:
                sync_contexts = [
                    model.no_sync for _ in range(len(model_inputs) - 1)
                ] + [nullcontext]
                sync_flags = [True] * (len(model_inputs))  # [added]
            else:
                sync_contexts = [nullcontext for _ in range(len(model_inputs))]
                sync_flags = [False] * (len(model_inputs))  # [added]

            # [modified]
            for x, state, gradient, sync_context, sync_flag in zip(
                model_inputs, random_states, cached_gradients, sync_contexts, sync_flags
            ):
                with sync_context():
                    with state:
                        y = self.model_call(model, x)
                    reps = self.get_reps(y)
                    surrogate = torch.dot(reps.flatten(), gradient.flatten())
                    if sync_flag:
                        model.require_backward_grad_sync = True
                    self.backward_fn(surrogate)  # [modified]
        else:  # [use base model (i.e. transformer)]
            for x, state, gradient in zip(
                model_inputs, random_states, cached_gradients
            ):
                with state:
                    y = self.model_call(model, x)
                reps = self.get_reps(y)
                surrogate = torch.dot(reps.flatten(), gradient.flatten())
                self.backward_fn(surrogate)  # [added]
    
    def cache_step_one_stage(
        self, *model_inputs, no_sync_except_last: bool = False, **loss_kwargs
    ) -> Tensor:
        """
        Run a cached step to compute gradient over the inputs.
        :param model_inputs: Input to each encoder model. Should be in similar order as the class's model.
        :param no_sync_except_last: If True, under distributed setup, for each model, only trigger gradient reduction
        across processes for the last sub-batch's forward-backward pass.
        :param loss_kwargs: Additional keyword arguments to the loss function.
        :return: The current's loss.
        """
        all_reps = []
        all_rnd_states = []

        model_inputs = [
            self.split_inputs(x, chunk_size)
            for x, chunk_size in zip(model_inputs, self.chunk_sizes)
        ]

        for model, x in zip(self.models, model_inputs):
            model_reps, rnd_states = self.forward_no_grad(model, x)
            all_reps.append(model_reps)
            all_rnd_states.append(rnd_states)

        cache, loss = self.build_cache(*all_reps, **loss_kwargs)
        cache = [c.split(chunk_size) for c, chunk_size in zip(cache, self.chunk_sizes)]

        for model, x, model_cache, rnd_states in zip(
            self.models, model_inputs, cache, all_rnd_states
        ):
            self.forward_backward_one_stage(
                model,
                x,
                model_cache,
                rnd_states,
                no_sync_except_last=no_sync_except_last,
            )

        return loss

    def forward_backward_two_stage(
        self,
        model: nn.Module,
        model_inputs,
        cached_gradients: List[Tensor],
        random_states: List[RandContext],
        no_sync_except_last: bool = False,
    ):
        """
        Run the second forward and the backward pass to compute gradient for a model.
        :param model: Encoder model.
        :param model_inputs: Chunked input to the encoder model.
        :param cached_gradients: Chunked gradient cache tensor for each input.
        :param random_states: Each input's device random state during the first forward.
        :param no_sync_except_last: If True, under distributed setup, only trigger gradient reduction across processes
        for the last sub-batch's forward-backward pass.
        """
        if isinstance(
            model, nn.parallel.DistributedDataParallel
        ):  

            if no_sync_except_last:
                sync_contexts = [
                    model.no_sync for _ in range(len(model_inputs) - 1)
                ] + [nullcontext]
                sync_flags = [True] * (len(model_inputs))  # [added]
            else:
                sync_contexts = [nullcontext for _ in range(len(model_inputs))]
                sync_flags = [False] * (len(model_inputs))  # [added]

            # [modified]
            for x, state, gradient, sync_context, sync_flag in zip(
                model_inputs, random_states, cached_gradients, sync_contexts, sync_flags
            ):
                with sync_context():
                    with state:
                        y = self.model_call(model, x)
                    reps = self.get_reps(y)
                    surrogate = torch.dot(reps.flatten(), gradient.flatten())
                    if sync_flag:
                        model.require_backward_grad_sync = True
                    self.backward_fn(surrogate)  # [modified]
        else:  # [use base model (i.e. transformer)]

            for x, state, gradient in zip(
                model_inputs, random_states, cached_gradients
            ):
                with nullcontext():
                    with state:
                        y = self.model_call(model, x)
                    reps = self.get_reps(y)
                    surrogate = torch.dot(reps.flatten(), gradient.flatten())
                    self.backward_fn(surrogate)  # [added]

    def cache_step_two_stage(
        self, *model_inputs, no_sync_except_last: bool = False, **loss_kwargs
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
            self.split_inputs(x, chunk_size)
            for x, chunk_size in zip(model_inputs, self.chunk_sizes)
        ]

        ##
        ## First stage no grad
        ##
        all_reps_1 = []
        all_rnd_states_1 = []
        for model, x in zip(self.models, model_inputs):
            model_reps = []
            with torch.no_grad():
                for x in model_inputs:
                    all_rnd_states_1.append(RandContext(*self.get_input_tensors(x)))
                    y = model.forward_first_stage(
                        **x
                    )
                    model_reps.append(self.get_reps(y))
            # concatenate all sub-batch representations
            all_reps_1.append(torch.cat(model_reps, dim=0))

        ##
        ## Second stage no grad
        ##
        # concatenate all sub-batch representations
        all_reps_2 = []
        all_rnd_states_2 = []
        for model, x in zip(self.models, model_inputs):
            model_reps = []
            with torch.no_grad():
                for x in model_inputs:
                    all_rnd_states_2.append(RandContext(*self.get_input_tensors(x)))
                    y = self.model_call(model, x)
                    model_reps.append(self.get_reps(y))
            # concatenate all sub-batch representations
            all_reps_2.append(torch.cat(model_reps, dim=0))

        cache, loss = self.build_cache(*all_reps_2, **loss_kwargs)
        cached_gradients_list = [
            c.split(chunk_size) for c, chunk_size in zip(cache, self.chunk_sizes)
        ]

        for model, x, cached_gradients, rnd_states in zip(
            self.models, model_inputs, cached_gradients_list, all_rnd_states_2
        ):
            self.forward_backward_two_stage(
                model,
                x,
                cached_gradients,
                rnd_states,
                no_sync_except_last=no_sync_except_last,
            )

        return loss