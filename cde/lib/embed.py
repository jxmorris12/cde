from typing import Dict, List, Mapping, Optional, Union

import functools
import os
import random

import datasets
import numpy as np
import torch
import transformers

from cde.lib.misc import (
    datasets_fast_load_from_disk, 
    get_cde_cache_dir,
    tqdm_if_main_worker,
)
from cde.lib.tensor import (
    mean_pool,
)


fingerprint_rng = random.Random()
def generate_random_fingerprint(nbits: int = 64) -> str:
    return f"{fingerprint_rng.getrandbits(nbits):0{nbits//4}x}"

def embed_dataloader(
        encoder, 
        data_loader, col: str, 
        show_progress_bar: bool = True, 
        leave_progress_bar: bool = True, 
        convert_to_tensor: bool = True,
        output_device: str = "cuda",
        output_dtype: torch.dtype = torch.float32,
        **kwargs
    ) -> List[torch.Tensor]:
    encoded_embeds = []
    pbar = tqdm_if_main_worker(
        data_loader, desc=f"Encoding {col}", 
        disable=(not show_progress_bar),
        leave=leave_progress_bar,
    )
    embed_device = ("cuda" if torch.cuda.is_available() else "cpu")
    for batch_dict in pbar:
        if "token_type_ids" in batch_dict:
            batch_dict.pop("token_type_ids")
        batch_dict = move_to_cuda(batch_dict)
        # TODO: Intelligently look into encoder kwargs and delete unmatching ones.
        for unwanted_key in ["prompt_name", "request_qid"]:
            if unwanted_key in kwargs:
                kwargs.pop(unwanted_key)
        with torch.autocast(embed_device, dtype=torch.bfloat16), torch.no_grad():
            outputs = encoder(**batch_dict, **kwargs)
        if hasattr(outputs, 'pooler_output') and not (outputs.pooler_output is None):
            embeds = outputs.pooler_output
        elif hasattr(outputs, 'last_hidden_state'):
            embeds = mean_pool(
                hidden_states=outputs.last_hidden_state,
                attention_mask=batch_dict['attention_mask']
            )
        else:
            # already pooled
            embeds = outputs
        
        encoded_embeds.append(embeds.cpu())
        
    if convert_to_tensor:
        return torch.cat(encoded_embeds, dim=0).to(output_device).to(output_dtype)
    else:
        encoded_embeds = [embeds.cpu().float().numpy() for embeds in encoded_embeds]
        output_array = np.concatenate(encoded_embeds, axis=0)
        return output_array


def move_to_cuda(sample):
    if len(sample) == 0:
        return {}
    elif not torch.cuda.is_available():
        return sample

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda(non_blocking=True)
        elif isinstance(maybe_tensor, dict):
            return {key: _move_to_cuda(value) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return tuple([_move_to_cuda(x) for x in maybe_tensor])
        elif isinstance(maybe_tensor, Mapping):
            return type(maybe_tensor)({k: _move_to_cuda(v) for k, v in maybe_tensor.items()})
        else:
            return maybe_tensor

    return _move_to_cuda(sample)


def tokenize_transform(
        examples: Dict[str, List],
        prefix: str,
        col: str,
        max_length: int,
        max_num_chars: int,
        tokenizer: transformers.PreTrainedTokenizer,
    ) -> Dict[str, torch.Tensor]:
    texts = examples[col]
    batch_dict = tokenizer(
        [prefix + t[:max_num_chars] for t in texts],
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    other_cols = [k for k in examples.keys() if k != col]
    return {**batch_dict, **{k: examples[k] for k in other_cols}}

class DenseEncoder(torch.nn.Module):
    def __init__(
            self, 
            model_name_or_path: str = "", 
            max_seq_length = 128, 
            encoder: Optional[torch.nn.Module] = None,
            query_prefix: str = "",
            document_prefix: str = "",
            normalize_embeds: bool = False,
            default_doc_prefix: bool = False,
        ):
        super().__init__()
        self.encoder = encoder
        self.model_name_or_path = model_name_or_path
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name_or_path,
            padding_side="right",
        )
        if not (self.tokenizer.pad_token) and self.tokenizer.bos_token:
            self.tokenizer.pad_token = self.tokenizer.bos_token
        self.tokenizer.add_eos_token = True
        self.gpu_count = torch.cuda.device_count()
        self.model_is_on_device = False
        self.max_length = max_seq_length
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.model_kwargs = {}
        self._max_num_chars = (self.max_length * 16)
        self.normalize_embeds = normalize_embeds
        self.default_doc_prefix = default_doc_prefix

    def _consider_putting_model_on_device(self) -> None:
        if self.model_is_on_device:
            return
        if self.encoder is None:
            print("[DenseEncoder] initializing from", self.model_name_or_path)
            self.encoder = transformers.AutoModel.from_pretrained(
                self.model_name_or_path, 
                # torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            self.encoder.eval()
        if hasattr(self.encoder, "decoder") and hasattr(self.encoder, "encoder"):
            print("[DenseEncoder] taking encoder from an encoder-decoder model")
            self.encoder = self.encoder.encoder
        if self.gpu_count > 0:
            self.encoder.cuda()

        if self.gpu_count > 1:
            print(f"[DenseEncoder] Wrapped DenseEncoder in {self.gpu_count} GPUs")
            self.encoder = torch.nn.DataParallel(self.encoder)
        self.model_is_on_device = True
    
    def encode_corpus(self, corpus: List[Dict[str, str]], *args, **kwargs) -> np.ndarray:
        if isinstance(corpus, list) and isinstance(corpus[0], dict) and 'text' in corpus[0]:
            passages = ['{} {}'.format(doc.get('title', ''), doc['text']).strip() for doc in corpus]
        else:
            passages = corpus
        return self.encode(dataset=passages, *args, **kwargs, prefix=self.document_prefix)

    def encode_queries(self, query_list, **kwargs) -> np.ndarray:
        return self.encode(
            dataset=query_list, **kwargs, prefix=self.query_prefix
        )

    def _create_dataloader(
            self, 
            dataset: Union[list[str], datasets.Dataset], 
            col: str,
            prefix: str,
            batch_size: int,
            num_workers: Optional[int] = None,
        ) -> torch.utils.data.DataLoader:
        if isinstance(dataset, list):
            if isinstance(dataset[0], str):
                dataset = datasets.Dataset.from_dict({ col: dataset }) #, fingerprint=str(uuid.uuid4()))
            elif isinstance(dataset[0], dict):
                dataset = datasets.Dataset.from_list(dataset)
            else:
                raise RuntimeError("unknown dataset format to encode")
        
        print("_create_dataloader() called with batch size:", batch_size, "and num_workers:", num_workers)
        os.environ["TOKENIZERS_PARALLELISM"] = "0"

        tokenize_fn = functools.partial(
            tokenize_transform, 
            prefix=prefix,
            col=col, 
            max_length=self.max_length,
            max_num_chars=self._max_num_chars,
            tokenizer=self.tokenizer,
        )
        if isinstance(dataset, datasets.IterableDataset):
            dataset = dataset.map(tokenize_fn, batched=True)
            dataset = dataset.remove_columns([col])
        else:
            dataset.set_transform(tokenize_fn)
        data_collator = transformers.DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=8)
        effective_batch_size = (batch_size * max(1, self.gpu_count))
        if num_workers is None:
            num_workers = min(len(os.sched_getaffinity(0)), self.gpu_count * 4)

        try:
            len_dataset = len(dataset)
            effective_batch_size = min(effective_batch_size, len(dataset) // max(1, num_workers))
        except TypeError:
            # iterable dataset has no len
            len_dataset = "unknown"
            effective_batch_size = batch_size
        effective_batch_size = max(1, effective_batch_size)

        print(f"[DenseEncoder] created dataloader with {num_workers} workers, effective_batch_size={effective_batch_size}, len(dataset)={len_dataset}")
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=effective_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers, 
            prefetch_factor=(4 if num_workers > 0 else None),
            collate_fn=data_collator,
            pin_memory=True
        )
    
    @torch.no_grad()
    def encode(
            self, 
            dataset: Union[list[str], datasets.Dataset], 
            col: str = "text", 
            batch_size: int = 256, 
            prefix: str = "", 
            show_progress_bar: bool = True,
            convert_to_tensor: bool = True,
            output_device: str = "cuda",
            output_dtype: torch.dtype = torch.float32,
            num_workers: Optional[int] = None,
            **kwargs,
        ) -> Union[torch.Tensor, np.ndarray]:
        """ Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """
        if not len(prefix) and self.default_doc_prefix:
            prefix = self.document_prefix
        
        data_loader = self._create_dataloader(
            dataset=dataset,
            col=col,
            batch_size=batch_size,
            prefix=prefix,
            num_workers=num_workers,
        )
        self._consider_putting_model_on_device()

        try: 
            len_dataset = len(dataset)
        except TypeError:
            len_dataset = 0

        show_progress_bar = (len_dataset >= 128) and show_progress_bar
        print(f"[DenseEncoder] encode() calling embed_dataloader (len(dataset)={len_dataset}, convert_to_tensor={convert_to_tensor}, output_device={output_device})")
        encoded_embeds = embed_dataloader(
            self.encoder,
            data_loader, 
            col=col,
            convert_to_tensor=convert_to_tensor,
            output_dtype=output_dtype,
            show_progress_bar=show_progress_bar,
            output_device=output_device,
            **kwargs,
            **self.model_kwargs
        )
        print("[DenseEncoder] encode() done calling embed_dataloader")
        if isinstance(dataset, datasets.Dataset):
            num_cleaned_cache_files = dataset.cleanup_cache_files()
            if num_cleaned_cache_files: 
                print(f"[DenseEncoder] cleaned {num_cleaned_cache_files} files")
        
        if self.normalize_embeds:
            encoded_embeds = torch.nn.functional.normalize(encoded_embeds, p=2, dim=-1)

        return encoded_embeds


def embed_with_cache(
        model_name: str, 
        cache_name: str, 
        d: datasets.Dataset,              
        col: str, 
        save_to_disk: bool = True,
        batch_size: int = 4096,
        max_seq_length: int = 512,
        model: Optional[DenseEncoder] = None,
        normalize_embeds: bool = False,
        prefix: str = "",
    ) -> datasets.Dataset:
    embedder_cache_path = model_name.replace('/', '__')
    cache_folder = os.path.join(get_cde_cache_dir(), 'corpus_embeddings', embedder_cache_path)
    os.makedirs(cache_folder, exist_ok=True)
    cache_path = os.path.join(cache_folder, cache_name) #  + "_small")

    if os.path.exists(cache_path):
        print("[embed_with_cache] Loading embeddings at path:", cache_path)
        d = datasets_fast_load_from_disk(cache_path)
        d.set_format("pt")
        return d
    
    print("[embed_with_cache] Will save embeddings to path:", cache_path)
    d.set_format(type=None, columns=[col]) # get python objects (strings!)
    all_other_colums = [c for c in d.column_names if c != col]
    d_subset = d.remove_columns(all_other_colums)

    print(f"[embed_with_cache] encoding {len(d)} with batch size {batch_size}")
    if model is None:
        model = DenseEncoder(
            model_name, 
            max_seq_length=max_seq_length,
            normalize_embeds=normalize_embeds,
        )
    embeddings = model.encode(
        d_subset, 
        col, 
        batch_size=batch_size, 
        output_device="cpu",
        prefix=prefix,
    )

    print(f"[embed_with_cache] creating datasets")

    if not save_to_disk:
        print("[embed_with_cache] returning dict")
        return { "embeds": embeddings }

    datasets_list = []
    max_dataset_size = 10_000_000
    i = 0
    while i < len(embeddings):
        dataset = datasets.Dataset.from_dict({
            "embeds": embeddings[i : i + max_dataset_size] 
        })
        datasets_list.append(dataset)
        i += max_dataset_size
    
    print("[embed_with_cache] concatenating datasets")
    d = datasets.concatenate_datasets(datasets_list)
    d.set_format("pt")
    print("[embed_with_cache] saving datasets")
    d.save_to_disk(cache_path)
    return d
