from typing import Dict, List, Mapping

import functools
import os

import datasets
import numpy as np
import torch
import transformers

from lib.misc import (
    datasets_fast_load_from_disk, 
    get_tti_cache_dir,
    tqdm_if_main_worker,
)
from lib.tensor import (
    mean_pool,
)


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


class DenseEncoder(torch.nn.Module):
    def __init__(self, model_name_or_path: str, max_seq_length = 128):
        super().__init__()
        self.encoder = None
        self.model_name_or_path = model_name_or_path
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name_or_path)
        self.gpu_count = torch.cuda.device_count()

        self.max_length = max_seq_length
        self._model_is_on_device = False

    def tokenize_transform(
            self,
            examples: Dict[str, List],
            col: str,
            max_length: int):
        texts = examples[col]
        batch_dict = self.tokenizer(
            texts,
            max_length=max_length,
            padding=True,
            truncation=True
        )
        return batch_dict

    def _put_model_on_device(self) -> None:
        self.encoder = transformers.AutoModel.from_pretrained(
            self.model_name_or_path, torch_dtype=torch.float16)
        self.encoder.eval()
        if hasattr(self.encoder, "encoder"):
            print("[DE] taking encoder from encoder-decoder")
            self.encoder = self.encoder.encoder
        if self.gpu_count > 0:
            self.encoder.cuda()

        if self.gpu_count > 1:
            self.encoder = torch.nn.DataParallel(self.encoder)
            print(f"[DE] Wrapped DenseEncoder in {self.gpu_count} GPUs")
        self._model_is_on_device = True

    @torch.no_grad()
    def encode(self, dataset: datasets.Dataset, col: str, batch_size: int) -> np.ndarray:
        """ Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """

        use_threads = False
        num_cpus = len(os.sched_getaffinity(0))
        if use_threads:
            os.environ["RAYON_NUM_THREADS"] = str(num_cpus)
            os.environ["RAYON_RS_NUM_THREADS"] = str(num_cpus)
            os.environ["TOKENIZERS_PARALLELISM"] = "1"
            dataset = dataset.map(
                functools.partial(
                    self.tokenize_transform, 
                    col=col, 
                    max_length=self.max_length
                ),
                batch_size=100_000,
                batched=True,
                keep_in_memory=False,
                remove_columns=[col],
                desc=f"Tokenizing [{col}]"
            )
        else:
            os.environ["TOKENIZERS_PARALLELISM"] = "0"
            print(f"[embed_with_cache] tokenizing with {num_cpus} processes")
            dataset = dataset.map(
                functools.partial(
                    self.tokenize_transform, 
                    col=col, 
                    max_length=self.max_length
                ),
                batch_size=10_000,
                batched=True,
                num_proc=num_cpus,
                keep_in_memory=False,
                remove_columns=[col],
                desc=f"Tokenizing {col}"
            )

        if not self._model_is_on_device:
            self._put_model_on_device()

        data_collator = transformers.DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=8)
        num_workers = max(1, self.gpu_count)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size * num_workers,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers, 
            collate_fn=data_collator,
            persistent_workers=True,
            pin_memory=True
        )

        encoded_embeds = []
        pbar = tqdm_if_main_worker(
            data_loader, desc=f"Encoding {col}", 
            disable=(dataset.num_rows < 128)
        )
        for batch_dict in pbar:
            batch_dict = move_to_cuda(batch_dict)
            outputs = self.encoder(**batch_dict)
            if hasattr(outputs, 'pooler_output'):
                embeds = outputs.pooler_output
            else:
                embeds = mean_pool(
                    hidden_states=outputs.last_hidden_state,
                    attention_mask=batch_dict['attention_mask']
                )
            embeds = torch.nn.functional.normalize(embeds, p=2, dim=-1)
            encoded_embeds.append(embeds.cpu().numpy())

        num_cleaned_cache_files = dataset.cleanup_cache_files()
        print(f"cleaned {num_cleaned_cache_files} files")
        return np.concatenate(encoded_embeds, axis=0)


def embed_with_cache(
        model_name: str, 
        cache_name: str, 
        d: datasets.Dataset,              
        col: str, 
        save_to_disk: bool = True,
        batch_size: int = 4096
    ) -> datasets.Dataset:
    embedder_cache_path = model_name.replace('/', '__')
    cache_folder = os.path.join(get_tti_cache_dir(), 'corpus_embeddings', embedder_cache_path)
    os.makedirs(cache_folder, exist_ok=True)
    cache_path = os.path.join(cache_folder, cache_name) #  + "_small")

    if os.path.exists(cache_path):
        # print("[embed_with_cache] Loading embeddings at path:", cache_path)
        d = datasets_fast_load_from_disk(cache_path)
        d.set_format("pt")
        return d
    
    d.set_format(type=None, columns=[col]) # get python objects (strings!)
    all_other_colums = [c for c in d.column_names if c != col]
    d_subset = d.remove_columns(all_other_colums)

    print(f"[embed_with_cache] encoding {len(d)} with batch size {batch_size}")
    model = DenseEncoder(model_name, max_seq_length=128)
    embeddings = model.encode(d_subset, col, batch_size=batch_size)

    print(f"[embed_with_cache] creating datasets")

    if not save_to_disk:
        return { "embeds": torch.tensor(embeddings) }

    datasets_list = []
    max_dataset_size = 8_000_000
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
    d.save_to_disk(cache_path)
    return d
