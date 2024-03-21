import functools
import os

import datasets
import numpy as np
import torch
import transformers

from .misc import tqdm_if_main_worker


class DenseEncoder(torch.nn.Module):
    def __init__(self, model_name_or_path: str, max_seq_length = 128):
        super().__init__()
        self.encoder = transformers.AutoModel.from_pretrained(model_name_or_path)
        if hasattr(self.encoder, "encoder"):
            print("[DE] taking encoder from encoder-decoder")
            self.encoder = self.encoder.encoder
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
        self.gpu_count = torch.cuda.device_count()

        self.encoder.eval()
        self.encoder.cuda()

        if self.gpu_count > 1:
            self.encoder = torch.nn.DataParallel(self.encoder)
            print(f"[DE] Wrapped DenseEncoder in {self.gpu_count} GPUs")
        
        self.max_length = max_seq_length

    @torch.no_grad()
    def encode(self, dataset: datasets.Dataset, col: str, batch_size: int) -> np.ndarray:
        """ Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """
        os.environ["TOKENIZERS_PARALLELISM"] = "1"
        dataset = dataset.map(
            functools.partial(tokenize_transform_func, self.tokenizer, col=col, max_length=self.max_length),
            batch_size=10_000,
            batched=True,
            num_proc=None,
            # num_proc=len(os.sched_getaffinity(0)),
            keep_in_memory=False,
            remove_columns=[col],
            desc=f"Tokenizing {col}"
        )

        data_collator = transformers.DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=8)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size * self.gpu_count,
            shuffle=False,
            drop_last=False,
            num_workers=self.gpu_count, # 2 # 2 * self.gpu_count,
            collate_fn=data_collator,
            persistent_workers=True,
            pin_memory=True
        )

        encoded_embeds = []
        pbar = tqdm_if_main_worker(
            data_loader, desc='encoding', 
            disable=(dataset.num_rows < 128)
        )
        for batch_dict in pbar:
            batch_dict = move_to_cuda(batch_dict)

            # with torch.cuda.amp.autocast():
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

def embed_with_cache(model_name: str, cache_name: str, d: datasets.Dataset, 
                     col: str, save_to_disk: bool = True,
                     model=None, batch_size: int = 4096) -> datasets.Dataset:
    embedder_cache_path = model_name.replace('/', '__')
    # cache_folder = datasets.config.HF_DATASETS_CACHE
    cache_folder = os.path.join(get_tti_cache_dir(), 'corpus_embeddings', embedder_cache_path)
    os.makedirs(cache_folder, exist_ok=True)
    cache_path = os.path.join(cache_folder, cache_name) #  + "_small")

    if os.path.exists(cache_path):
        print("[embed_with_cache] Loading embeddings at path:", cache_path)
        d = datasets_fast_load_from_disk(cache_path)
        d.set_format("pt")
        return d
    
    d.set_format(type=None, columns=[col]) # get python objects (strings!)
    all_other_colums = [c for c in d.column_names if c != col]
    d_subset = d.remove_columns(all_other_colums)

    print(f"[embed_with_cache] encoding {len(d)} with batch size {batch_size}")
    if model is None:
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
