from typing import Dict, List, Optional, Tuple

import collections
import json
import io
import os
import pickle
import random

import datasets
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
import tqdm
from transformers import AutoTokenizer


# Data is within the root folder in data/
DATA_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")

class RetrievalDataset(Dataset):
    """This is the Super Class of all the other Dataset classes. It provides general 
       utilities for loading in the data and sampling random episodes.
    """
  
    def __init__(
        self, 
        dataset_name: str,
        model_name: str,
        token_max_length: int,
        split: str, 
        text_key: str,
        time_key: str,
        is_queries: bool = False,
        sanity: Optional[int] = None,
    ):
        """Initializes the Dataset class.

        Args:
            dataset_name (str): name of the dataset
            model_name (str): name of model (to load its tokenizer)
            token_max_length (int): Max number of tokens in a window
            split (str): Name of the split: train, validation, or test.
            text_key (str)
            time_key (str)
            is_queries (bool, optional): Whether or not we're reading in the queries.
            sanity (int | None): if set, use only this many authors (for debugging)
        """
        self.split = split
        assert self.split in ["train", "validation", "test"]
        
        self.dataset_name = dataset_name
        self.token_max_length = token_max_length
        self.is_queries = is_queries
        self.sanity = sanity
      
        self.dataset_path = os.path.join(DATA_PATH, self.dataset_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name) 
        
        self.text_key = text_key
        self.time_key = time_key

        self.min_char_length = 30

    def tokenize_text(self, all_text) -> Tuple[torch.Tensor, torch.Tensor]:
        tokenized_text = self.tokenizer(
            all_text, 
            padding=True if self.use_random_windows else "max_length", 
            truncation=False if self.use_random_windows else True, 
            max_length=None if self.use_random_windows else self.token_max_length, 
            return_tensors='pt'
        )
        return self.reformat_tokenized_inputs(tokenized_text)

    def reformat_tokenized_inputs(self, tokenized_text) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reformats the output from HugginFace Tokenizers.
        """
        if len(tokenized_text.keys()) == 3:
            # removes pesky 'token_type_ids'
            input_ids, _, attention_mask = tokenized_text.values()
        else:
            input_ids, attention_mask = tokenized_text.values()
        return [input_ids, attention_mask]
    
    def sample_random_window(self, data, window_length):
        """Samples a random window from the text.
        """
        input_ids, attention_mask = data

        cls = self.tokenizer.cls_token_id
        pad = self.tokenizer.pad_token_id
        eos = self.tokenizer.eos_token_id
        if type(eos) != int:
            eos = self.tokenizer.sep_token_id

        # Inputs are smaller than window size -> add padding
        padding = window_length - input_ids.shape[1]
        if padding > 0:
            input_ids = torch.nn.functional.pad(input_ids, (0, padding), 'constant', pad) 
            attention_mask = torch.nn.functional.pad(attention_mask, (0, padding), 'constant', 0) 
            return [input_ids, attention_mask]

        # Inputs are larger than window size -> sample random windows
        true_lengths = torch.sum(torch.where(input_ids != 1, 1, 0), 1)
        start_indices = torch.tensor([random.randint(1, l - window_length + 2) if l >= window_length else 1 for l in true_lengths])
        indices = torch.tensor([list(range(start, start + window_length - 2)) for start, l in zip(start_indices, true_lengths)])
        input_ids = input_ids.gather(1, indices)
        attention_mask = attention_mask.gather(1, indices)
        
        # Add cls token
        input_ids = torch.nn.functional.pad(input_ids, (1, 0), 'constant', cls)
        attention_mask = torch.nn.functional.pad(attention_mask, (1, 0), 'constant', 1)
        
        # Add eos token
        input_ids = torch.cat((input_ids, torch.where(true_lengths > window_length, eos, pad).unsqueeze(1)), 1)
        attention_mask = torch.cat((attention_mask, torch.where(true_lengths > window_length, 1, 0).unsqueeze(1)), 1)

        return [input_ids, attention_mask]

    def load_data(self, filename: str):
        """Loads in the data specified in `filename` and populates the necessary 
           variables for sampling the dataset.

        Args:
            filename (str): JSON file to load.
        """
        query_file = 'query' if self.is_queries else 'targets'
        print("Loading {} dataset {} {} file: {}".format(
              self.dataset_name, self.split, query_file, filename))

        if self.dataset_name in ["iur_dataset", "raw_all"]:
            self.build_byte_count_list(filename, load_first_N=self.sanity)
            self.num_authors = len(self.byte_count_list)
        else: 
            self.data = pd.read_json(filename, lines=True, nrows=self.sanity)
            self.num_authors = len(self.data)
        
    def build_byte_count_list(self, filename: str, load_first_N: int):
        """Builds a list where each element contains the number of bytes for 
           that particular line.
        """
        byte_count_list = []

        with open(filename, 'r') as fhandle:
            i = 0
            line = fhandle.readline()

            while line != "":
                if load_first_N is not None and i > load_first_N - 1:
                    break

                byte_count_list.append(len(line))
                line = fhandle.readline()
                i += 1

        assert len(byte_count_list) > 0, "got empty byte count list"
        self.byte_count_list = byte_count_list

    def read_line(self, fhandle, index):
        """Reads one line from the filehandle provided. 
           Assumes that build_byte_count_list() has already been called.
        """
        num_bytes_to_seek = sum(self.byte_count_list[:index])
        num_bytes_to_read = self.byte_count_list[index]

        fhandle.seek(0)
        fhandle.seek(num_bytes_to_seek)
        line = str(fhandle.read(num_bytes_to_read))
        json_line = json.loads(line)

        return json_line

    def __len__(self):
      return self.num_authors

    def __getitem__(self, index: int):
        """Returns the item located at `index` from the dataset.
           Must be implemented by the sub-classes.
        """
        raise NotImplementedError

class RedditDataset(RetrievalDataset):
    """Torch Dataset object for the Reddit datasets
    """
    def __init__(
        self, 
        model_name: str,
        token_max_length: int,
        split: str, 
        dataset_name: str = "raw_all",
        text_key: str = "syms",
        time_key: str = "hour",
        is_queries: bool = True,
        sanity: Optional[int] = None,
    ):
        super().__init__(
            dataset_name=dataset_name, 
            model_name=model_name,
            token_max_length=token_max_length,
            split=split, 
            text_key=text_key,
            time_key=time_key,
            is_queries=is_queries,
            sanity=sanity
        )

        # There are two Reddit datasets available, each with their own files:
        dataset_files = {
            "iur_dataset": {
                "train": ("train.jsonl",),
                "validation": ("train.jsonl", "validation.jsonl"),
                "test": ("test_queries.jsonl", "test_targets.jsonl"),
            },
            "raw_all": {
                "train": ("data.jsonl",),
                "validation": ("validation_queries.jsonl", "validation_targets.jsonl"),
                "test": ("test_queries.jsonl", "test_targets.jsonl"),
            },
        }

        self.dataset_path = os.path.join(DATA_PATH, dataset_name)
        assert os.path.exists(self.dataset_path), f"couldn't find dataset at path: {self.dataset_path}"
        idx = 0 if is_queries or self.sanity else 1
        split = "train" if self.sanity else split
        filename = dataset_files[self.dataset_name][split][idx]

        self.filename = os.path.join(self.dataset_path, filename)
        self.load_data(self.filename)
        self.is_test = split != "train"
        self.use_random_windows = False
        self.min_posts_per_subreddit = 100
        self.subreddits = self.process_subreddits()
        self.subreddit_keys = list(self.subreddits.keys())

    def process_subreddits(self) -> Dict[str, List[Tuple[int, int]]]:
        self.fhandle = open(self.filename, "r")
        subreddits = collections.defaultdict(list)
        for author_idx in tqdm.tqdm(range(self.num_authors), desc='counting subreddits'):
            author_data = self.read_line(self.fhandle, author_idx)
            for subreddit_idx, subreddit_name in enumerate(author_data['action_type']):
                text = author_data[self.text_key][subreddit_idx]
                if len(text) < self.min_char_length: continue
                subreddits[subreddit_name].append([author_idx, subreddit_idx])
        return { 
            k: v for k,v in subreddits.items() if len(v) > self.min_posts_per_subreddit 
        }

    def _get_doc(self, fhandle: io.TextIOWrapper, author_idx: int, document_idx: int) -> str:
        author_data = self.read_line(self.fhandle, author_idx)
        return author_data['syms'][document_idx]
        
    def __getitem__(
        self, 
        index: int
    ) -> torch.Tensor:
        self.fhandle = open(self.filename, "r")
        
        A = [] # left side of unsupervised contrastive
        B = [] # right side of unsupervised contrastive
        subreddit_key = self.subreddit_keys[index]
        for author_idx, document_idx in self.subreddits[subreddit_key]:
            text = self._get_doc(self.fhandle, author_idx, document_idx)
            input_ids_A, _ = self.tokenize_text(text)
            A.append(input_ids_A)
        
        return torch.cat(A, dim=0)

    def __len__(self):
        return len(self.subreddits)

if __name__ == '__main__':
    data_folder = "data/full"
    os.makedirs(data_folder, exist_ok=True)
    output_file = "test.dataset"
    dataset = RedditDataset(
        model_name="bert-base-uncased", # Used for tokenization (TODO argparse)
        split="train",
        token_max_length=256, # TODO argparse ...
        sanity=None,  # TODO argparse ...
    )
    print(dataset[0])
    print(len(dataset))
    print({k: len(v) for k,v in sorted(dataset.subreddits.items(), key=lambda item: len(item[1]))})

    # Creating full dataset
    output_dataset = None
    total = 0
    subreddit_idxs = {}
    for subreddit_idx in tqdm.trange(len(dataset)):
        batch_input_ids = dataset[subreddit_idx]
        batch = {
            "input_ids": batch_input_ids,
            "subreddit_idx": [subreddit_idx] * len(batch_input_ids),
        }
        # Also track a mapping of which idxs are in which dataset
        subreddit_idxs[subreddit_idx] = (
            [id + total for id in range(len(batch_input_ids))]
        )
        total += len(batch_input_ids)
        batch_dataset = datasets.Dataset.from_dict(batch)
        if output_dataset is None:
            output_dataset = batch_dataset
        else:
            output_dataset = datasets.concatenate_datasets(
                (output_dataset, batch_dataset)
            )
    # Save to disk
    output_dataset.save_to_disk(os.path.join(data_folder, output_file))
    pickle.dump(subreddit_idxs, open(os.path.join(data_folder, "subreddit_idxs.p"), "wb"))
    pickle.dump(dataset.subreddit_keys, open(os.path.join(data_folder, "subreddit_keys.p"), "wb"))
