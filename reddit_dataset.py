from typing import Optional

import json
import os
import random

import pandas as pd
import torch
from transformers import AutoTokenizer
from torch.utils.data.dataset import Dataset


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
        episode_length: int,
        token_max_length: int,
        split: str, 
        num_sample_per_author: int, 
        text_key: str,
        time_key: str,
        is_queries: bool = False,
        sanity: Optional[int] = None,
    ):
        """Initializes the Dataset class.

        Args:
            dataset_name (str): name of the dataset
            model_name (str): name of model (to load its tokenizer)
            episode_length (int): Number of windows per sampled example
            token_max_length (int): Max number of tokens in a window
            split (str): Name of the split: train, validation, or test.
            text_key (str)
            time_key (str)
            num_sample_per_author (int): Number of data points to sample per author.
            is_queries (bool, optional): Whether or not we're reading in the queries.
            sanity (int | None): if set, use only this many authors (for debugging)
        """
        self.split = split
        assert self.split in ["train", "validation", "test"]
        
        self.dataset_name = dataset_name
        self.num_sample_per_author = num_sample_per_author
        self.token_max_length = token_max_length
        self.is_queries = is_queries
        self.sanity = sanity
        self.episode_length = episode_length
      
        self.dataset_path = os.path.join(DATA_PATH, self.dataset_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name) 
        
        self.text_key = text_key
        self.time_key = time_key

    def tokenize_text(self, all_text):
        tokenized_episode = self.tokenizer(
            all_text, 
            padding=True if self.use_random_windows else "max_length", 
            truncation=False if self.use_random_windows else True, 
            max_length=None if self.use_random_windows else self.token_max_length, 
            return_tensors='pt'
        )
        tokenized_episode =  self.reformat_tokenized_inputs(tokenized_episode)
        
        return tokenized_episode

    def reformat_tokenized_inputs(self, tokenized_episode):
        """Reformats the output from HugginFace Tokenizers.
        """
        if len(tokenized_episode.keys()) == 3:
            input_ids, _, attention_mask = tokenized_episode.values()
            data = [input_ids, attention_mask]
        else:
            input_ids, attention_mask = tokenized_episode.values()
            data = [input_ids, attention_mask]

        return data
   
    def sample_random_episode(self, index: int, is_test=False):
        """Samples a random episode of size `episode_length`.

        Args:
            index (int): Index of the author we're sampling.
        """
        if hasattr(self, "fhandle"):
            author_data = self.read_line(self.fhandle, index)
        else:
            author_data = self.data.iloc[index].to_dict()

        assert self.text_key in author_data, f"key '{self.text_key} missing in {author_data.keys()}"
        num_docs = len(author_data[self.text_key])
        episode_length = num_docs if num_docs < self.episode_length else self.episode_length

        maxval = num_docs - episode_length

        if is_test:
            start_index = 0
        else:
            start_index = random.randint(0, maxval)
    
        episode = {k: v[start_index: start_index + episode_length] 
                   for k, v in author_data.items() if isinstance(v, list) and len(v) > 0}

        if self.sanity or "author_id" not in author_data:
            episode["author_id"] = index
        else:            
            episode["author_id"] = author_data["author_id"] 
            
        return episode
    
    def sample_random_window(self, data, window_length=32):
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
        input_ids = torch.cat((input_ids, torch.where(true_lengths >= window_length, eos, pad).unsqueeze(1)), 1)
        attention_mask = torch.cat((attention_mask, torch.where(true_lengths >= window_length, 1, 0).unsqueeze(1)), 1)

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
        episode_length: int,
        token_max_length: int,
        split: str, 
        num_sample_per_author: int, 
        dataset_name: str = "raw_all",
        text_key: str = "syms",
        time_key: str = "hour",
        is_queries: bool = True,
        sanity: Optional[int] = None,
    ):
        super().__init__(
            dataset_name=dataset_name, 
            model_name=model_name,
            episode_length=episode_length,
            token_max_length=token_max_length,
            split=split, 
            text_key=text_key,
            time_key=time_key,
            num_sample_per_author=num_sample_per_author, 
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
        
    def __getitem__(
        self, 
        index: int
    ):
        self.fhandle = open(self.filename, "r")

        text = []
        for _ in range(self.num_sample_per_author):
            episode = self.sample_random_episode(index, is_test=self.is_test)
            text.extend(episode["syms"])
                
        if self.split == "train":
            author = torch.tensor([index for _ in range(self.num_sample_per_author)])
        else:
            author = torch.tensor([episode["author_id"] for _ in range(self.num_sample_per_author)])

        data = self.tokenize_text(text)
        if self.use_random_windows:
            data = self.sample_random_window(data)
        data = [d.reshape(self.num_sample_per_author, -1, self.token_max_length) for d in data]

        return data, author

if __name__ == '__main__':
    dataset = RedditDataset(
        model_name="bert-base-uncased",
        split="train",
        episode_length=1,
        token_max_length=32,
        num_sample_per_author=2,
        sanity=1000,
    )
    print(dataset[0])
    print(len(dataset))