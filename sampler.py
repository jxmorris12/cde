from typing import Dict, Iterable, Optional, Union

import collections
import random
import torch

from dataset import NomicDataset, RedditDataset

class RedditSampler(torch.utils.data.Sampler):
    """Samples randomly from subreddits during training."""
    dataset: Union[NomicDataset, RedditDataset]
    batch_size: int
    max_num_batches: Optional[int]
    def __init__(
            self, 
            dataset: Union[NomicDataset, RedditDataset], batch_size: int, 
            shuffle: bool, max_num_batches: Optional[int] = None,
        ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_num_batches = max_num_batches

        len_minus_batch = lambda L: len(L) - (len(L) % batch_size)
        num_questions = { k: len_minus_batch(v) for k,v in self.dataset.subdomain_idxs.items() }
        self.subreddit_batches = [x for k,v in num_questions.items() for x in [k] * v]
        identity = lambda x: x
        self.shuffle_func = random.shuffle if shuffle else identity

    def __iter__(self) -> Iterable[Dict]:
        # randomly sample questions per-subreddit (w/o replacement)
        subreddit_idx_pointer = collections.defaultdict(lambda: 0)
        self.shuffle_func(self.subreddit_batches)

        for i, subreddit_idx in enumerate(self.subreddit_batches):
            # manually track length and stop
            if (i * self.batch_size) >= len(self):
                break

            pointer = subreddit_idx_pointer[subreddit_idx]
            if pointer == 0:
                self.shuffle_func(self.dataset.subdomain_idxs[subreddit_idx])
            subreddit_idx_pointer[subreddit_idx] += 1
            questions = self.dataset.subdomain_idxs[subreddit_idx]

            if (pointer + 1) * self.batch_size > len(questions):
                # drop small batches
                continue
            else:
                for j in range(pointer * self.batch_size, (pointer + 1) * self.batch_size):
                    yield questions[j]

    def __len__(self) -> int:
        if self.max_num_batches:
            return min(len(self.subreddit_batches), self.max_num_batches) * self.batch_size
        else:
            return len(self.subreddit_batches) * self.batch_size
