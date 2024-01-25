from typing import List

import collections
import datasets
import os
import pickle
import random
import tqdm

from dataset import RedditDataset


output_file = "test.dataset"

# Minimum number of documents per subreddit.
MIN_N_DOCS: int = 1000

# Number of passages to generate documents for.
N_PASSAGES: int = 50

# Number of questions to generate per passage.
N_QUESTIONS: int = 3


def generate_questions_from_passage(passage: str, n: int) -> List[str]:
    return ["test"] * n

def main():
    # Number of documents to generate questions for per subreddit.
    data_folder: str = "/home/jxm3/research/retrieval/tti3/data/mini"
    ds = RedditDataset(data_folder=data_folder)
    subreddit_lists = {key: lst for key, lst in ds.subreddit_idxs.items() if len(lst) >= MIN_N_DOCS}
    print(f"Num subreddits with at least {MIN_N_DOCS} documents: {len(subreddit_lists)}")
    print(f"\tTotal number of documents: {sum(map(len, subreddit_lists.values()))}")

    all_questions = []
    question_idxs = collections.defaultdict(list)
    for subreddit_key, subreddit_values in tqdm.tqdm(subreddit_lists.items()):
        passage_idxs = random.sample(subreddit_values, N_PASSAGES)
        question_idxs[subreddit_key].extend(passage_idxs)

        for passage_idx in tqdm.tqdm(passage_idxs, leave=False, desc=f"Generating questionos for subreddit {subreddit_key}...")
            passage = ds.dataset[passage_idxs]["text"]
            questions = generate_questions_from_passage(passage, n=N_QUESTIONS)
            for question in questions:
                all_questions.append({
                    "question": question,
                    "passage_idx": passage_idx,
                })

        breakpoint()
    
    question_data_folder = os.path.join(data_folder, "questions")
    os.makedirs(question_data_folder, exist_ok=True)
    dataset = datasets.Dataset.from_list(all_questions)
    dataset.save_to_disk(os.path.join(data_folder, output_file))
    pickle.dump(question_idxs, open(os.path.join(data_folder, "question_idxs.p"), "wb"))

if __name__ == "__main__":
    main()