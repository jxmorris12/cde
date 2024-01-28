from typing import List

import collections
import datasets
import openai
import os
import pickle
import random
import time
import tqdm

from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
)

from dataset import RedditDataset


output_file = "test.dataset"

# Minimum number of documents per subreddit.
MIN_N_DOCS: int = 256

# Number of passages to generate documents for.
N_PASSAGES: int = 4

# Number of questions to generate per passage.
N_QUESTIONS: int = 3


client = openai.OpenAI(
    # This is the default and can be omitted
    # api_key=os.environ.get("OPENAI_API_KEY"),
)


# @retry(wait=wait_fixed(2), stop=stop_after_attempt(5))
def chat_compeletion_openai(model, messages, temperature=1, max_tokens=512):
    return "a\nb\nc"
    response = client.chat.completions.create(
            model=model,
            messages=messages,
            n=1,
            temperature=temperature,
            max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def generate_questions_from_passage(passage: str, n: int, max_n_words: int = 100) -> List[str]:
    assert isinstance(passage, str), f"need str passage; got type {type(passage)}"
    messages = [
        {
            "role": "system", 
            "content": f"You are a question generator. Your task is to generate {n} questions based on a document provided by the user. Documents should be specific to this passage only. Create one question per line."
        }
    ]
    passage = " ".join(passage.split()[:max_n_words])
    prompt = f"Document:\n{passage}"
    messages.append({"role": "user", "content": prompt})
    output = chat_compeletion_openai(
        model="gpt-3.5-turbo-1106",
        messages=messages
    )
    
    # TODO: Smarter parsing.
    parsed_output = [line.strip() for line in output.strip().split("\n") if len(line.strip())]
    return parsed_output


def main():
    # Number of documents to generate questions for per subreddit.
    # data_folder: str = "/home/jxm3/research/retrieval/tti3/data/mini"
    data_folder: str = "/home/jxm3/research/retrieval/tti3/data/full"
    ds = RedditDataset(data_folder=data_folder)
    subreddit_lists = { key: lst for key, lst in ds.subreddit_idxs.items() if len(lst) >= MIN_N_DOCS }
    print(f"Num subreddits with at least {MIN_N_DOCS} documents: {len(subreddit_lists)}")
    print(f"\tTotal number of documents: {sum(map(len, subreddit_lists.values()))}")
    print(f"\Estimated number of questions: {N_PASSAGES}*{len(subreddit_lists.items())}*{N_QUESTIONS}={N_PASSAGES*len(subreddit_lists.items())*N_QUESTIONS}")

    all_questions = []
    question_idxs = collections.defaultdict(list)
    for subreddit_key, subreddit_values in tqdm.tqdm(subreddit_lists.items(), total=len(subreddit_lists), desc="Processing data"):
        passage_idxs = random.sample(subreddit_values, N_PASSAGES)
        question_idxs[subreddit_key].extend(passage_idxs)
        for passage_idx in tqdm.tqdm(passage_idxs, leave=False, desc=f"Generating questions for subreddit {subreddit_key}..."):
            passage = ds.dataset[passage_idx]["text"]
            questions = generate_questions_from_passage(passage, n=N_QUESTIONS)
            for question in questions:
                all_questions.append({
                    "question": question,
                    "passage_idx": passage_idx,
                    "subreddit_key": subreddit_key,
                })
    
    print(f"generated {len(all_questions)} questions!")
    question_data_folder = os.path.join(data_folder, "questions")
    os.makedirs(question_data_folder, exist_ok=True)
    dataset = datasets.Dataset.from_list(all_questions)
    dataset.save_to_disk(os.path.join(question_data_folder, output_file))
    pickle.dump(question_idxs, open(os.path.join(question_data_folder, "question_idxs.p"), "wb"))
    print(f"wrote dataset of length {len(dataset)} and question-idx-map of length {len(question_idxs)} to {question_data_folder}")

if __name__ == "__main__":
    main()
