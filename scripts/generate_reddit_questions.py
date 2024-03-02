from typing import Dict, List

import collections
import concurrent.futures
import os
import pickle
import random
import time

import datasets
import openai
import transformers
import tqdm

from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
)

from dataset import RedditDataset


output_file = "test.dataset"

# Minimum number of documents per subreddit.
MIN_N_DOCS: int = 64

# Number of passages to generate documents for.
N_PASSAGES: int = 64

# Number of questions to generate per passage.
N_QUESTIONS: int = 5 # 3

# Number of threads that will make requests in parallel.
N_THREADS = 128


client = openai.OpenAI()


@retry(wait=wait_fixed(2), stop=stop_after_attempt(5))
def chat_compeletion_openai(model, messages, temperature=1, max_tokens=512):
    response = client.chat.completions.create(
            model=model,
            messages=messages,
            n=1,
            temperature=temperature,
            max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def chat_completion_openai_with_error(*args, **kwargs):
    try:
       response = chat_compeletion_openai(*args, **kwargs)
    except Exception as e:
        print("Got API errror:", e, "with args:", args, "/ kwargs:", kwargs)
        response = "[[[ ERROR ]]]"
    return response


def generate_questions_from_passage(passage: str, n: int, max_n_words: int = 100) -> List[str]:
    assert isinstance(passage, str), f"need str passage; got type {type(passage)}"
    messages = [
        {
            "role": "system", 
            "content": f"You are a question generator. Your task is to generate {n} questions based on a document provided by the user. Documents should be specific to this passage only. Respond with only the questions, no other text. Create one question per line."
        }
    ]
    passage = " ".join(passage.split()[:max_n_words])
    prompt = f"Document:\n{passage}"
    messages.append({"role": "user", "content": prompt})
    output = chat_completion_openai_with_error(
        model="gpt-3.5-turbo-1106",
        messages=messages
    )
    
    # TODO: Smarter parsing.
    parsed_output = [line.strip() for line in output.strip().split("\n") if len(line.strip())]
    return parsed_output


def main():
    start_time = time.time()

    # Number of documents to generate questions for per subreddit.
    # data_folder: str = "/home/jxm3/research/retrieval/tti3/data/mini"
    data_folder: str = "/home/jxm3/research/retrieval/tti3/data/full"

    questions_outfolder = "questions64"

    ds = RedditDataset(
        data_folder=data_folder,
        min_examples_per_subreddit=MIN_N_DOCS,
    )
    subreddit_lists = { key: lst for key, lst in ds.subreddit_idxs.items() if len(lst) >= MIN_N_DOCS }
    print(f"Num subreddits with at least {MIN_N_DOCS} documents: {len(subreddit_lists)}")
    print(f"\t Total number of documents: {sum(map(len, subreddit_lists.values()))}")
    print(f"\t Estimated number of questions: {N_PASSAGES}*{len(subreddit_lists.items())}*{N_QUESTIONS}={N_PASSAGES*len(subreddit_lists.items())*N_QUESTIONS}")

    all_questions = []

    def process_subreddit(subreddit_idx, subreddit_values):
        passage_idxs = random.sample(subreddit_values, N_PASSAGES)
        subreddit_questions = []
        for passage_idx in passage_idxs:
            passage = ds.dataset[passage_idx]["text"]
            questions = generate_questions_from_passage(passage, n=N_QUESTIONS)
            for question in questions:
                subreddit_questions.append({
                    "question": question,
                    "passage_idx": passage_idx,
                    "subreddit_idx": subreddit_idx,
                })
        return subreddit_questions

    all_questions = []
    question_idxs = collections.defaultdict(list)

    questions_total = 0
    with tqdm.tqdm(total=len(subreddit_lists), desc="Processing subreddits", colour="green") as pbar:
        multiproc_cls = concurrent.futures.ThreadPoolExecutor
        # multiproc_cls = concurrent.futures.ProcessPoolExecutor
        with multiproc_cls(
            max_workers=N_THREADS
        ) as executor:
            futures = [executor.submit(process_subreddit, subreddit_idx, subreddit_values) for subreddit_idx, subreddit_values in subreddit_lists.items()]
            for future in concurrent.futures.as_completed(futures):
                questions_list = future.result()
                for q in questions_list:
                    q['question_idx'] = questions_total
                    all_questions.append(q)
                    question_idxs[q['subreddit_idx']].append(questions_total)
                    questions_total += 1
                pbar.update(1)

        
    question_data_folder = os.path.join(data_folder, questions_outfolder)
    os.makedirs(question_data_folder, exist_ok=True)
    dataset = datasets.Dataset.from_list(all_questions)

    # TODO: Argparse for tokenizer!
    print(f"generated {len(all_questions)} questions! tokenizing...")
    model_name = "bert-base-uncased"
    token_max_length = 128
    bert_tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
    t5_tokenizer = transformers.AutoTokenizer.from_pretrained("t5-base")
    def tokenize_ex(ex: Dict) -> Dict:
        tt_bert = bert_tokenizer(
            ex["text"], 
            padding=True, 
            truncation=True,
            max_length=token_max_length, 
            return_tensors='pt'
        )
        tt_t5 = t5_tokenizer(
            ex["text"], 
            padding=True, 
            truncation=True,
            max_length=token_max_length, 
            return_tensors='pt'
        )
        ex["question_input_ids_t5"] = tt_t5.input_ids
        ex["question_input_ids_bert"] = tt_bert.input_ids
        return ex
    
    cache_file_name = os.path.join(question_data_folder, output_file) + f"{len(dataset)}.cache"
    dataset = dataset.map(
        tokenize_ex, batch_size=1000, batched=True, cache_file_name=cache_file_name
    )
    dataset.save_to_disk(os.path.join(question_data_folder, output_file))
    pickle.dump(question_idxs, open(os.path.join(question_data_folder, "question_idxs.p"), "wb"))
    print(f"wrote dataset of length {len(dataset)} and question-idx-map of length {len(question_idxs)} to {question_data_folder}")
    
    time_difference = time.time() - start_time
    time_difference_minutes = time_difference // 60
    time_difference_hours = time_difference_minutes // 60    
    print(f"finished in {time_difference_minutes:.1f} minutes or {time_difference_hours:.1f} hours")

if __name__ == "__main__":
    main()
