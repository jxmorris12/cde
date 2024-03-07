from typing import Dict

import collections
import os
import pickle
import yaml

import datasets
import tqdm
import transformers


USE_MP = True
# num_proc = min(32, len(os.sched_getaffinity(0)))
num_proc = len(os.sched_getaffinity(0))
os.environ["TOKENIZERS_PARALLELISM"] = str(int(USE_MP))


max_num_hn = 7
token_max_length = 128
add_prefixes = True

def main():
    data_folder = os.path.join("data", "nomic_embed_supervised")
    os.makedirs(data_folder, exist_ok=True)
    output_file = "test.dataset"

    bert_tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    t5_tokenizer = transformers.AutoTokenizer.from_pretrained("t5-base", use_fast=True)
    
    print(f"downloading Nomic data; tokenizing to length {token_max_length}")
    output_dataset = datasets.load_dataset("nomic-ai/nomic_embed_supervised")["train"]
    print("got data")

    yaml_path = os.path.join(data_folder, "dataset_info.yaml")
    dataset_info = yaml.safe_load(open(yaml_path, "r"))
    dataset_info = dataset_info["datasets"]


    query_prefixes = {}
    document_prefixes = {}
    for d in dataset_info:
        name = d["name"]
        if d["query_only"]:
            query_prefixes[name] = d["query_prefix"]
            document_prefixes[name] = d["query_prefix"] # TODO: Check with Zach to make sure this is right.
        else:
            query_prefixes[name] = d["query_prefix"]
            document_prefixes[name] = d.get["document_prefix"]

    # Add prefixes
    if add_prefixes:
        breakpoint()


    # Tokenize
    def tokenize_ex(ex: Dict) -> Dict:
        ex["query_input_ids__bert"] = bert_tokenizer(
            ex["query"], 
            padding=True, 
            truncation=True,
            max_length=token_max_length, 
            return_tensors='pt'
        ).input_ids
        ex["document_input_ids__bert"] = bert_tokenizer(
            ex["document"], 
            padding=True, 
            truncation=True,
            max_length=token_max_length, 
            return_tensors='pt'
        ).input_ids

        ex["query_input_ids__t5"] = t5_tokenizer(
            ex["query"], 
            padding=True, 
            truncation=True,
            max_length=token_max_length, 
            return_tensors='pt'
        ).input_ids
        ex["document_input_ids__t5"] = t5_tokenizer(
            ex["document"], 
            padding=True, 
            truncation=True,
            max_length=token_max_length, 
            return_tensors='pt'
        ).input_ids

        negatives_list = [neg for neg_list in ex["negative"] for neg in neg_list[:max_num_hn]]

        negative_input_ids__bert = bert_tokenizer(
            negatives_list, 
            padding=True, 
            truncation=True,
            max_length=token_max_length, 
            return_tensors='pt'
        ).input_ids
        print("tokenizing:", len(negatives_list), "...")
        negative_input_ids__t5 = t5_tokenizer(
            negatives_list, 
            padding=True, 
            truncation=True,
            max_length=token_max_length, 
            return_tensors='pt'
        ).input_ids
        ex["hn_input_ids__bert"] = []
        ex["hn_input_ids__t5"] = []
        i = 0
        for b in ex["negative"]:
            bl = min(len(b), max_num_hn)
            ex["hn_input_ids__bert"].append(
                negative_input_ids__bert[i:i+bl]
            )
            ex["hn_input_ids__t5"].append(
                negative_input_ids__t5[i:i+bl]
            )
            i += len(b)

        return ex

    cache_file_name = os.path.join(data_folder, output_file) + f"{len(output_dataset)}.cache"
    print("tokenizing dataset of length:", len(output_dataset))
    output_dataset = output_dataset.map(
       tokenize_ex,
       batch_size=10_000,
       batched=True,
       cache_file_name=cache_file_name,
       num_proc=(None if USE_MP else num_proc),
    )

    print("collecting subdomains")
    subdomains = collections.defaultdict(list)
    subdomains_list = output_dataset["dataset"]
    for i, subdomain in tqdm.tqdm(enumerate(subdomains_list), desc="Counting datasets"):
        subdomains[subdomain].append(i)

    # Save to disk
    breakpoint()
    print("saving dataset of length:", len(output_dataset), "with", len(subdomains), "subdomains")
    output_dataset.save_to_disk(os.path.join(data_folder, output_file))
    pickle.dump(subdomains, open(os.path.join(data_folder, "subdomain_idxs.p"), "wb"))


if __name__ == '__main__':
    main()
