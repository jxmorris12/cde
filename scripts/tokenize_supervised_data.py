from typing import Dict

import collections
import os
import pickle

import datasets
import tqdm
import transformers


num_proc = len(os.sched_getaffinity(0))


def main():
    data_folder = f"data/nomic_embed_supervised"
    os.makedirs(data_folder, exist_ok=True)
    output_file = "test.dataset"

    subdomains = collections.defaultdict(list)
    bert_tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
    t5_tokenizer = transformers.AutoTokenizer.from_pretrained("t5-base")
    
    os.environ["TOKENIZERS_PARALLELISM"] = "1"
    token_max_length = 64

    output_dataset = datasets.load_dataset("nomic-ai/nomic_embed_supervised")["train"]

    # Tokenize
    def tokenize_ex(ex: Dict) -> Dict:
        ex["query_input_ids_bert"] = bert_tokenizer(
            ex["query"], 
            padding=True, 
            truncation=True,
            max_length=token_max_length, 
            return_tensors='pt'
        ).input_ids
        ex["document_input_ids_bert"] = bert_tokenizer(
            ex["document"], 
            padding=True, 
            truncation=True,
            max_length=token_max_length, 
            return_tensors='pt'
        ).input_ids

        ex["query_input_ids_t5"] = t5_tokenizer(
            ex["query"], 
            padding=True, 
            truncation=True,
            max_length=token_max_length, 
            return_tensors='pt'
        ).input_ids
        ex["document_input_ids_t5"] = t5_tokenizer(
            ex["document"], 
            padding=True, 
            truncation=True,
            max_length=token_max_length, 
            return_tensors='pt'
        ).input_ids

        ex["hn_input_ids__bert"] = []
        ex["hn_input_ids__t5"] = []
        for negatives_list in ex["negative"]:
            negative_input_ids__bert = t5_tokenizer(
                negatives_list, 
                padding=True, 
                truncation=True,
                max_length=token_max_length, 
                return_tensors='pt'
            ).input_ids
            negative_input_ids__t5 = t5_tokenizer(
                negatives_list, 
                padding=True, 
                truncation=True,
                max_length=token_max_length, 
                return_tensors='pt'
            ).input_ids
            ex["hn_input_ids__bert"].append(
                negative_input_ids__bert
            )
            ex["hn_input_ids__t5"].append(
                negative_input_ids__t5
            )

        return ex

    cache_file_name = os.path.join(data_folder, output_file) + f"{len(output_dataset)}.cache"
    print("tokenizing dataset of length:", len(output_dataset))
    #output_dataset = output_dataset.map(
    #    tokenize_ex,
    #    batch_size=1000,
    #    batched=True,
    #    cache_file_name=cache_file_name,
    #    num_proc=num_proc,
    #)

    for i in tqdm.trange(len(output_dataset), desc="Counting datasets"):
        subdomain = output_dataset[i]["dataset"]
        subdomains[subdomain].append(i)

    # Save to disk
    #print("saving dataset of length:", len(output_dataset), "with", len(subdomains), "subdomains")
    #output_dataset.save_to_disk(os.path.join(data_folder, output_file))
    pickle.dump(subdomains, open(os.path.join(data_folder, "subdomain_idxs.p"), "wb"))


if __name__ == '__main__':
    main()
