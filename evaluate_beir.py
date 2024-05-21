import argparse
import glob
import json
import os
import pandas as pd
import tqdm

from spider.lib.misc import md5_hash_kwargs
from spider.lib.model_configs import MODEL_FOLDER_DICT, ARGS_STR_DICT
from spider.lib.utils import analyze_utils


# can run with torchrun, for example:
#   > torchrun --nproc_per_node 8 evaluate_beir.py \
#               biencoder-scratch-02-cluster224
#

beir_dataset_names = [ 
    'arguana',
    'fiqa',
    'msmarco',
    'nfcorpus',
    'nq',
    'quora',
    'scidocs', 
    'scifact',
    'signal1m',
    'trec-covid',
    'trec-news',  
    'webis-touche2020',
]

# beir_dataset_names = [ 'msmarco' ]
# beir_dataset_names = [ 'nfcorpus', ]
# beir_dataset_names = [ 'nfcorpus', 'arguana' ]
# beir_dataset_names = [ 'hotpotqa' ]
# beir_dataset_names = [ 'dbpedia', 'climate-fever'] # , 'fever', 'msmarco', 'hotpotqa', 'fiqa' ]
# beir_dataset_names = ['arguana']
# beir_dataset_names = [ 'dbpedia', 'climate-fever'] # , 'fever', 'msmarco', 'hotpotqa', 'fiqa' ]

beir_dataset_names = [ 'dbpedia' ]

cwd = os.path.normpath(
    os.path.dirname(os.path.abspath(__file__)),
)
root_save_folder = os.path.join(
    cwd, "results_beir"
)


def setup_eval_cmd_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "model_key", 
        help="The key for the model", 
        type=str, 
        choices=MODEL_FOLDER_DICT.keys()
    )
    parser.add_argument(
        "--top_k", "--k",
        type=int, 
        default=8192,
    )
    parser.add_argument(
        "--batch_size", "--b",
        type=int, 
        default=128,
    )
    parser.add_argument(
        "--total", "--n",
        type=int, 
        default=512,
    )
    parser.add_argument(
        "--transductive_input_strategy", "--t",
        type=str, 
        default="topk",
        choices=["fake", "dummy", "random_corpus", "topk", "random_corpus__topk__interp", "topk_pool", "null", "null_topk"],
    )
    parser.add_argument(
        "--transductive_n_outputs_ensemble",
        type=int,
        default=1
    )
    parser.add_argument(
        "--embedder_rerank",
        type=str,
        default="sentence-transformers/gtr-t5-base"
    )

def evaluate_model(args):
    model_folder = MODEL_FOLDER_DICT[args.model_key]
    args_str = ARGS_STR_DICT.get(args.model_key)

    save_folder = os.path.join(root_save_folder, args.model_key)
    os.makedirs(save_folder, exist_ok=True)
    args_dict = dict(vars(args))
    args_dict.pop("func")

    ##########################################
    # Remove defaults from new args to preserve caching
    if args_dict["transductive_input_strategy"] == "topk":
        args_dict.pop("transductive_input_strategy")
    if args_dict["transductive_n_outputs_ensemble"] == 1:
        args_dict.pop("transductive_n_outputs_ensemble")
    if args_dict["embedder_rerank"] == "sentence-transformers/gtr-t5-base":
        args_dict.pop("embedder_rerank")
    ##########################################

    args_dict["datasets"] = tuple(beir_dataset_names)
    save_hash = md5_hash_kwargs(**args_dict)
    save_path = os.path.join(save_folder, save_hash  + ".json")
    print(f"checking for cached results at {save_path}")
    if os.path.exists(save_path):
        print(f"found cached results at {save_path}")
        exit()

    args_str += f' --embedder_rerank "{args.embedder_rerank}"'
    trainer = analyze_utils.load_trainer_from_checkpoint_and_args(
        model_folder=model_folder,
        args_str=args_str,
        beir_dataset_names=beir_dataset_names,
        load_from_checkpoint=True, # Set to false for random predictions
    )
    trainer.model.eval()
    trainer.args.max_batch_size_fits_in_memory = args.batch_size
    trainer.args.eval_rerank_topk = args.top_k
    trainer.args.transductive_input_strategy = args.transductive_input_strategy
    trainer.args.transductive_n_outputs_ensemble = args.transductive_n_outputs_ensemble
    results_dict = trainer.evaluate_retrieval_datasets(n=args.total)
    results_dict["_args"] = args_dict

    if trainer._is_main_worker:
        with open(save_path, "w") as json_file:
            json.dump(results_dict, json_file, indent=4)
        print(f"[rank 0] saved {len(results_dict)} results to {save_path}")

def print_results(args):
    print("printing :D")
    results_jsons = glob.glob(os.path.join(root_save_folder, "*", "*.json"))
    print(results_jsons)
    all_jsons = [json.load(open(j, "r")) for j in tqdm.tqdm(results_jsons, desc="Reading *.json", leave=False)]
    # add args to outer dict
    for i in range(len(all_jsons)):
        for k, v in all_jsons[i]["_args"].items():
            all_jsons[i][k] = v
    df = pd.DataFrame(all_jsons)
    df = df[df["top_k"] == 256].reset_index()
    df = df[df["total"] == 1024].reset_index()
    df = df.set_index("model_key")
    df = df[[c for c in df.columns if c.endswith("NDCG@10")]]
    df.columns = [c.replace("eval_BeIR/", "") for c in df.columns]
    df["NDCG@10__mean"] = df.mean(axis=1)
    pd.set_option('display.max_columns', None)
    print(df)
    breakpoint()


def main():
    print("Running something...")
    parser = argparse.ArgumentParser(description="Run and examine BEIR evaluation results")

    # Creating subparsers
    subparsers = parser.add_subparsers(title="subcommands", dest="command", required=True)

    # Subparser for print command
    print_parser = subparsers.add_parser("print", help="Print results df")
    print_parser.set_defaults(func=print_results)

    # Subparser for run command
    eval_parser = subparsers.add_parser("run", help="Run a task")
    # eval_parser.add_argument("eval", help="Evaluate a model")
    eval_parser.set_defaults(func=evaluate_model)
    setup_eval_cmd_parser(eval_parser)

    args = parser.parse_args()
    args.func(args)

    
if __name__ == '__main__':
    main()
