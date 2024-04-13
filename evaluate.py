from lib.utils import analyze_utils

args_str = "--per_device_train_batch_size 1024 --per_device_eval_batch_size 1024 --use_wandb 1 --bf16 1 --dataset nomic_unsupervised --sampling_strategy cluster_within_domain --exp_name 2024-04-04-transductive-28-dt-pos--gradclip --num_train_epochs 3 --learning_rate 2e-5 --embedder nomic-ai/nomic-bert-2048 --dataset_embedder nomic-ai/nomic-bert-2048 --clustering_model gtr_base --clustering_query_to_doc 1 --automatically_deduplicate_documents 1 --automatically_deduplicate_queries 1 --arch query_independent_dt --dataset_info batch --ddp_find_unused_parameters 0 --torch_compile 0 --eval_rerank_topk 1024 --lr_scheduler_type cosine --warmup_steps 20000 --disable_dropout 1 --eval_steps 10000 --max_seq_length 512 --max_batch_size_fits_in_memory 256 --use_gc 1 --logging_steps 40 --logit_scale 20.0"
model_folder = "/home/paperspace/tti3/saves/BACKUP--2024-04-04-transductive-28-dt-pos--gradclip/2024-04-04-transductive-28-dt-pos--gradclip/"

beir_dataset_names = [ 
    'arguana',
    'webis-touche2020',
    'quora',
    'nfcorpus',
    'scidocs', 
    'scifact',
    'trec-covid',
    'signal1m',
    'fiqa',
    'trec-news',  
    'msmarco',
    'nq'
]
def main():
    trainer = analyze_utils.load_trainer_from_checkpoint_and_args(
        model_folder=model_folder,
        args_str=args_str,
        beir_dataset_names=beir_dataset_names,
    )
    trainer.evaluate_retrieval_datasets()

if __name__ == '__main__':
    main()
