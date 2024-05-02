# 
# Retrain transductive model with fix
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --master_port 1338 --nproc_per_node 4 finetune.py --per_device_train_batch_size 256 --per_device_eval_batch_size 256 --bf16 1 --use_wandb 1 --dataset nomic_supervised --sampling_strategy cluster_within_domain --num_train_epochs 1 --learning_rate 2e-5 --embedder nomic-ai/nomic-embed-text-v1-unsupervised --clustering_model gtr_base --clustering_query_to_doc 1 --automatically_deduplicate_documents 1 --automatically_deduplicate_queries 1 --arch transductive --dataset_info batch --eval_rerank_topk 512 --use_prefix 1 --exp_name 2024-05-02-transductive-unsupervised-cluster224-1epoch-supervised-cluster224--FIXED-HN --overwrite_output_dir --use_wandb 1 --num_hard_negatives 7 --lr_scheduler_type linear --ddp_share_negatives_between_gpus 0 --use_gc 1 --max_batch_size_fits_in_memory 128 --warmup_steps 400 --logging_steps 50 --train_cluster_size 224 --save_steps 1000 --model_state_dict_from_path /data/saves/tti3/backup/2024-04-17-transductive-pretrain-16 --transductive_corpus_size 256

#
# Train transductive model for 10 epochs w/ checkpoints
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --master_port 1338 --nproc_per_node 4 finetune.py --per_device_train_batch_size 256 --per_device_eval_batch_size 256 --bf16 1 --use_wandb 1 --dataset nomic_supervised --sampling_strategy cluster_within_domain --num_train_epochs 10 --learning_rate 2e-5 --embedder nomic-ai/nomic-embed-text-v1-unsupervised --clustering_model gtr_base --clustering_query_to_doc 1 --automatically_deduplicate_documents 1 --automatically_deduplicate_queries 1 --arch transductive --dataset_info batch --eval_rerank_topk 512 --use_prefix 1 --exp_name 2024-05-02-transductive-unsupervised-cluster224-10epoch-supervised-cluster224--FIXED-HN --overwrite_output_dir --use_wandb 1 --num_hard_negatives 7 --lr_scheduler_type linear --ddp_share_negatives_between_gpus 0 --use_gc 1 --max_batch_size_fits_in_memory 128 --warmup_steps 400 --logging_steps 50 --train_cluster_size 224 --save_strategy epoch --save_total_limit 5 --model_state_dict_from_path /data/saves/tti3/backup/2024-04-17-transductive-pretrain-16 --transductive_corpus_size 256


#
# Run some evals
#
# MODELS=("biencoder-scratch-02-cluster224" "biencoder-16cluster-4kbatch-1epoch" "biencoder-domain-1" "supervised-baseline-domain-1epoch" "supervised-baseline-cluster224-1epoch" "supervised-baseline-cluster224-1epoch--nodedup" "supervised-baseline-cluster224-1epoch--fixdedup" "transductive-scratch-02-cluster224" "transductive-1epoch-supervised-cluster224")


#for model in "${MODELS[@]}"; do
#    echo "testing model ${model}";
#    CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 \
#        --master_port 1338 evaluate_beir.py run "${model}" \
#        --total 1024 \
#        --top_k 4096 \
#        --batch_size 512;
#done


# for model in "${MODELS[@]}"; do
#     echo "testing model ${model}";
#     CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 \
#         --master_port 1340 evaluate_beir.py run "${model}" \
#         --total 1024 \
#         --top_k 256 \
#         --batch_size 512;
# done


# for model in "${MODELS[@]}"; do
#     echo "testing model ${model}";
#     CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 \
#         --master_port 1338 evaluate_beir.py run "${model}" \
#         --total 8192 \
#         --top_k 1024 \
#         --batch_size 512;
# done
