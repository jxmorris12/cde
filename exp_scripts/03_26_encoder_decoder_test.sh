# test
# CUDA_VISIBLE_DEVICES=7 python finetune.py --per_device_train_batch_size 224 --per_device_eval_batch_size 224 --use_wandb 1 --bf16 1 --dataset nomic_unsupervised --sampling_strategy cluster_within_domain --exp_name "transductive-1-mlp" --num_train_epochs 0.00001 --learning_rate 1e-5 --embedder "sentence-transformers/gtr-t5-base" --dataset_embedder t5-small --clustering_model gtr_base --clustering_query_to_doc 1 --automatically_deduplicate_documents 1 --automatically_deduplicate_queries 1 --arch encoder_decoder_de --dataset_info batch 

# fake
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nproc_per_node 4 finetune.py --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --use_wandb 1 --bf16 1 --dataset nomic_unsupervised --sampling_strategy cluster_within_domain --exp_name "transductive-2-enc-dec--fake" --num_train_epochs 1.2 --learning_rate 1e-5 --embedder "sentence-transformers/gtr-t5-base" --dataset_embedder "sentence-transformers/gtr-t5-base" --clustering_model gtr_base --clustering_query_to_doc 1 --automatically_deduplicate_documents 1 --automatically_deduplicate_queries 1 --arch encoder_decoder_de --dataset_info fake --ddp_find_unused_parameters 1 --torch_compile 1 --torch_compile_mode max-autotune-no-cudagraphs


# real
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nproc_per_node 4 finetune.py --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --use_wandb 1 --bf16 1 --dataset nomic_unsupervised --sampling_strategy cluster_within_domain --exp_name "transductive-2-enc-dec" --num_train_epochs 1.2 --learning_rate 1e-5 --embedder "sentence-transformers/gtr-t5-base" --dataset_embedder "sentence-transformers/gtr-t5-base" --clustering_model gtr_base --clustering_query_to_doc 1 --automatically_deduplicate_documents 1 --automatically_deduplicate_queries 1 --arch encoder_decoder_de --dataset_info batch --ddp_find_unused_parameters 1 --torch_compile 1 --torch_compile_mode max-autotune-no-cudagraphs
# --tiny_debug 1 --use_wandb 0
#  --torch_compile 1 --torch_compile_mode max-autotune-no-cudagraphs
