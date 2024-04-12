torchrun --nproc_per_node 8 finetune.py --per_device_train_batch_size 224 --per_device_eval_batch_size 224 --use_wandb 1 --bf16 1 --dataset nomic_unsupervised --sampling_strategy cluster_within_domain --exp_name "unsupervised-3-cluster-temp-1" --num_train_epochs 0.4 --learning_rate 1e-5 --embedder "nomic-ai/nomic-bert-2048" --clustering_model gtr_base --clustering_query_to_doc 1 --logit_scale 40

torchrun --nproc_per_node 8 finetune.py --per_device_train_batch_size 224 --per_device_eval_batch_size 224 --use_wandb 1 --bf16 1 --dataset nomic_unsupervised --sampling_strategy cluster_within_domain --exp_name "unsupervised-3-cluster-temp-5" --num_train_epochs 0.4 --learning_rate 1e-5 --embedder "nomic-ai/nomic-bert-2048" --clustering_model gtr_base --clustering_query_to_doc 1 --logit_scale 10


##############################################################################################################

torchrun --nproc_per_node 8 finetune.py --per_device_train_batch_size 224 --per_device_eval_batch_size 224 --use_wandb 1 --bf16 1 --dataset nomic_unsupervised --sampling_strategy cluster_within_domain --exp_name "unsupervised-3-cluster" --num_train_epochs 0.4 --learning_rate 1e-5 --embedder "nomic-ai/nomic-bert-2048" --clustering_model gtr_base --clustering_query_to_doc 1 --automatically_deduplicate_documents 1 --automatically_deduplicate_queries 1

torchrun --nproc_per_node 8 finetune.py --per_device_train_batch_size 224 --per_device_eval_batch_size 224 --use_wandb 1 --bf16 1 --dataset nomic_unsupervised --sampling_strategy cluster_within_domain --exp_name "unsupervised-3-cluster-nodedup-docs" --num_train_epochs 0.4 --learning_rate 1e-5 --embedder "nomic-ai/nomic-bert-2048" --clustering_model gtr_base --clustering_query_to_doc 1 --automatically_deduplicate_documents 0 --automatically_deduplicate_queries 1

torchrun --nproc_per_node 8 finetune.py --per_device_train_batch_size 224 --per_device_eval_batch_size 224 --use_wandb 1 --bf16 1 --dataset nomic_unsupervised --sampling_strategy cluster_within_domain --exp_name "unsupervised-3-cluster-nodedup-queries" --num_train_epochs 0.4 --learning_rate 1e-5 --embedder "nomic-ai/nomic-bert-2048" --clustering_model gtr_base --clustering_query_to_doc 1 --automatically_deduplicate_documents 1 --automatically_deduplicate_queries 0
