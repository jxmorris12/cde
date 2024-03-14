torchrun --nproc_per_node 8 finetune.py --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --torch_compile 0 --use_wandb --bf16 1 --dataset nomic --sampling_strategy cluster --clustering_query_to_doc 1 --num_hard_negatives 7 --exp_name "nomicembed-cluster-6-gtr-qtd--faiss--indomain" --num_train_epochs 10  --learning_rate 2e-6 --embedder "nomic-ai/nomic-embed-text-v1-unsupervised" --clustering_model gtr_base

torchrun --nproc_per_node 8 finetune.py --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --torch_compile 0 --use_wandb --bf16 1 --dataset nomic --sampling_strategy cluster --clustering_query_to_doc 1 --num_hard_negatives 7 --exp_name "nomicbert-cluster-6-gtr-qtd--faiss--indomain" --num_train_epochs 10  --learning_rate 2e-6 --embedder "nomic-ai/nomic-bert-2048" --clustering_model gtr_base



