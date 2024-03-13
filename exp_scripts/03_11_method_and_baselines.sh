# torchrun --nproc_per_node 8 finetune.py --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --torch_compile 0 --use_wandb --bf16 1 --dataset nomic --sampling_strategy random --num_hard_negatives 7 --exp_name "nomicbert-random-5" --num_train_epochs 10 --learning_rate 2e-6 --embedder "nomic-ai/nomic-bert-2048"

# torchrun --nproc_per_node 8 finetune.py --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --torch_compile 0 --use_wandb --bf16 1 --dataset nomic --sampling_strategy random --num_hard_negatives 7 --exp_name "nomicembed-random-5" --num_train_epochs 10 --learning_rate 2e-6 --embedder "nomic-ai/nomic-embed-text-v1-unsupervised"

# # #########################################################################################################################

# torchrun --nproc_per_node 8 finetune.py --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --torch_compile 0 --use_wandb --bf16 1 --dataset nomic --sampling_strategy domain --num_hard_negatives 7 --exp_name "nomicbert-domain-5" --num_train_epochs 10 --learning_rate 2e-6 --embedder "nomic-ai/nomic-bert-2048"

# torchrun --nproc_per_node 8 finetune.py --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --torch_compile 0 --use_wandb --bf16 1 --dataset nomic --sampling_strategy domain --num_hard_negatives 7 --exp_name "nomicembed-domain-5" --num_train_epochs 10 --learning_rate 2e-6 --embedder "nomic-ai/nomic-embed-text-v1-unsupervised"

# #########################################################################################################################

# torchrun --nproc_per_node 8 finetune.py --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --torch_compile 0 --use_wandb --bf16 1 --dataset nomic --sampling_strategy cluster --num_hard_negatives 7 --exp_name "nomicembed-cluster-5-bm25" --num_train_epochs 10  --learning_rate 2e-6 --embedder "nomic-ai/nomic-embed-text-v1-unsupervised"

# torchrun --nproc_per_node 8 finetune.py --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --torch_compile 0 --use_wandb --bf16 1 --dataset nomic --sampling_strategy cluster --num_hard_negatives 7 --exp_name "nomicbert-cluster-5-bm25" --num_train_epochs 10  --learning_rate 2e-6 --embedder "nomic-ai/nomic-bert-2048"


#########################################################################################################################

# torchrun --nproc_per_node 8 finetune.py --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --torch_compile 0 --use_wandb --bf16 1 --dataset nomic --sampling_strategy cluster --clustering_query_to_doc 1 --num_hard_negatives 7 --exp_name "nomicembed-cluster-5-bm25-qtd" --num_train_epochs 10  --learning_rate 2e-6 --embedder "nomic-ai/nomic-embed-text-v1-unsupervised"

# torchrun --nproc_per_node 8 finetune.py --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --torch_compile 0 --use_wandb --bf16 1 --dataset nomic --sampling_strategy cluster --clustering_query_to_doc 1 --num_hard_negatives 7 --exp_name "nomicbert-cluster-5-bm25-qtd" --num_train_epochs 10  --learning_rate 2e-6 --embedder "nomic-ai/nomic-bert-2048"



#########################################################################################################################

# torchrun --nproc_per_node 8 finetune.py --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --torch_compile 0 --use_wandb --bf16 1 --dataset nomic --sampling_strategy cluster --clustering_query_to_doc 1 --num_hard_negatives 7 --exp_name "nomicembed-cluster-5-gtr-qtd" --num_train_epochs 10  --learning_rate 2e-6 --embedder "nomic-ai/nomic-embed-text-v1-unsupervised" --clustering_model gtr_base

# torchrun --nproc_per_node 8 finetune.py --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --torch_compile 0 --use_wandb --bf16 1 --dataset nomic --sampling_strategy cluster --clustering_query_to_doc 1 --num_hard_negatives 7 --exp_name "nomicbert-cluster-5-gtr-qtd" --num_train_epochs 10  --learning_rate 2e-6 --embedder "nomic-ai/nomic-bert-2048" --clustering_model gtr_base

#########################################################################################################################


torchrun --nproc_per_node 8 finetune.py --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --torch_compile 0 --use_wandb --bf16 1 --dataset nomic --sampling_strategy cluster --clustering_query_to_doc 1 --num_hard_negatives 7 --exp_name "nomicembed-cluster-6-gtr-qtd--faiss" --num_train_epochs 40  --learning_rate 1e-6 --embedder "nomic-ai/nomic-embed-text-v1-unsupervised" --clustering_model gtr_base

torchrun --nproc_per_node 8 finetune.py --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --torch_compile 0 --use_wandb --bf16 1 --dataset nomic --sampling_strategy cluster --clustering_query_to_doc 1 --num_hard_negatives 7 --exp_name "nomicbert-cluster-6-gtr-qtd--faiss" --num_train_epochs 40  --learning_rate 1e-6 --embedder "nomic-ai/nomic-bert-2048" --clustering_model gtr_base



