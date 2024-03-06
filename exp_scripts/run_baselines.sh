# torchrun --nproc_per_node 8 finetune.py --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --torch_compile 0 --use_wandb 1 --bf16 1 --exp_name bienc-dataset-hn --dataset nomic --sampling_strategy domain --num_hard_negatives 7
# torchrun --nproc_per_node 8 finetune.py --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --torch_compile 0 --use_wandb --bf16 1 --exp_name bienc-random-hn --dataset nomic --sampling_strategy random --num_hard_negatives 7
torchrun --nproc_per_node 8 finetune.py --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --torch_compile 0 --use_wandb 1 --bf16 1 --exp_name bienc-dataset --dataset nomic --sampling_strategy domain --num_hard_negatives 0
torchrun --nproc_per_node 8 finetune.py --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --torch_compile 0 --use_wandb --bf16 1 --exp_name bienc-random --dataset nomic --sampling_strategy random --num_hard_negatives 0

