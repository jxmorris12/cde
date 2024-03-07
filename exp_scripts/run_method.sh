for temp in 15 20; do
    for mgn in 1 10; do
        timeout 1.1h torchrun --nproc_per_node 8 finetune.py --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --torch_compile 0 --use_wandb --bf16 1 --dataset nomic --sampling_strategy cluster --num_hard_negatives 7 --contrastive_temp $temp --exp_name "bienc-kmeans-hn-temp${temp}_mgn_${mgn}" --num_train_epochs 5 --max_grad_norm $mgn
    done
done

# one long one
torchrun --nproc_per_node 8 finetune.py --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --torch_compile 0 --use_wandb --bf16 1 --dataset nomic --sampling_strategy cluster --num_hard_negatives 7 --contrastive_temp $temp --exp_name "bienc-kmeans-hn---long" --num_train_epochs 100 --max_grad_norm $mgn --learning_rate 5e-6