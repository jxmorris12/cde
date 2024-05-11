
MODELS=("supervised-baseline-cluster224-1epoch--nodedup" "transductive-scratch-02-cluster224--1.5" "hard-biencoder-supervised-224cluster-1epoch" "hard-biencoder-supervised-domain-1epoch" "transductive-1epoch-supervised-cluster224-sd0.01" "transductive-1epoch-supervised-cluster224-sd0.1" "transductive-1epoch-supervised-cluster224-sd0.2")


for model in "${MODELS[@]}"; do
    echo "testing model ${model}";
    CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 \
        --master_port 1340 evaluate_beir.py run "${model}" \
        --total 1024 \
        --top_k 256 \
        --batch_size 512;
done

for model in "${MODELS[@]}"; do
   echo "testing model ${model}";
   CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 \
       --master_port 1338 evaluate_beir.py run "${model}" \
       --total 1024 \
       --top_k 1024 \
       --batch_size 512;
done



# for model in "${MODELS[@]}"; do
#     echo "testing model ${model}";
#     CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 \
#         --master_port 1338 evaluate_beir.py run "${model}" \
#         --total 8192 \
#         --top_k 1024 \
#         --batch_size 512;
# done
