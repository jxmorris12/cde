MODELS=("biencoder-scratch-02-cluster224" "biencoder-16cluster-4kbatch-1epoch" "biencoder-domain-1" "supervised-baseline-domain-1epoch" "supervised-baseline-cluster224-1epoch" "supervised-baseline-cluster224-1epoch--nodedup" "supervised-baseline-cluster224-1epoch--fixdedup" "transductive-scratch-02-cluster224")


for model in "${MODELS[@]}"; do
    echo "testing model ${model}";
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 \
        --master_port 1337 evaluate_beir.py "${model}" --total 10000 \
        --batch_size 4096;
done