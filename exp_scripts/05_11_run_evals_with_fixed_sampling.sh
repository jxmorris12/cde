MODELS=(
    "baseline-supervised-cluster256--share-sample")

for model in "${MODELS[@]}"; do
    echo "testing model ${model}";
    torchrun --nproc_per_node 8 \
        evaluate_beir.py run "${model}" \
        --total 1024 \
        --top_k 256 \
        --batch_size 512;
done