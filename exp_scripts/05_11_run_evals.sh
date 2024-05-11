# MODELS=(
#     "transductive-scratch-16kdomain--1"
#     "biencoder-domain-bs256"
#     "transductive-scratch-02-cluster224--3" 
#     "transductive-scratch-02-cluster224--2.25" 
# )

# compared deduplication strategies. they really don't make much of a difference;
# honestly i think i need to focus on bigger fish (this is probably a good thing)
MODELS=(
    "supervised-baseline-cluster224-1epoch--hash-dedup-neither"
    "supervised-baseline-cluster224-1epoch--hash-dedup-both"
    "supervised-baseline-cluster224-1epoch--hash-dedup-document-only"
    "supervised-baseline-cluster224-1epoch--hash-dedup-query-only")

for model in "${MODELS[@]}"; do
    echo "testing model ${model}";
    torchrun --nproc_per_node 8 \
        evaluate_beir.py run "${model}" \
        --total 1024 \
        --top_k 256 \
        --batch_size 512;
done


# gonna try again with best supervised baseline model and see how results compare to nomic
# supervised training
python evaluate_mteb.py "supervised-baseline-cluster224-1epoch--nodedup"