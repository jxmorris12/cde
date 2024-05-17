for MODEL in "transductive-scratch-06-cluster224--2.8" "transductive-scratch-02-cluster224--2"; do
 echo $MODEL;
    # fake
    torchrun --nproc_per_node  evaluate_beir.py run $MODEL --total 1024 --top_k 256 --batch_size 256 --transductive_input_strategy fake --transductive_n_outputs_ensemble 1
    # random
    torchrun --nproc_per_node  evaluate_beir.py run $MODEL --total 1024 --top_k 256 --batch_size 256 --transductive_input_strategy random --transductive_n_outputs_ensemble 1
    # top-k_pool (factor of 4 is set in the code)
    torchrun --nproc_per_node  evaluate_beir.py run $MODEL --total 1024 --top_k 256 --batch_size 256 --transductive_input_strategy topk_pool --transductive_n_outputs_ensemble 1
    # topk-ensemble-4
    torchrun --nproc_per_node  evaluate_beir.py run $MODEL --total 1024 --top_k 256 --batch_size 256 --transductive_input_strategy topk --transductive_n_outputs_ensemble 4
done