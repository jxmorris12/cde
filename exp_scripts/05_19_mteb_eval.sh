# MODELS=(
#     "cde--supervised-2"
#     "cde--supervised-3"
# )
MODELS=(
    "cde"
    "cde--supervised-1"
    "cde--no-prefix"
    "cde--no-prefix--supervised-1"
)

for model in "${MODELS[@]}"; do
    echo $model;
    CUDA_VISIBLE_DEVICES=0 python evaluate_mteb.py $model;
done