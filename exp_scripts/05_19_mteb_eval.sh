# MODELS=(
#     "cde"
#     "cde--supervised-1"
#     "cde--supervised-2"
#     "cde--supervised-3"
# )
MODELS=(
    "cde--no-prefix"
    "cde--no-prefix--supervised-1"
    "cde--no-prefix--supervised-2"
    "cde--no-prefix--supervised-3"
)

for model in "${MODELS[@]}"; do
    echo $model;
    CUDA_VISIBLE_DEVICES=0 python evaluate_mteb.py $model;
done