#!/bin/bash
# setup.sh

# Stop script execution if any command fails
set -e

# =============================================
# TRAINING CONFIGURATION
# =============================================
HF_REPO_ID="${HF_REPO_ID:-an-20225432-hust/elph-checkpoints}"  # Default repo nếu không set

echo "📊 Training configuration:"
echo "  - Wandb Entity: $WANDB_ENTITY"
echo "  - Wandb Project: $WANDB_PROJECT"
echo "  - HuggingFace Repo: $HF_REPO_ID"
echo ""

# collab baseline
echo "======================="
echo "RUNNING COLLAB BASELINE"
echo "======================="
python runners/run.py --dataset_name ogbl-collab --K 50 --lr 0.002 \
  --feature_dropout 0.2 --add_normed_features 1 --cache_subgraph_features \
  --label_dropout 0.1 --year 2007 --model BUDDY \
  --train_node_embedding  \
  --wandb --wandb_project $WANDB_PROJECT --wandb_entity $WANDB_ENTITY \
  --wandb_run_name buddy-baseline-fixed \
  --hf_repo_id $HF_REPO_ID

# python runners/run.py \
#   --dataset ogbl-collab --K 50 --model ELPH --save_model \
#   --use_feature 0 --train_node_embedding --propagate_embeddings \
#   --epochs 20 \
#   --batch_size 131072 \
#   --hidden_channels 256 \
#   --label_dropout 0.25 \
#   --lr 0.0005 \
#   --num_negs 6 \
#   --sign_k 2 \
#   --wandb --wandb_project $WANDB_PROJECT --wandb_entity $WANDB_ENTITY \
#   --wandb_run_name buddy-baseline \
#   --hf_repo_id $HF_REPO_ID \
#   --use_valedges_as_input

# # collab depth sweep, only increase max hop
# echo "==============================="
# echo "RUNNING COLLAB max_hash_hop = 3"
# echo "==============================="
# python runners/run.py \
#   --dataset ogbl-collab --K 50 --model ELPH --save_model \
#   --use_feature 0 --train_node_embedding --propagate_embeddings \
#   --epochs 20 \
#   --batch_size 131072 \
#   --hidden_channels 256 \
#   --label_dropout 0.25 \
#   --lr 0.0015 \
#   --num_negs 6 \
#   --sign_k 2 \
#   --max_hash_hops 3 \
#   --wandb --wandb_project $WANDB_PROJECT --wandb_entity $WANDB_ENTITY \
#   --wandb_run_name collab-depth-sweep-max-hop-3 \
#   --hf_repo_id $HF_REPO_ID

# # collab learnable (already in baseline, increase SIGN(k))
# echo "=========================="
# echo "RUNNING COLLAB SIGN(k = 3)"
# echo "=========================="
# python runners/run.py \
#   --dataset ogbl-collab --K 50 --model ELPH --save_model \
#   --use_feature 0 --train_node_embedding --propagate_embeddings \
#   --epochs 20 \
#   --batch_size 131072 \
#   --hidden_channels 256 \
#   --label_dropout 0.25 \
#   --lr 0.0015 \
#   --num_negs 6 \
#   --sign_k 3 \
#   --wandb --wandb_project $WANDB_PROJECT --wandb_entity $WANDB_ENTITY \
#   --wandb_run_name collab-learnable-sign-k-3 \
#   --hf_repo_id $HF_REPO_ID

# # collab raw feature only (from dataset, no initializing embedding)
# echo "=============================================="
# echo "RUNNING COLLAB RAW FEATURE ONLY (NO EMBEDDING)"
# echo "=============================================="
# python runners/run.py \
#   --dataset ogbl-collab --K 50 --model ELPH --save_model \
#   --use_feature 1 \
#   --epochs 20 \
#   --batch_size 131072 \
#   --hidden_channels 256 \
#   --label_dropout 0.25 \
#   --lr 0.0015 \
#   --num_negs 6 \
#   --sign_k 2 \
#   --wandb --wandb_project $WANDB_PROJECT --wandb_entity $WANDB_ENTITY \
#   --wandb_run_name collab-raw-feat-no-embed \
#   --hf_repo_id $HF_REPO_ID

# # collab raw feature and embedding
# echo "========================================"
# echo "RUNNING COLLAB RAW FEATURE AND EMBEDDING"
# echo "========================================"
# python runners/run.py \
#   --dataset ogbl-collab --K 50 --model ELPH --save_model \
#   --use_feature 1 --train_node_embedding --propagate_embeddings \
#   --epochs 20 \
#   --batch_size 131072 \
#   --hidden_channels 256 \
#   --label_dropout 0.25 \
#   --lr 0.0015 \
#   --num_negs 6 \
#   --sign_k 2 \
#   --wandb --wandb_project $WANDB_PROJECT --wandb_entity $WANDB_ENTITY \
#   --wandb_run_name collab-raw-feat-no-embed \
#   --hf_repo_id $HF_REPO_ID

# # collab feature prop: residual
# echo "================================="
# echo "RUNNING COLLAB FEAT PROP RESIDUAL"
# echo "================================="
# python runners/run.py \
#   --dataset ogbl-collab --K 50 --model ELPH --save_model \
#   --use_feature 0 --train_node_embedding --propagate_embeddings \
#   --epochs 20 \
#   --batch_size 131072 \
#   --hidden_channels 256 \
#   --label_dropout 0.25 \
#   --lr 0.0015 \
#   --num_negs 6 \
#   --sign_k 2 \
#   --feature_prop residual \
#   --wandb --wandb_project $WANDB_PROJECT --wandb_entity $WANDB_ENTITY \
#   --wandb_run_name collab-feat-prop-res \
#   --hf_repo_id $HF_REPO_ID

# # collab feature prop: concat
# echo "==============================="
# echo "RUNNING COLLAB FEAT PROP CONCAT"
# echo "==============================="
# python runners/run.py \
#   --dataset ogbl-collab --K 50 --model ELPH --save_model \
#   --use_feature 0 --train_node_embedding --propagate_embeddings \
#   --epochs 20 \
#   --batch_size 131072 \
#   --hidden_channels 256 \
#   --label_dropout 0.25 \
#   --lr 0.0015 \
#   --num_negs 6 \
#   --sign_k 2 \
#   --feature_prop cat \
#   --wandb --wandb_project $WANDB_PROJECT --wandb_entity $WANDB_ENTITY \
#   --wandb_run_name collab-feat-prop-cat \
#   --hf_repo_id $HF_REPO_ID

# # collab embedding init orthogonal
# echo "==============================="
# echo "RUNNING COLLAB ORTHOGONAL EMBED"
# echo "==============================="
# python runners/run.py \
#   --dataset ogbl-collab --K 50 --model ELPH --save_model \
#   --use_feature 0 --train_node_embedding --propagate_embeddings \
#   --epochs 20 \
#   --batch_size 131072 \
#   --hidden_channels 256 \
#   --label_dropout 0.25 \
#   --lr 0.0015 \
#   --num_negs 6 \
#   --sign_k 2 \
#   --orthogonal_init \
#   --wandb --wandb_project $WANDB_PROJECT --wandb_entity $WANDB_ENTITY \
#   --wandb_run_name collab-orthogonal \
#   --hf_repo_id $HF_REPO_ID