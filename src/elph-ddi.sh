#!/bin/bash
# elph-ddi.sh - Main training script with automatic authentication

# Stop script execution if any command fails
set -e


# =============================================
# TRAINING CONFIGURATION
# =============================================
WANDB_ENTITY="an-20225432-hust"
WANDB_PROJECT="ELPH"
HF_REPO_ID="${HF_REPO_ID:-an-20225432-hust/elph-checkpoints}"  # Default repo nếu không set

echo "📊 Training configuration:"
echo "  - Wandb Entity: $WANDB_ENTITY"
echo "  - Wandb Project: $WANDB_PROJECT"
echo "  - HuggingFace Repo: $HF_REPO_ID"
echo ""

# ddi baseline
echo "===================="
echo "RUNNING DDI BASELINE"
echo "===================="
python runners/run.py \
  --dataset ogbl-ddi --K 20 --model ELPH --save_model \
  --use_feature 0 --train_node_embedding --propagate_embeddings \
  --epochs 20 \
  --batch_size 131072 \
  --hidden_channels 256 \
  --label_dropout 0.25 \
  --lr 0.0015 \
  --num_negs 6 \
  --sign_k 2 \
  --wandb --wandb_project $WANDB_PROJECT --wandb_entity $WANDB_ENTITY \
  --wandb_run_name ddi-baseline \
  --hf_repo_id $HF_REPO_ID

# ddi depth sweep, only increase max hop
echo "============================"
echo "RUNNING DDI max_hash_hop = 3"
echo "============================"
python runners/run.py \
  --dataset ogbl-ddi --K 20 --model ELPH --save_model \
  --use_feature 0 --train_node_embedding --propagate_embeddings \
  --epochs 20 \
  --batch_size 131072 \
  --hidden_channels 256 \
  --label_dropout 0.25 \
  --lr 0.0015 \
  --num_negs 6 \
  --sign_k 2 \
  --max_hash_hops 3 \
  --wandb --wandb_project $WANDB_PROJECT --wandb_entity $WANDB_ENTITY \
  --wandb_run_name ddi-depth-sweep-max-hop-3 \
  --hf_repo_id $HF_REPO_ID

# ddi learnable (already in baseline, increase SIGN(k))
echo "======================="
echo "RUNNING DDI SIGN(k = 3)"
echo "======================="
python runners/run.py \
  --dataset ogbl-ddi --K 20 --model ELPH --save_model \
  --use_feature 0 --train_node_embedding --propagate_embeddings \
  --epochs 20 \
  --batch_size 131072 \
  --hidden_channels 256 \
  --label_dropout 0.25 \
  --lr 0.0015 \
  --num_negs 6 \
  --sign_k 3 \
  --wandb --wandb_project $WANDB_PROJECT --wandb_entity $WANDB_ENTITY \
  --wandb_run_name ddi-learnable-sign-k-3 \
  --hf_repo_id $HF_REPO_ID

# ddi raw feature only (from dataset, no initializing embedding) (ddi has no feature, it means disabling embedding)
echo "==========================================="
echo "RUNNING DDI RAW FEATURE ONLY (NO EMBEDDING)"
echo "==========================================="
python runners/run.py \
  --dataset ogbl-ddi --K 20 --model ELPH --save_model \
  --use_feature 1 \
  --epochs 20 \
  --batch_size 131072 \
  --hidden_channels 256 \
  --label_dropout 0.25 \
  --lr 0.0015 \
  --num_negs 6 \
  --sign_k 2 \
  --wandb --wandb_project $WANDB_PROJECT --wandb_entity $WANDB_ENTITY \
  --wandb_run_name ddi-raw-feat-no-embed \
  --hf_repo_id $HF_REPO_ID

# ddi feature prop: residual
echo "=============================="
echo "RUNNING DDI FEAT PROP RESIDUAL"
echo "=============================="
python runners/run.py \
  --dataset ogbl-ddi --K 20 --model ELPH --save_model \
  --use_feature 0 --train_node_embedding --propagate_embeddings \
  --epochs 20 \
  --batch_size 131072 \
  --hidden_channels 256 \
  --label_dropout 0.25 \
  --lr 0.0015 \
  --num_negs 6 \
  --sign_k 2 \
  --feature_prop residual \
  --wandb --wandb_project $WANDB_PROJECT --wandb_entity $WANDB_ENTITY \
  --wandb_run_name ddi-feat-prop-res \
  --hf_repo_id $HF_REPO_ID

# ddi feature prop: concat
echo "============================"
echo "RUNNING DDI FEAT PROP CONCAT"
echo "============================"
python runners/run.py \
  --dataset ogbl-ddi --K 20 --model ELPH --save_model \
  --use_feature 0 --train_node_embedding --propagate_embeddings \
  --epochs 20 \
  --batch_size 131072 \
  --hidden_channels 256 \
  --label_dropout 0.25 \
  --lr 0.0015 \
  --num_negs 6 \
  --sign_k 2 \
  --feature_prop cat \
  --wandb --wandb_project $WANDB_PROJECT --wandb_entity $WANDB_ENTITY \
  --wandb_run_name ddi-feat-prop-cat \
  --hf_repo_id $HF_REPO_ID

# ddi embedding init orthogonal
echo "============================"
echo "RUNNING DDI ORTHOGONAL EMBED"
echo "============================"
python runners/run.py \
  --dataset ogbl-ddi --K 20 --model ELPH --save_model \
  --use_feature 0 --train_node_embedding --propagate_embeddings \
  --epochs 20 \
  --batch_size 131072 \
  --hidden_channels 256 \
  --label_dropout 0.25 \
  --lr 0.0015 \
  --num_negs 6 \
  --sign_k 2 \
  --orthogonal_init \
  --wandb --wandb_project $WANDB_PROJECT --wandb_entity $WANDB_ENTITY \
  --wandb_run_name ddi-orthogonal \
  --hf_repo_id $HF_REPO_ID