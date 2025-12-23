#!/bin/bash
# test-automation.sh - Test script with minimal epochs to verify automation

# Stop script execution if any command fails
set -e

echo "🧪 ========================================"
echo "🧪   TESTING AUTOMATION SCRIPT"
echo "🧪 ========================================"
echo ""

# =============================================
# CONFIGURATION
# =============================================
echo "⚙️ Step 2/3: Setting up configuration..."
echo ""

WANDB_ENTITY="an-20225432-hust"
WANDB_PROJECT="ELPH-TEST"
HF_REPO_ID="${HF_REPO_ID:-an-20225432-hust/elph-test-checkpoints}"

echo "📊 Configuration:"
echo "  - Wandb Entity: $WANDB_ENTITY"
echo "  - Wandb Project: $WANDB_PROJECT"
echo "  - HuggingFace Repo: $HF_REPO_ID"
echo "  - Test Mode: Minimal epochs (2 epochs only)"
echo ""

# =============================================
# TEST RUN
# =============================================
echo "🚀 Step 3/3: Running test training..."
echo ""

echo "===================="
echo "TEST RUN - DDI BASELINE (2 EPOCHS)"
echo "===================="
python runners/run.py \
  --dataset ogbl-ddi --K 20 --model ELPH --save_model \
  --use_feature 0 --train_node_embedding --propagate_embeddings \
  --epochs 2 \
  --batch_size 131072 \
  --hidden_channels 128 \
  --label_dropout 0.25 \
  --lr 0.0015 \
  --num_negs 6 \
  --sign_k 2 \
  --wandb --wandb_project $WANDB_PROJECT --wandb_entity $WANDB_ENTITY \
  --wandb_run_name test-automation-baseline \
  --hf_repo_id $HF_REPO_ID

echo ""
echo "✅ ========================================"
echo "✅   TEST COMPLETED SUCCESSFULLY!"
echo "✅ ========================================"
echo ""
echo "📝 Check your results:"
echo "   - Wandb: https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT"
echo "   - HuggingFace: https://huggingface.co/$HF_REPO_ID"
echo ""
