#!/bin/bash
# elph-collab.sh
set -euo pipefail

# -----------------------------
# USER CONFIG
# -----------------------------
WANDB_ENTITY="${WANDB_ENTITY:-refined-gae}"
WANDB_PROJECT="${WANDB_PROJECT:-Refined-GAE}"
WANDB_GROUP="${WANDB_GROUP:-elph-collab-ablation}"
HF_REPO_ID="${HF_REPO_ID:-an-20225432-hust/elph-checkpoints}"

# 0 = không dùng val edges khi infer test (setting chuẩn để model selection)
# 1 = dùng val edges khi infer test (setting "use most recent edges" cho ogbl-collab)
USE_VALEDGES_AS_INPUT="${USE_VALEDGES_AS_INPUT:-0}"

# Repro: run.py của bạn set_seed(rep), nên reps=1 sẽ cố định theo rep=0
REPS="${REPS:-1}"

echo "📊 Config:"
echo "  WANDB_ENTITY=$WANDB_ENTITY"
echo "  WANDB_PROJECT=$WANDB_PROJECT"
echo "  WANDB_GROUP=$WANDB_GROUP"
echo "  HF_REPO_ID=$HF_REPO_ID"
echo "  USE_VALEDGES_AS_INPUT=$USE_VALEDGES_AS_INPUT"
echo "  REPS=$REPS"
echo ""

# -----------------------------
# COMMON ARGS (ổn định cho ogbl-collab)
# -----------------------------
COMMON_ARGS=(
  --dataset_name ogbl-collab
  --model ELPH
  --K 50
  --epochs 20
  --eval_steps 1
  --reps "${REPS}"
  --batch_size 1024
  --hidden_channels 512
  --num_negs 1
  --weight_decay 1e-5
  --year 2007
  --cache_subgraph_features
  --add_normed_features 1
  --wandb
  --wandb_project "${WANDB_PROJECT}"
  --wandb_entity "${WANDB_ENTITY}"
  --wandb_group "${WANDB_GROUP}"
  --save_model
  --hf_repo_id "${HF_REPO_ID}"
)

# optional: dùng val edges ở inference time (ogbl-collab hay tăng mạnh)
if [[ "${USE_VALEDGES_AS_INPUT}" == "1" ]]; then
  COMMON_ARGS+=( --use_valedges_as_input )
fi

run_exp () {
  local RUN_NAME="$1"; shift
  echo "=================================================="
  echo "RUN: ${RUN_NAME}"
  echo "EXTRA ARGS: $*"
  echo "=================================================="
  python runners/run.py \
    "${COMMON_ARGS[@]}" \
    --wandb_run_name "${RUN_NAME}" \
    "$@"
}

# ==================================================
# 1) Baseline (struct feature only)  [baseline]
#   - chỉ dùng structural/subgraph-sketch features
#   - tắt raw node feature + tắt learnable embedding
# ==================================================
run_exp "collab-elph-baseline-struct-only" \
  --use_feature 0 \
  --use_struct_feature 1 \
  --label_dropout 0.10 \
  --feature_dropout 0.20 \
  --lr 0.002

# ==================================================
# 2) Depth-sweep  [depth-sweep]
#   - tăng max_hash_hops (dễ overfit/unstable → giảm lr nhẹ)
# ==================================================
run_exp "collab-elph-depth-max_hash_hops-3" \
  --use_feature 0 \
  --use_struct_feature 1 \
  --max_hash_hops 3 \
  --label_dropout 0.15 \
  --feature_dropout 0.25 \
  --lr 0.0015

# ==================================================
# 3) Learnable embedding  [learnable embedding]
#   - chỉ learnable embedding (no raw feature), và propagate embeddings
#   - case này collab hay overfit → tăng dropout + weight_decay, giảm lr
# ==================================================
run_exp "collab-elph-learnable-emb-only" \
  --use_feature 0 \
  --use_struct_feature 1 \
  --train_node_embedding \
  --propagate_embeddings \
  --label_dropout 0.25 \
  --feature_dropout 0.35 \
  --weight_decay 5e-5 \
  --lr 0.001

# ==================================================
# 4) Raw feature only  [raw feature]
#   - chỉ raw node feature, tắt structural sketch + tắt learnable embedding
# ==================================================
run_exp "collab-elph-raw-feature-only" \
  --use_feature 1 \
  --use_struct_feature 0 \
  --label_dropout 0.10 \
  --feature_dropout 0.20 \
  --lr 0.0015

# ==================================================
# 5) Raw feature + learnable embeddings  [raw + learnable]
#   - raw feature + trainable embedding (dễ overfit hơn raw-only → tăng dropout)
# ==================================================
run_exp "collab-elph-raw-plus-learnable-emb" \
  --use_feature 1 \
  --use_struct_feature 0 \
  --train_node_embedding \
  --propagate_embeddings \
  --label_dropout 0.20 \
  --feature_dropout 0.30 \
  --weight_decay 5e-5 \
  --lr 0.001

# ==================================================
# 6) Residual encoder  [residual]
#   - bật feature_prop residual (capacity tăng → giữ lr thấp, dropout cao)
# ==================================================
run_exp "collab-elph-encoder-residual" \
  --use_feature 1 \
  --use_struct_feature 0 \
  --train_node_embedding \
  --propagate_embeddings \
  --feature_prop residual \
  --initial_residual \
  --label_dropout 0.25 \
  --feature_dropout 0.35 \
  --weight_decay 1e-4 \
  --lr 0.0008

# ==================================================
# 7) Concat encoder (JK cat)  [concat]
# ==================================================
run_exp "collab-elph-encoder-concat" \
  --use_feature 1 \
  --use_struct_feature 0 \
  --train_node_embedding \
  --propagate_embeddings \
  --feature_prop cat \
  --label_dropout 0.25 \
  --feature_dropout 0.35 \
  --weight_decay 1e-4 \
  --lr 0.0008

# ==================================================
# 8) Orthogonal init  [orthogonal initialization]
# ==================================================
run_exp "collab-elph-orthogonal-init" \
  --use_feature 1 \
  --use_struct_feature 0 \
  --train_node_embedding \
  --propagate_embeddings \
  --orthogonal_init \
  --label_dropout 0.20 \
  --feature_dropout 0.30 \
  --weight_decay 5e-5 \
  --lr 0.001

echo "✅ Done. All 8 experiments finished."
