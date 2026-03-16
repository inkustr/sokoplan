#!/usr/bin/env bash
set -euo pipefail

TRAIN_SCRIPT="${TRAIN_SCRIPT:-scripts_hydra/blocks/run_train_packs_array.sbatch}"
SELF_EVAL_SCRIPT="${SELF_EVAL_SCRIPT:-scripts_hydra/blocks/run_self_eval_packs_array.sbatch}"
CROSS_EVAL_SCRIPT="${CROSS_EVAL_SCRIPT:-scripts_hydra/blocks/run_cross_eval_pairs_array.sbatch}"

echo "[submit] training: ${TRAIN_SCRIPT}"
TRAIN_JOB_ID="$(sbatch --parsable "${TRAIN_SCRIPT}")"
echo "[submit] train job id: ${TRAIN_JOB_ID}"

echo "[submit] self-eval (afterok:${TRAIN_JOB_ID}): ${SELF_EVAL_SCRIPT}"
SELF_JOB_ID="$(sbatch --parsable --dependency="afterok:${TRAIN_JOB_ID}" "${SELF_EVAL_SCRIPT}")"
echo "[submit] self-eval job id: ${SELF_JOB_ID}"

echo "[submit] cross-eval (afterok:${TRAIN_JOB_ID}): ${CROSS_EVAL_SCRIPT}"
CROSS_JOB_ID="$(sbatch --parsable --dependency="afterok:${TRAIN_JOB_ID}" "${CROSS_EVAL_SCRIPT}")"
echo "[submit] cross-eval job id: ${CROSS_JOB_ID}"

