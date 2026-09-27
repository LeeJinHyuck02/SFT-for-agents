#!/usr/bin/env bash
# [추론 전용] 학습된 어댑터로 지문을 생성한다. 학습은 하지 않는다 → 학습은 scripts/run_train.sh
#
#   bash scripts/run_infer.sh --detach                          # outputs/{agent}_think/final, data/SFT_think_en_test.jsonl 19건
#   bash scripts/run_infer.sh --detach --split valid            # valid 18건
#   bash scripts/run_infer.sh --detach --limit 3                # 앞의 3건만
#   bash scripts/run_infer.sh --detach --target-length 1500     # 파일의 지문별 목표 분량 대신 모든 샘플에 같은 분량을 지시 (고정 조건 비교)
#   bash scripts/run_infer.sh --detach --input data/my.jsonl --output results/my.jsonl
#   bash scripts/run_infer.sh --detach --no-adapter             # 미세조정 전 베이스 모델 (비교용)
#   bash scripts/run_infer.sh --detach --batch-size 8           # [hf] 한 번에 생성할 건수 (기본 20)
#
# vLLM 경로 (bf16 원본 + 실행 중 LoRA, 별도 venv. 계획: issue/0919/0919_inf.md, 환경 수정: issue/0925/0925.md 부록 A)
#   bash scripts/run_infer.sh --detach --backend vllm --parity                 # 먼저: HF 4-bit 기준값과 loss 비교 + 형식 게이지
#   bash scripts/run_infer.sh --detach --backend vllm --limit 1 --greedy       # smoke
#   bash scripts/run_infer.sh --detach --backend vllm                          # test 19건
#   bash scripts/run_infer.sh --detach --backend vllm --max-num-seqs 4 --max-num-batched-tokens 2048   # KV cache 부족(엔진 초기화 실패) 시
#   bash scripts/run_infer.sh --detach --backend vllm --temperature 0.7 \
#       --adapters final=outputs/generation_think/final ep1=outputs/generation_think/checkpoint-37 ep2=outputs/generation_think/checkpoint-74
#
# --detach, --from, --backend 외의 인자는 그대로 `python -m ugrp.inference.infer`(hf) 또는 `ugrp.inference.infer_vllm`(vllm)으로 전달된다.
# 단계: setup → vllm_setup → adapter → infer → report → pack
#   setup      같은 Pod에서 run_train.sh가 이미 했으면 건너뛴다. 새 Pod면 모델을 다시 받는다
#   vllm_setup [vllm] scripts/setup_vllm.sh로 /root/vllm-env 설치 (이미 있으면 건너뜀). hf면 건너뛴다
#   adapter    어댑터가 있는지 먼저 확인 (31B 모델을 올린 뒤에 실패하지 않도록)
#   infer      배치마다 results/*.jsonl에 기록. 다시 실행하면 이미 있는 id는 건너뛴다
#   report     형식 지표 집계 (분할 추론일 때만)
#   pack       results_*.tar.gz 갱신
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

AGENT="generation"; SPLIT="test"; FROM=""; DETACH=0; BACKEND="hf"; PARITY=0
ADAPTER=""; ADAPTER_SPECS=(); NO_ADAPTER=0; CUSTOM_IO=0; INFER_ARGS=(); ORIGINAL_ARGS=("$@")
HF_ONLY=""; VLLM_ONLY=""
VLLM_ENV="${VLLM_ENV:-/root/vllm-env}"
while [ $# -gt 0 ]; do
  case "$1" in
    --detach) DETACH=1; shift ;;
    --from) FROM="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --agent) AGENT="$2"; INFER_ARGS+=("$1" "$2"); shift 2 ;;
    --split) SPLIT="$2"; INFER_ARGS+=("$1" "$2"); shift 2 ;;
    --adapter) ADAPTER="$2"; INFER_ARGS+=("$1" "$2"); shift 2 ;;
    --adapters)
      VLLM_ONLY+=" $1"; INFER_ARGS+=("$1"); shift
      while [ $# -gt 0 ] && [[ "$1" != --* ]]; do ADAPTER_SPECS+=("$1"); INFER_ARGS+=("$1"); shift; done ;;
    --no-adapter) NO_ADAPTER=1; INFER_ARGS+=("$1"); shift ;;
    --input|--output) CUSTOM_IO=1; INFER_ARGS+=("$1" "$2"); shift 2 ;;
    --limit|--max-new-tokens|--structure-from|--target-length) INFER_ARGS+=("$1" "$2"); shift 2 ;;
    --batch-size) HF_ONLY+=" $1"; INFER_ARGS+=("$1" "$2"); shift 2 ;;
    --greedy) VLLM_ONLY+=" $1"; INFER_ARGS+=("$1"); shift ;;
    --parity) PARITY=1; VLLM_ONLY+=" $1"; INFER_ARGS+=("$1"); shift ;;
    --temperature|--seed|--repetition-penalty|--gpu-mem|--max-model-len|--max-num-seqs|--max-num-batched-tokens)
      VLLM_ONLY+=" $1"; INFER_ARGS+=("$1" "$2"); shift 2 ;;
    *) echo "알 수 없는 인자: $1"; exit 1 ;;
  esac
done

case "$BACKEND" in
  hf) [ -z "$VLLM_ONLY" ] || { echo "[ABORT] 다음 인자는 --backend vllm에서만 쓸 수 있습니다:$VLLM_ONLY"; exit 1; } ;;
  vllm) [ -z "$HF_ONLY" ] || { echo "[ABORT] 다음 인자는 --backend hf에서만 쓸 수 있습니다:$HF_ONLY"; exit 1; } ;;
  *) echo "[ABORT] --backend는 hf 또는 vllm이어야 합니다: $BACKEND"; exit 1 ;;
esac

if [ "$DETACH" -eq 1 ]; then
  FORWARD=(); for arg in "${ORIGINAL_ARGS[@]}"; do [ "$arg" != "--detach" ] && FORWARD+=("$arg"); done
  detach ugrp-infer scripts/run_infer.sh "${FORWARD[@]}"
  exit 0
fi

check_adapter() {
  local path="$1"
  if [ -f "$path/adapter_config.json" ]; then
    echo "[ADAPTER] $path"
  elif [[ "$path" != outputs/* && "$path" != /* && "$path" != ./* ]]; then
    echo "[ADAPTER] 로컬에 없음 → HF Hub repo로 간주: $path"
  else
    echo "[ABORT] 어댑터가 없습니다: $path"
    echo "        먼저 학습하거나(bash scripts/run_train.sh), 내려받아 둔 결과를 푸십시오(tar -xzf results_*.tar.gz)."
    exit 1
  fi
}

# 추론 결과 파일 목록. infer.py / infer_vllm.py의 기본 출력 경로 규칙과 같아야 한다
result_files() {
  local suffix=""; [ "$NO_ADAPTER" -eq 1 ] && suffix="_base"
  local stem="results/${AGENT}_think${suffix}_${SPLIT}"
  if [ "$BACKEND" = "hf" ]; then
    echo "${stem}.jsonl"
  elif [ "${#ADAPTER_SPECS[@]}" -gt 1 ]; then
    for spec in "${ADAPTER_SPECS[@]}"; do echo "${stem}_vllm_${spec%%=*}.jsonl"; done
  else
    echo "${stem}_vllm.jsonl"
  fi
}

run_stage() {
  case "$1" in
    setup) ensure_setup ;;
    vllm_setup)
      if [ "$BACKEND" = "vllm" ]; then bash scripts/setup_vllm.sh; else echo "[VLLM] --backend hf: 건너뜀"; fi ;;
    adapter)
      if [ "$NO_ADAPTER" -eq 1 ]; then
        echo "[ADAPTER] 어댑터 없이 베이스 모델로 추론합니다."
      elif [ "${#ADAPTER_SPECS[@]}" -gt 0 ]; then
        for spec in "${ADAPTER_SPECS[@]}"; do check_adapter "${spec#*=}"; done
      else
        check_adapter "${ADAPTER:-outputs/${AGENT}_think/final}"
      fi ;;
    infer)
      if [ "$BACKEND" = "vllm" ]; then
        # venv를 activate하지 않으므로 bin을 PATH에 넣는다: FlashInfer가 샘플링 커널을 JIT 빌드할 때 ninja를 이름으로 찾는다 (0925.md A-2)
        PATH="$VLLM_ENV/bin:$PATH" "$VLLM_ENV/bin/python" -m ugrp.inference.infer_vllm "${INFER_ARGS[@]}"
      else
        python -m ugrp.inference.infer "${INFER_ARGS[@]}"
      fi ;;
    report)
      local files=(); while read -r f; do [ -f "$f" ] && files+=("$f"); done < <(result_files)
      local suffix=""; [ "$NO_ADAPTER" -eq 1 ] && suffix="_base"
      local tag=""; [ "$BACKEND" = "vllm" ] && tag="_vllm"
      # 어댑터 결과를 집계할 때 같은 split·백엔드의 베이스 결과가 이미 있으면 표에 함께 넣는다 (베이스 대비 비교)
      local base_file="results/${AGENT}_think_base_${SPLIT}${tag}.jsonl"
      if [ "$NO_ADAPTER" -eq 0 ] && [ -f "$base_file" ]; then files+=("$base_file"); fi
      if [ "$PARITY" -eq 1 ]; then
        echo "[REPORT] --parity는 생성 결과가 없어 집계하지 않습니다."
      elif [ "$CUSTOM_IO" -eq 1 ] || [ "${#files[@]}" -eq 0 ]; then
        echo "[REPORT] --input/--output을 직접 준 추론은 집계하지 않습니다. 필요하면: python -m ugrp.eval_report <결과.jsonl>"
      else
        python -m ugrp.eval_report "${files[@]}" --output "results/eval_report_${AGENT}${suffix}_${SPLIT}${tag}.json"
      fi ;;
    pack) bash scripts/pack_results.sh ;;
  esac
}

run_stages "bash scripts/run_infer.sh --detach --backend $BACKEND ${INFER_ARGS[*]:-}" "$FROM" \
  setup vllm_setup adapter infer report pack
print_download_notice
