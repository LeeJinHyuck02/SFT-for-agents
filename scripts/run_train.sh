#!/usr/bin/env bash
# [학습 전용] think QLoRA 어댑터를 학습한다. 추론은 하지 않는다 → 추론은 scripts/run_infer.sh
#
#   bash scripts/run_train.sh --detach                # tmux 세션 'ugrp-train'. 접속이 끊겨도 계속 돈다
#   bash scripts/run_train.sh --detach --from train   # 같은 Pod에서 중단된 단계부터 재개
#   bash scripts/run_train.sh --detach --resume       # 마지막 체크포인트에서 학습 이어가기 (--from train과 함께)
#
# 단계: setup → boundary → smoke → train → reference → pack
#   setup    사전 점검, 패키지 설치, 모델 다운로드 (scripts/pod_setup.sh)
#   boundary 미세조정 전 모델의 think 출력 경계가 학습 completion 형식과 같은지 대조
#   smoke    3 step만 학습해 OOM·설정 오류를 먼저 걸러낸다 (최장 4건). peak_vram_gb 출력
#   train    본 학습 → outputs/{agent}_think/final (LoRA 어댑터)
#   reference vLLM --parity가 비교할 기준값을 HF 4-bit로 잰다 (학습 2건의 NLL) → outputs/{agent}_think/parity_reference.json
#   pack     results_*.tar.gz 갱신 (어댑터, trainer_state, metrics, parity 기준값, 로그)
# 환경변수 PUSH_TO_HUB=<계정>/<repo> 를 주면 학습 종료 시 어댑터를 HF private repo로 백업한다.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

AGENT="generation"; FROM=""; DETACH=0; RESUME=""; ORIGINAL_ARGS=("$@")
while [ $# -gt 0 ]; do
  case "$1" in
    --agent) AGENT="$2"; shift 2 ;;
    --from) FROM="$2"; shift 2 ;;
    --resume) RESUME="--resume-from-checkpoint"; shift ;;
    --detach) DETACH=1; shift ;;
    *) echo "알 수 없는 인자: $1"; exit 1 ;;
  esac
done

if [ "$DETACH" -eq 1 ]; then
  FORWARD=(); for arg in "${ORIGINAL_ARGS[@]}"; do [ "$arg" != "--detach" ] && FORWARD+=("$arg"); done
  detach ugrp-train scripts/run_train.sh "${FORWARD[@]}"
  exit 0
fi

PUSH=""; [ -n "${PUSH_TO_HUB:-}" ] && PUSH="--push-to-hub $PUSH_TO_HUB"

run_stage() {
  case "$1" in
    setup)    ensure_setup ;;
    boundary) python -m ugrp.training.check_boundary --agent "$AGENT" ;;
    smoke)
      python -m ugrp.training.train --agent "$AGENT" --max-steps 3 --limit 4 --longest --output-dir outputs/_smoke
      python -c "import json; m = json.load(open('outputs/_smoke/metrics.json')); print(f'[SMOKE] peak_vram_gb = {m.get(\"peak_vram_gb\")} (reserved: {m.get(\"max_memory_reserved_gb\")} GB)')"
      rm -rf outputs/_smoke ;;
    train)    python -m ugrp.training.train --agent "$AGENT" $RESUME $PUSH ;;
    reference)
      # 실패해도 학습 결과는 묶어야 하므로 멈추지 않는다. 기준값이 없으면 --parity는 절대 기준(감소폭)만 본다
      python -m ugrp.inference.parity_reference --agent "$AGENT" \
        || echo "[WARN] parity 기준값을 재지 못했습니다. 나중에: python -m ugrp.inference.parity_reference --agent $AGENT" ;;
    pack)     bash scripts/pack_results.sh ;;
  esac
}

run_stages "bash scripts/run_train.sh --detach" "$FROM" setup boundary smoke train reference pack
print_download_notice
echo " 다음 단계(추론):  bash scripts/run_infer.sh --detach"
