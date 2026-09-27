#!/usr/bin/env bash
# 내려받을 결과물을 하나의 tar.gz로 묶는다. Pod를 끄면 전부 사라지므로 학습이 하나 끝날 때마다 갱신한다.
# 포함: final 어댑터, epoch별 체크포인트의 어댑터 가중치(adapter_model.safetensors, adapter_config.json), trainer_state/metrics,
#       parity 기준값, 추론 결과, 로그, MANIFEST.
# 제외: 체크포인트의 학습 재개용 상태(optimizer.pt, scheduler.pt, rng_state 등), 모델 캐시.
# 체크포인트 어댑터는 하나에 약 245MB(압축 후 약 200MB)다. PACK_CHECKPOINTS=0이면 final만 넣는다 (2026-09-25, issue/0925/0925_plan.md 7.3절).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
BUNDLE_ID=$(python -c "import json; print(json.load(open('MANIFEST.json'))['bundle'])" 2>/dev/null || echo "nomanifest")
OUT="$ROOT/results_${BUNDLE_ID}.tar.gz"
PACK_CHECKPOINTS="${PACK_CHECKPOINTS:-1}"

FILES=()
for f in MANIFEST.json logs results; do [ -e "$f" ] && FILES+=("$f"); done
for run in outputs/*/; do
  [ -d "$run" ] || continue
  for f in final trainer_state.json metrics.json parity_reference.json; do [ -e "$run$f" ] && FILES+=("$run$f"); done
  [ "$PACK_CHECKPOINTS" != "0" ] || continue
  for ckpt in "$run"checkpoint-*/; do
    [ -d "$ckpt" ] || continue
    for f in adapter_model.safetensors adapter_config.json; do [ -e "$ckpt$f" ] && FILES+=("$ckpt$f"); done
  done
done

tar -czf "$OUT.tmp" "${FILES[@]}" && mv "$OUT.tmp" "$OUT"
echo "[PACK] $OUT ($(du -h "$OUT" | cut -f1))"
