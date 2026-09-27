#!/usr/bin/env bash
# Pod 부트스트랩: 사전 점검 → 패키지 설치 → 모델 다운로드. 번들 루트(ugrp/)에서 실행한다.
# 영구 저장소가 없으므로 세션마다 한 번 실행한다. run_train.sh / run_infer.sh가 첫 단계로 호출한다.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export UGRP_ROOT="$ROOT" PYTHONPATH="$ROOT/src" PYTHONIOENCODING=utf-8
MIN_VRAM_GB="${MIN_VRAM_GB:-75}"
MIN_DISK_GB="${MIN_DISK_GB:-30}"          # 번들 위치(/workspace): 체크포인트(<1GB × 4) + 결과·로그 + 여유
MIN_CACHE_DISK_GB="${MIN_CACHE_DISK_GB:-65}"  # 모델 캐시 위치(configs의 cache_dir): 가중치 ~62GB + 여유

echo "== 1. 사전 점검 =="
if [ -z "${HF_TOKEN:-}" ]; then
  echo "[INFO] HF_TOKEN 없음: 모델은 토큰 없이 받을 수 있다. 다만 비로그인 요청은 rate limit이 낮아 ~62GB 다운로드가"
  echo "       느리거나 429로 끊길 수 있다. 그럴 때는 export HF_TOKEN=hf_... 후 FORCE_SETUP=1로 다시 실행."
fi
command -v nvidia-smi >/dev/null || { echo "[ABORT] nvidia-smi가 없습니다. GPU Pod인지 확인하십시오."; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
VRAM_GB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | awk '{print int($1/1024)}')
if [ "$VRAM_GB" -lt "$MIN_VRAM_GB" ]; then
  echo "[ABORT] VRAM ${VRAM_GB}GB < ${MIN_VRAM_GB}GB. (smoke에서 잰 peak_vram_gb가 충분히 낮으면 MIN_VRAM_GB를 낮춰 실행)"; exit 1
fi
DISK_GB=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
if [ "$DISK_GB" -lt "$MIN_DISK_GB" ]; then
  echo "[ABORT] $ROOT 디스크 여유 ${DISK_GB}GB < ${MIN_DISK_GB}GB. Pod의 Volume Disk를 늘리십시오."; exit 1
fi
echo "VRAM ${VRAM_GB}GB / $ROOT 디스크 여유 ${DISK_GB}GB"

echo "== 2. 패키지 설치 =="
python -c "import torch; assert torch.cuda.is_available(), 'CUDA 사용 불가'; assert torch.cuda.is_bf16_supported(), 'bf16 미지원 GPU'; print('torch', torch.__version__, '/ CUDA', torch.version.cuda)"
pip install -q -r requirements.txt
mkdir -p "$ROOT/logs"
pip freeze > "$ROOT/logs/pip_freeze.txt"   # 결과물과 함께 내려받아 실제 설치 버전을 기록으로 남긴다

echo "== 3. 모델 다운로드 =="
MODEL_ID=$(python -c "from ugrp.common.config import load_config; print(load_config('generation')['model_id'])")
CACHE_DIR=$(python -c "from ugrp.common.config import load_config; print(load_config('generation')['cache_dir'])")
mkdir -p "$CACHE_DIR"
CACHE_DISK_GB=$(df -BG --output=avail "$CACHE_DIR" | tail -1 | tr -dc '0-9')
# FORCE_SETUP 재실행 시 이미 받은 가중치만큼 여유가 줄어 있으므로 그만큼 더해서 본다
CACHED_GB=$(du -s -BG "$CACHE_DIR" | cut -f1 | tr -dc '0-9')
if [ $(( CACHE_DISK_GB + CACHED_GB )) -lt "$MIN_CACHE_DISK_GB" ]; then
  echo "[ABORT] $CACHE_DIR 디스크 여유 ${CACHE_DISK_GB}GB (+ 받아 둔 모델 ${CACHED_GB}GB) < ${MIN_CACHE_DISK_GB}GB."
  echo "        Pod의 Container Disk를 늘리십시오."; exit 1
fi
echo "$CACHE_DIR 디스크 여유 ${CACHE_DISK_GB}GB (받아 둔 모델 ${CACHED_GB}GB)"
export HF_XET_HIGH_PERFORMANCE=1
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0   # xet 청크 캐시(기본 최대 10GB, ~/.cache)를 끈다: 한 번 받고 끝이라 쓸모가 없고 디스크만 먹는다
START=$(date +%s)
hf download "$MODEL_ID" --cache-dir "$CACHE_DIR" --quiet
echo "다운로드 완료: $MODEL_ID ($(( $(date +%s) - START ))초, $(du -sh "$CACHE_DIR" | cut -f1))"
