#!/usr/bin/env bash
# vLLM 추론 환경을 학습 환경과 분리된 venv에 설치한다. run_infer.sh --backend vllm이 setup 뒤에 호출한다.
# venv는 컨테이너 디스크(/root)에 만든다. /workspace(네트워크 볼륨)는 쓰기가 느려 쓰지 않는다.
# 컨테이너 디스크는 Pod를 끄면 사라지므로 새 Pod에서는 다시 설치된다.
# 계획: issue/0919/0919_inf.md. 2026-09-25 수정(torch CUDA 빌드 명시, 설치 전 드라이버 검사): issue/0925/0925.md 부록 A-1
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
VLLM_ENV="${VLLM_ENV:-/root/vllm-env}"
MARKER="$VLLM_ENV/.ugrp_ready"
# PyPI의 vllm 0.29.0 wheel은 CUDA 13 빌드다(libcudart.so.13). torch도 같은 CUDA 빌드여야 하고, 드라이버는 그 버전 이상이어야 한다.
# vllm 버전을 바꾸면 그 wheel의 CUDA 빌드에 맞춰 TORCH_INDEX와 REQUIRED_CUDA를 함께 바꾼다 (예: cu129 빌드면 .../whl/cu129, 12.9).
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
REQUIRED_CUDA="${REQUIRED_CUDA:-13.0}"

if [ -f "$MARKER" ] && [ -z "${FORCE_SETUP:-}" ]; then
  echo "[VLLM] 이미 설치됨: $VLLM_ENV (다시 하려면 FORCE_SETUP=1)"
  exit 0
fi

echo "== 1. 드라이버 점검 =="
DRIVER_CUDA=$(nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | awk '{print $3}' || true)
[ -n "$DRIVER_CUDA" ] || { echo "[ABORT] nvidia-smi에서 드라이버의 CUDA 버전을 읽지 못했습니다."; exit 1; }
echo "드라이버가 지원하는 CUDA: $DRIVER_CUDA (vllm wheel 요구: $REQUIRED_CUDA 이상)"
# 설치(약 9GB)를 시작하기 전에 막는다. 예전에는 설치 뒤 "3. 확인"에서야 걸렸다
lowest=$(printf '%s\n%s\n' "$DRIVER_CUDA" "$REQUIRED_CUDA" | sort -V | head -1)
if [ "$lowest" = "$DRIVER_CUDA" ] && [ "$DRIVER_CUDA" != "$REQUIRED_CUDA" ]; then
  echo "[ABORT] 드라이버는 CUDA $DRIVER_CUDA까지만 지원하는데 requirements-vllm.txt의 vllm wheel은 CUDA $REQUIRED_CUDA 빌드입니다."
  echo "        드라이버가 더 새로운 Pod를 쓰거나, vllm 버전과 TORCH_INDEX·REQUIRED_CUDA를 드라이버에 맞는 것으로 바꾸십시오."
  exit 1
fi

echo "== 2. 설치 ($VLLM_ENV) =="
command -v uv >/dev/null || pip install -q uv
# 캐시를 끄지 않으면 wheel 사본이 컨테이너 디스크를 5GB 이상 더 쓴다.
# UV_TORCH_BACKEND=auto는 쓰지 않는다: uv 0.9.0은 cu130을 몰라 드라이버가 13.0이어도 cu126 torch를 깔고, 그러면 import vllm이
# libcudart.so.13 ImportError로 죽는다 (2026-09-25). 대신 torch 인덱스를 명시한다.
export UV_NO_CACHE=1
REINSTALL=()
if [ -x "$VLLM_ENV/bin/python" ]; then
  # 마커 없이 venv만 남아 있으면(예: 예전 스크립트가 cu126 torch를 깐 경우) torch 계열만 다시 설치한다.
  # vllm의 torch==2.13.0 요구는 2.13.0+cu126도 만족하므로 그냥 두면 바뀌지 않는다
  REINSTALL=(--reinstall-package torch --reinstall-package torchvision --reinstall-package torchaudio)
else
  uv venv "$VLLM_ENV" --python 3.12
fi
# --index-url: torch·torchvision·torchaudio를 PyTorch 인덱스에서 받는다. --extra-index-url: vllm 등 나머지는 PyPI.
# --index-strategy unsafe-best-match: 두 인덱스에 모두 있는 패키지(torch)는 가장 높은 버전(2.13.0+cu130 > 2.13.0)을 고른다.
#   기본 전략(first-index)은 uv가 extra-index를 먼저 보므로 PyPI의 torch가 설치된다.
uv pip install --python "$VLLM_ENV/bin/python" -r requirements-vllm.txt \
  --index-url "$TORCH_INDEX" --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match \
  ${REINSTALL[@]+"${REINSTALL[@]}"}

echo "== 3. 확인 =="
TORCH_CUDA=$("$VLLM_ENV/bin/python" -c "import torch; print(torch.version.cuda)")
lowest=$(printf '%s\n%s\n' "$DRIVER_CUDA" "$TORCH_CUDA" | sort -V | head -1)
if [ "$lowest" = "$DRIVER_CUDA" ] && [ "$DRIVER_CUDA" != "$TORCH_CUDA" ]; then
  echo "[ABORT] vLLM venv의 torch는 CUDA $TORCH_CUDA 빌드인데 드라이버는 CUDA $DRIVER_CUDA까지만 지원합니다."
  echo "        드라이버가 더 새로운 Pod를 쓰거나, requirements-vllm.txt의 vllm 버전을 드라이버에 맞는 것으로 바꾸십시오."
  exit 1
fi
"$VLLM_ENV/bin/python" -c "
import torch, vllm
assert torch.cuda.is_available(), 'vLLM venv에서 CUDA를 쓸 수 없습니다'
print('vllm', vllm.__version__, '/ torch', torch.__version__, '/ CUDA', torch.version.cuda)"
du -sh "$VLLM_ENV" | awk '{print "venv 크기:", $1}'
touch "$MARKER"
