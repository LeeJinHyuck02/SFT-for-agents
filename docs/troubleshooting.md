# 문제 해결 (Troubleshooting)

이 문서는 학습 및 추론 과정에서 자주 발생하는 오류 현상과 원인별 조치 방법을 정리한 가이드입니다.

---

## 1. 프로세스 및 세션 관리

### `[ABORT] tmux 세션 'ugrp-train'이 이미 있습니다`
- **원인**: 백그라운드에서 이전 학습/추론 작업이 여전히 돌고 있거나, 비정상 종료된 세션이 남아있음.
- **조치**:
  1. 현재 실행 상태 확인: `tmux attach -t ugrp-train`
  2. 이미 완료되었거나 불필요한 세션인 경우 삭제: `tmux kill-session -t ugrp-train`

### 원격 접속(SSH/웹)이 끊겨 상태를 알 수 없을 때
- **조치**: 스크립트는 `tmux` 백그라운드에서 계속 실행 중입니다. 재접속 후 다음 명령으로 확인합니다:
  ```bash
  cat logs/STATE                  # 최근 완료된 단계 및 소요 시간
  tail -n 50 logs/ugrp-train.log  # 최근 로그 출력
  tmux ls                         # 활성 세션 목록 확인
  ```

### 스크립트 대신 파이썬 모듈을 직접 실행하고 싶을 때
- **조치**: 모듈 경로(`src/`)를 환경변수에 등록하고 실행합니다:
  ```bash
  cd /workspace/ugrp
  export PYTHONPATH=$PWD/src
  python -m ugrp.training.train --help
  ```

---

## 2. 환경 설정 및 디스크 문제

### `$'\r': command not found` (윈도우 줄바꿈 CRLF 에러)
- **원인**: 윈도우 환경에서 편집된 셸 스크립트의 `\r`(CR) 문자가 리눅스 환경에서 인식되지 않음.
- **조치**:
  ```bash
  sed -i 's/\r$//' scripts/*.sh
  ```

### 로컬 실행 시 `No module named 'ugrp'` 발생
- **조치**: 루트 디렉토리에서 패키지를 editable 모드로 설치합니다:
  ```bash
  pip install -e .
  # 또는 임시로 PYTHONPATH 설정 (PowerShell)
  $env:PYTHONPATH = "src"
  ```

### `setup`: 디스크 여유 공간 부족
- **원인**: RunPod 인스턴스의 볼륨 또는 컨테이너 디스크 크기 미달.
- **조치**:
  - `/workspace` 여유 공간 30GB 미만일 때: Pod 볼륨을 30GB 이상으로 증설.
  - `/root/model_cache` 여유 공간 65GB 미만일 때: Pod 생성 시 Container Disk를 **80GB 이상**(vLLM 사용 시 100GB)으로 설정.

### 모델 다운로드가 너무 느리거나 `HTTP 429 Too Many Requests` 발생
- **원인**: Hugging Face 인증 토큰 부재로 다운로드 rate limit에 도달함.
- **조치**: 토큰을 설정하고 강제 재설치를 실행합니다:
  ```bash
  export HF_TOKEN=hf_...
  FORCE_SETUP=1 bash scripts/run_train.sh --detach
  ```

---

## 3. 학습 단계 오류

### `boundary`: `[MISMATCH]` 발생
- **원인**: 미세조정 전 베이스 모델이 출력한 태그와 규격(`gemma_format.py`의 `THINK_OPEN`, `COMPLETION_SUFFIX`)이 불일치함.
- **조치**:
  1. 로그의 `[RAW ...]` 출력문을 확인하고 `gemma_format.py`의 정규식/태그 상수 수정.
  2. `UGRP2/data`의 03번 `convert` 스크립트도 동일한 태그로 수정 후 재생성 및 재복사 → `make_bundle.py` 다시 실행.
  3. 단순히 모델의 사고 생성이 길어서 중간에 끊긴 경우라면 `--max-new-tokens 8192`로 늘려서 재시도.

### 학습 도중 `CUDA Out of Memory (OOM)` 발생
- **원인**: 특정 긴 샘플에서 VRAM 한계(80GB) 초과.
- **조치**:
  1. `configs/base.yaml`의 `max_len`을 현재 실측된 최대 토큰 수 바로 위로 낮춤.
  2. 또는 LoRA `target_modules`를 `(q_proj|v_proj)`로 축소하여 메모리 절약.

### `adapter`: 어댑터가 없습니다
- **조치**:
  - 이전 단계에서 학습이 정상 완료되었는지 확인합니다.
  - 새 Pod라면 이전 Pod의 `results_*.tar.gz`를 가져와 `tar -xzf results_*.tar.gz`로 `outputs/generation_think/final` 디렉토리를 복원합니다.

---

## 4. 추론 및 생성 단계 오류

### [Hugging Face] `CUDA 메모리 부족 → 배치 크기를 N으로`
- **설명**: 정상적인 방어 로직입니다. OOM 감지 시 자동으로 배치 크기를 절반으로 줄여 재시도합니다.
- **조치**: 경고가 반복될 경우 처음부터 옵션으로 `--batch-size 8`을 지정합니다.

### 생성 도중 `finish_reason`이 `length`로 끝나며 내용이 잘림
- **조치**:
  - 같은 단어가 무한 반복되는 형태(`가 가 가...`)인 경우: `--greedy` 또는 `--temperature 0.7`을 지정하여 붕괴 방지.
  - 정상적인 사고 과정이 매우 길어서 발생한 경우: `--max-new-tokens 8192`로 확장.

### 추론 스크립트를 다시 돌렸는데 아무것도 생성되지 않고 바로 끝남
- **설명**: `run_infer.sh`는 이미 결과 파일에 존재하는 `id`를 중복 실행하지 않고 건너뜁니다.
- **조치**: 새로 생성하려면 기존 `results/*.jsonl` 파일을 삭제하거나 `--output results/new_result.jsonl`과 같이 다른 출력 경로를 지정합니다.

---

## 5. vLLM 전용 오류 및 대응

### `--parity` 검증 시 "기준값을 쓰지 않습니다" 경고
- **원인**: `outputs/{agent}_think/parity_reference.json` 파일이 없거나 다른 학습 데이터셋으로 생성된 경우.
- **조치**: 절대 감소폭 기준(base 대비 어댑터 NLL 0.3 이상 감소)으로만 판정됩니다. 정밀 검증이 필요한 경우 학습 환경에서 `python -m ugrp.inference.parity_reference`를 먼저 실행합니다.

### `--parity` 검증 FAIL 발생
- **원인**:
  1. 감소폭이 0.3 미만인 경우: LoRA 정규식 `target_modules`가 vLLM 모델 레이어에 제대로 바인딩되지 않음 ([issue/0919/0919_inf.md](../issue/0919/0919_inf.md) 대안 1 참조).
  2. NLL 손실 절대값이 HF 기준값과 0.05 이상 차이나는 경우: bf16과 4-bit 베이스 차이 또는 어댑터 로딩 오류.

### `import vllm` 시 `libcudart.so.13` ImportError
- **원인**: Pod의 PyTorch 빌드가 CUDA 13.0이 아닌 다른 버전(cu126 등)으로 설치된 경우.
- **조치**:
  - 수정된 `scripts/setup_vllm.sh`를 사용하여 재설치: `FORCE_SETUP=1 bash scripts/setup_vllm.sh`
  - 호스트 머신의 드라이버가 CUDA 13.0 미만인 Pod에서는 vLLM 0.29.0을 구동할 수 없습니다 ([issue/0925/0925.md](../issue/0925/0925.md) 부록 A-1).

### `FileNotFoundError: 'ninja'` (EngineCore 초기화 실패)
- **원인**: FlashInfer가 샘플링 커널을 JIT 컴파일할 때 PATH에서 `ninja`를 찾지 못함.
- **조치**:
  - `run_infer.sh`는 내부적으로 PATH를 자동 주입합니다.
  - vLLM 파이썬 코드를 셸에서 직접 실행할 때는 사전에 환경변수를 등록해야 합니다:
    ```bash
    export PATH=/root/vllm-env/bin:$PATH
    ```

### `KV cache is needed, which is larger than the available` (엔진 초기화 실패)
- **원인**: Gemma 4 31B 모델의 파라미터가 거대하여 기본 설정 시 PagedAttention KV Cache를 위한 VRAM이 부족함.
- **조치**:
  - 반드시 아래 2개 옵션을 포함하여 실행합니다:
    ```bash
    --max-num-seqs 4 --max-num-batched-tokens 2048
    ```
    (활성화 메모리와 CUDA 그래프 할당 메모리가 절약되어 약 2GB 이상의 VRAM이 KV Cache로 확보됩니다.)
  - ⚠️ 주의: `--max-model-len`을 줄이면 긴 프롬프트(5.9k 토큰)가 잘리므로 줄이지 마십시오.

### `Maximum concurrency` 경고 (4 미만)
- **설명**: 오류가 아닙니다. Concurrency가 2.4x 수준이어도 19건 전체 추론이 약 11분 만에 완료됩니다. 엔진이 정상 기동되고 토큰이 출력되면 그대로 사용하시면 됩니다.

