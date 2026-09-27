# 추론 백엔드 및 전체 옵션 가이드

이 문서는 학습 완료된 어댑터 가중치를 활용하여 지문을 생성하고 평가하는 추론 스크립트(`run_infer.sh`)의 사용법, 백엔드 차이점 및 전체 옵션을 다룹니다.

---

## 1. 백엔드 비교: vLLM vs Hugging Face

| 항목 | vLLM 백엔드 (`--backend vllm`, 권장) | Hugging Face 백엔드 (`--backend hf`, 기본값) |
| :--- | :--- | :--- |
| **모델 로딩** | BF16 원본 가중치 + 실행 중 LoRA 동적 마운트 | 4-bit NF4 양자화 베이스 + PEFT LoRA |
| **추론 속도** | **약 3~5배 빠름** (PagedAttention, 병렬 배치 최적화) | 느림 (순차 또는 소규모 배치 처리) |
| **다중 어댑터** | **지원** (`--adapters`로 1회 엔진 기동 후 여러 어댑터 순차 서빙) | 미지원 (어댑터 교체 시마다 모델 재로딩 필요) |
| **환경** | `/root/vllm-env` 전용 venv 필요 (자동 설치) | 기본 파이썬 환경 사용 |
| **권장 용도** | **대량 평가, 에포크별 체크포인트 일괄 비교, 프로덕션 서빙** | 디버깅, 단일 샘플 빠른 확인, VRAM 극도로 제한된 환경 |

---

## 2. 권장 실행 명령어 모음

모든 명령은 RunPod의 `/workspace/ugrp` 디렉토리에서 실행합니다.

### 2.1 다중 어댑터 일괄 추론·평가 (가장 권장됨)
vLLM 엔진을 1회만 올린 뒤 `final`, `checkpoint-37(ep1)`, `checkpoint-74(ep2)`를 순차적으로 추론하고 각각의 평가 리포트를 생성합니다.
```bash
bash scripts/run_infer.sh --detach --backend vllm --max-num-seqs 4 --max-num-batched-tokens 2048 \
    --adapters final=outputs/generation_think/final ep1=outputs/generation_think/checkpoint-37 ep2=outputs/generation_think/checkpoint-74
```
- **출력 결과**:
  - `results/generation_think_test_vllm_final.jsonl` / `results/eval_report_generation_test_vllm_final.json`
  - `results/generation_think_test_vllm_ep1.jsonl` / `results/eval_report_generation_test_vllm_ep1.json`
  - `results/generation_think_test_vllm_ep2.jsonl` / `results/eval_report_generation_test_vllm_ep2.json`

### 2.2 파인튜닝 전 순수 베이스 모델 추론 (비교 대조군)
동일한 test 19건에 대해 어댑터를 적용하지 않고 기본 모델의 성능을 측정합니다.
```bash
bash scripts/run_infer.sh --detach --backend vllm --max-num-seqs 4 --max-num-batched-tokens 2048 --no-adapter
```
- **출력 결과**: `results/generation_think_base_test_vllm.jsonl`, `results/eval_report_generation_base_test_vllm.json`

### 2.3 단일 어댑터 추론
```bash
# vLLM 백엔드로 final 어댑터 추론
bash scripts/run_infer.sh --detach --backend vllm --max-num-seqs 4 --max-num-batched-tokens 2048

# Hugging Face 백엔드로 final 어댑터 추론 (4-bit PEFT)
bash scripts/run_infer.sh --detach
```

### 2.4 사전 검증 (Smoke & Parity Check)
vLLM 엔진이 정상 동작하는지, LoRA 가중치가 올바르게 반영되는지 1~2건으로 확인합니다.
```bash
# 1) Parity 점검: 학습 데이터 2건의 NLL loss가 HF 기준값과 일치하는지 확인
bash scripts/run_infer.sh --backend vllm --parity

# 2) 1건 Smoke 테스트: 태그(<channel|>, <turn|>)로 정상 종료되는지 확인
bash scripts/run_infer.sh --backend vllm --limit 1 --greedy
```

### 2.5 기타 유용한 실행 형태
```bash
# 검증 셋(valid 18건)으로 과적합 여부 점검
bash scripts/run_infer.sh --detach --split valid

# 앞의 3건만 빠르게 확인
bash scripts/run_infer.sh --detach --limit 3

# 직접 작성한 외부 JSONL 파일로 추론 (목표 분량 1800자 지정)
bash scripts/run_infer.sh --detach --input data/custom_input.jsonl --output results/custom_output.jsonl --target-length 1800

# 구조 에이전트 출력(answer)을 생성 에이전트의 입력으로 연쇄 실행
export PYTHONPATH=$PWD/src && python -m ugrp.inference.run_pipeline --split test
```

---

## 3. 추론 파이프라인 단계

`run_infer.sh`는 다음 단계를 거쳐 실행됩니다:

| 단계 | 동작 내용 |
| :--- | :--- |
| `setup` | Pod 환경 및 모델 가중치 캐시 확인 (새 Pod일 경우 다운로드) |
| `vllm_setup` | `--backend vllm`일 경우 `/root/vllm-env` 가상환경 설치 ([`scripts/setup_vllm.sh`](../scripts/setup_vllm.sh)) |
| `adapter` | 지정한 어댑터 가중치 디렉토리가 존재하는지 사전 확인 (없으면 HF Hub 검색) |
| `infer` | 배치 단위로 추론을 수행하고 `results/*.jsonl`에 즉시 기록. **(이미 결과 파일에 존재하는 ID는 자동으로 건너뜁니다)** |
| `report` | 분할 데이터셋(`valid`, `test`) 추론인 경우 `ugrp.eval_report`를 실행하여 형식 지표 보고서 JSON 생성 |
| `pack` | 최종 결과를 압축 파일(`results_{번들ID}.tar.gz`)에 자동 갱신 |

---

## 4. 전체 CLI 옵션 상세 명세

| 옵션명 | 기본값 | 설명 |
| :--- | :-: | :--- |
| `--backend` | `hf` | 추론 엔진 선택 (`hf` \| `vllm`) |
| `--adapter <경로\|HF repo>` | `outputs/{agent}_think/final` | 사용할 LoRA 어댑터 경로 또는 Hugging Face 레포지토리 이름 |
| `--adapters 이름=경로 …` | — | **[vllm 전용]** 1회 엔진 기동으로 여러 어댑터를 순차 추론할 때 사용 |
| `--no-adapter` | — | 어댑터를 올리지 않고 베이스 모델만으로 추론 (결과 파일명에 `_base` 추가) |
| `--split` | `test` | 평가할 데이터셋 분할 (`valid` \| `test`) |
| `--input <파일>` / `--output <파일>` | 자동 지정 | 입출력 파일 경로를 직접 지정 (지정 시 `report` 단계는 건너뜀) |
| `--target-length N` | 파일 지정값 | 모든 샘플의 목표 글자 수를 N자로 통일하여 프롬프트 재구성 |
| `--limit N` | 전체 | 앞에서부터 N건만 잘라서 추론 |
| `--batch-size N` | 20 | **[hf 전용]** 한 번에 생성할 배치 크기 (OOM 발생 시 자동 절반 감소) |
| `--max-num-seqs N` | — | **[vllm 전용]** 동시 처리 시퀀스 수 (Gemma 4 31B는 **4** 권장) |
| `--max-num-batched-tokens N` | — | **[vllm 전용]** 배치당 최대 토큰 수 (**2048** 권장) |
| `--gpu-mem <비율>` | 0.92 | **[vllm 전용]** GPU 메모리 할당 비율 |
| `--max-model-len N` | 12288 | **[vllm 전용]** 컨텍스트 길이 (test 프롬프트 5,947 + 생성 6,144 토큰 지원) |
| `--greedy` | False | Greedy 디코딩 적용 (온도 0) |
| `--temperature T` | 1.0 | 샘플링 온도 (`configs/base.yaml` 기본값) |
| `--seed N` | 42 | 재현성을 위한 랜덤 시드 |
| `--max-new-tokens N` | 6144 | 최대 생성 토큰 수 |
| `--structure-from <결과.jsonl>` | — | 이전 구조 에이전트의 추론 결과(`answer`)를 생성 에이전트의 입력으로 주입 |
| `--parity` | — | **[vllm 전용]** 생성을 수행하지 않고 NLL 손실을 측정하여 LoRA 정합성 검증 |

---

## 5. 생성 및 토큰 제어 규격

- **생성 파라미터**: `temperature: 1.0`, `top_p: 0.95`, `top_k: 64`, `max_new_tokens: 6144`
- **종료(Stop) 토큰**: `[<eos>(1), <turn|>(106)]`
  - 토크나이저의 기본 eos 토큰은 `<eos>`뿐이므로, 대화 턴 종료 토큰인 `<turn|>`을 추가 Stop 토큰으로 지정합니다.
  - 응답 파서([`src/ugrp/common/gemma_format.py`](../src/ugrp/common/gemma_format.py))는 첫 번째 `<turn|>` 이후의 텍스트를 제거하여 깔끔한 응답만 추출합니다.
- **출력 JSONL 행 구조**:
  - `{id, target_length, thinking, answer, reference, raw, backend, tokens, finish_reason}`
  - vLLM 실행 시 `{adapter, sampling}` 메타데이터가 추가로 기록됩니다.

