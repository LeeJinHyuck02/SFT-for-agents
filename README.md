# UGRP2 — 수능 국어 비문학 지문 생성 에이전트

`google/gemma-4-31B-it` 모델을 think mode 기반 **QLoRA SFT**로 파인튜닝하여, **원본 문헌 + 문단별 구조 데이터 → 수능 비문학 지문**을 작성하는 에이전트 파이프라인입니다.

---

## 📌 전체 파이프라인 개요

| 단계 | 위치 | 실행 명령 | 입력 데이터 | 출력 결과물 |
| :-: | :--- | :--- | :--- | :--- |
| **1** | 로컬 | `python scripts/make_bundle.py` | `data/SFT_think_en_*.jsonl` (3종) | `bundles/ugrp_bundle_{시각}.zip` |
| **2** | Pod | `bash scripts/run_train.sh --detach` | 번들 zip 파일 | `outputs/generation_think/final` (어댑터) |
| **3** | Pod | `bash scripts/run_infer.sh --detach ...` | 어댑터 + `data/SFT_think_en_test.jsonl` | `results/*.jsonl`, `results/eval_report_*.json` |
| **4** | 로컬/Pod | `python -m ugrp.eval_report <결과.jsonl>` | 생성 결과 JSONL 파일 | 형식 준수율 및 복사율 평가 리포트 |

> [!NOTE]
> **데이터 생성 분리 원칙**: 이 저장소는 데이터를 직접 합성하지 않습니다. 지문 정제, 사고 과정 합성·검수·병합은 전부 [`UGRP2/data`](../data/README.md)에서 수행하며, 본 저장소에는 완성된 파일 3개만 복사하여 사용합니다.

---

## 🚀 빠른 실행 가이드 (Quick Start)

### 1) [로컬] 가상환경 구축 & 데이터 준비

```powershell
# 1. 가상환경 생성 및 활성화 (Linux/macOS: source .venv/bin/activate)
python -m venv .venv
.venv\Scripts\activate

# 2. 의존성 패키지 설치
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .

# 3. 데이터 준비 (UGRP2/data 에서 만든 3개 파일을 data/ 폴더에 복사)
#    - SFT_think_en_train_gemma4.jsonl
#    - SFT_think_en_valid.jsonl
#    - SFT_think_en_test.jsonl
```

### 2) [로컬] 번들 패키징

```powershell
# 데이터 형식 검증 및 RunPod 업로드용 단일 번들 생성
python scripts/make_bundle.py
```
→ `bundles/ugrp_bundle_{YYYYMMDD_HHMM}.zip` 생성 (코드 + 데이터 + `MANIFEST.json`).  
데이터 형식에 오류가 있거나 복사가 누락된 경우 `[ABORT]`로 즉시 중단됩니다.

### 3) [Pod] 번들 전송 및 압축 해제

*권장 Pod 사양: A100 80GB 1장, Container Disk 80GB(vLLM 시 100GB), `/workspace` 30GB 이상*

```powershell
# 로컬에서 번들 전송 (또는 JupyterLab 웹 업로드)
scp -P <포트> -i $HOME\.ssh\id_ed25519 bundles\ugrp_bundle_*.zip root@<IP>:/workspace/
```

```bash
# Pod 접속 후 압축 해제
cd /workspace && unzip ugrp_bundle_*.zip && cd ugrp
export HF_TOKEN=hf_...   # 모델 다운로드 속도 향상을 위해 설정 권장
```

### 4) [Pod] 학습 실행 (`run_train.sh`)

```bash
# 전체 학습 파이프라인 백그라운드 실행 (setup → boundary → smoke → train → reference → pack, 약 4시간)
bash scripts/run_train.sh --detach

# 진행 상태 확인
tail -f logs/ugrp-train.log     # 실시간 로그
cat logs/STATE                  # 완료된 단계 및 소요 시간
```
→ 학습 완료 시 어댑터(`outputs/generation_think/final`), 에포크별 체크포인트, 기준값(`parity_reference.json`)이 `results_{번들ID}.tar.gz`로 자동 패킹됩니다.

### 5) [Pod] 추론 및 평가 (`run_infer.sh`)

추론 백엔드는 **vLLM(권장, 고속·다중 어댑터 서빙)**과 **Hugging Face(기본 4-bit)**를 지원합니다.  
Gemma 4 31B 모델의 KV Cache 부족 방지를 위해 `--max-num-seqs 4 --max-num-batched-tokens 2048`을 기본 지정합니다.

#### ① [추천] 여러 어댑터(final 및 에포크별 체크포인트) 일괄 추론·평가
```bash
# final 및 checkpoint-37(ep1), checkpoint-74(ep2)를 vLLM 엔진 1회 기동으로 연속 추론
bash scripts/run_infer.sh --detach --backend vllm --max-num-seqs 4 --max-num-batched-tokens 2048 \
    --adapters final=outputs/generation_think/final ep1=outputs/generation_think/checkpoint-37 ep2=outputs/generation_think/checkpoint-74
```
→ 어댑터별 결과: `results/generation_think_test_vllm_{final,ep1,ep2}.jsonl` 및 평가 리포트 `results/eval_report_generation_test_vllm_{final,ep1,ep2}.json`

#### ② [비교 대조군] 파인튜닝 전 순수 베이스 모델 추론
```bash
# 어댑터를 적용하지 않고 동일한 조건(test 19건)에서 베이스 모델 성능 측정
bash scripts/run_infer.sh --detach --backend vllm --max-num-seqs 4 --max-num-batched-tokens 2048 --no-adapter
```
→ `results/generation_think_base_test_vllm.jsonl`, `results/eval_report_generation_base_test_vllm.json`

#### ③ [단일] final 어댑터만 단독 추론할 때
```bash
bash scripts/run_infer.sh --detach --backend vllm --max-num-seqs 4 --max-num-batched-tokens 2048   # vLLM final 추론
# 또는
bash scripts/run_infer.sh --detach                                                                 # HF 백엔드 final 추론
```

#### ④ 진행 확인
```bash
tail -f logs/ugrp-infer.log             # 추론 진행 상황 실시간 확인 (tmux 백그라운드)
cat logs/STATE                          # 완료된 단계 및 소요 시간 확인
```

### 6) [결과] 결과물 수령 및 Pod 종료

```powershell
# 로컬 머신에서 결과 파일 다운로드
scp -P <포트> -i $HOME\.ssh\id_ed25519 root@<IP>:/workspace/ugrp/results_*.tar.gz .
tar -xzf results_*.tar.gz               # outputs/, results/, logs/ 복원
```

> [!CAUTION]
> **RunPod는 자동으로 종료되지 않습니다.**  
> 압축 파일 다운로드가 완료되고 정상적으로 풀리는 것을 확인한 뒤, RunPod 콘솔에서 인스턴스를 반드시 **Terminate**해 주십시오.

---

## 📖 상세 기술 문서 안내

세부 사양, 설정 근거 및 문제 해결 방법은 아래 문서에서 확인할 수 있습니다:

- 📊 **[데이터 명세 및 프롬프트 규격](docs/data.md)**: 데이터 파일 형식, Chat Template과 완성문(Completion) 직접 조립 근거, 지문별 가변 목표 분량 규칙, 번들 검사 알고리즘
- 🏋️ **[학습 파이프라인 및 설정 상세](docs/training.md)**: RunPod 요구 사양, 6단계 파이프라인 동작 방식, LoRA/최적화 하이퍼파라미터 설계 근거, Parity Reference 및 형식 학습 게이지
- ⚡ **[추론 백엔드 및 전체 옵션](docs/inference.md)**: vLLM vs HF 백엔드 심층 비교, 전체 CLI 파라미터 테이블, Stop 토큰 처리, 에이전트 파이프라인 연쇄
- 📈 **[형식 평가지표(eval_report) 가이드](docs/evaluation.md)**: 15개 형식 준수 지표 계산식 및 해석 기준, 원문 및 구조 복사율(`source_copy_ratio`)의 정성 평가 대체 원리
- 🛠️ **[문제 해결 (Troubleshooting)](docs/troubleshooting.md)**: tmux 세션 충돌, 디스크 부족, `boundary` 불일치, CUDA OOM, vLLM `libcudart` 및 KV Cache 부족 등 20개 상황별 대응법

---

## 📂 코드베이스 구조

```text
SFT-for-agents/
├── configs/
│   ├── base.yaml              # 공통 학습/모델/양자화/생성 단일 출처 설정
│   ├── generation.yaml        # 생성 에이전트 전용 오버라이드
│   └── structure.yaml         # 구조 에이전트 전용 오버라이드
├── docs/                      # 📖 상세 가이드 및 레퍼런스 문서
├── src/ugrp/
│   ├── common/                # 프롬프트(prompts), 태그 포맷(gemma_format), 입출력 유틸
│   ├── training/              # train.py, check_boundary.py
│   ├── inference/             # infer.py (HF), infer_vllm.py (vLLM), inputs.py, run_pipeline.py
│   └── eval_report.py         # 형식 지표 및 복사율 집계 모듈
├── scripts/
│   ├── make_bundle.py         # [로컬] 데이터 검사 및 압축
│   ├── run_train.sh           # [Pod] 전체 학습 파이프라인 제어 (tmux)
│   ├── run_infer.sh           # [Pod] 추론 및 평가 제어 (tmux)
│   ├── pod_setup.sh           # [Pod] 환경 점검 및 모델 다운로드
│   ├── setup_vllm.sh          # [Pod] vLLM 전용 환경 구성
│   └── pack_results.sh        # [Pod] 결과물 자동 압축 패킹
├── tests/                     # 프롬프트 및 포맷팅 무결성 단위 테스트
├── pyproject.toml             # 패키지 설정
└── requirements*.txt          # 패키지 의존성 목록
```
