# UGRP2 — 수능 국어 비문학 지문 생성 에이전트

`google/gemma-4-31B-it`를 think mode로 QLoRA SFT하여, **원본 문헌 + 문단별 구조 데이터 → 수능 비문학 지문**을 쓰게 한다.
이 문서는 명령어별 명세다: 무엇을 입력하면, 어떤 데이터와 근거로, 무엇을 내놓는가. (이전 방식의 코드는 [legacy/](legacy/))

## 빠른 실행: 번들 만들기 → RunPod에서 학습·추론

**1) 로컬 — 번들 만들기** (`UGRP2/agents`에서)

```powershell
.venv\Scripts\activate                  # 가상환경 먼저 (Linux/macOS: source .venv/bin/activate). 없으면 2.0절에서 만든다
# 먼저 UGRP2/data에서 만든 파일 3개가 data/에 있어야 한다 (2.1절)
#   SFT_think_en_train_gemma4.jsonl, SFT_think_en_valid.jsonl, SFT_think_en_test.jsonl
python scripts/make_bundle.py
```

→ `bundles/ugrp_bundle_{YYYYMMDD_HHMM}.zip` (코드 + 위 3개 파일 + `MANIFEST.json`, 수 MB). 데이터 형식이 어긋나거나 복사가 빠졌으면 `[ABORT]`로 멈춘다.

**2) Pod로 옮기기** — Pod 사양은 3절 (80GB GPU 1장, Container Disk 80GB, `/workspace` 볼륨 30GB)

```powershell
scp -P <포트> -i $HOME\.ssh\id_ed25519 bundles\ugrp_bundle_*.zip root@<IP>:/workspace/     # 또는 JupyterLab 업로드, runpodctl send
```

```bash
cd /workspace && unzip ugrp_bundle_*.zip && cd ugrp      # 이후 명령은 모두 /workspace/ugrp 에서
export HF_TOKEN=hf_...                                    # 권장 (모델 다운로드 rate limit)
```

**3) Pod — 학습**

```bash
bash scripts/run_train.sh --detach      # setup → boundary → smoke → train → reference → pack, 약 4시간
tail -f logs/ugrp-train.log             # 진행 확인. 끝난 단계는 cat logs/STATE
```

→ `outputs/generation_think/final/` (LoRA 어댑터), `outputs/generation_think/checkpoint-37/74/111/148/185` (epoch별), `outputs/generation_think/parity_reference.json` (vLLM 점검용 기준값 + 형식 학습 게이지), `results_{번들ID}.tar.gz`

**4) Pod — 추론·평가**

```bash
bash scripts/run_infer.sh --detach      # final 어댑터로 test 19건 생성 → 형식 지표 집계 → pack
```

→ `results/generation_think_test.jsonl` (사고 과정 + 지문), `results/eval_report_generation_test.json`, 갱신된 `results_{번들ID}.tar.gz`

```bash
bash scripts/run_infer.sh --detach --backend vllm --parity            # 어댑터가 bf16에서도 붙는지 (로그의 [PARITY] 줄이 OK)
bash scripts/run_infer.sh --detach --backend vllm --limit 1 --greedy  # 1건 smoke. 엔진이 뜨고 사고·지문이 <channel|>·<turn|>로 닫히는지
bash scripts/run_infer.sh --detach --backend vllm                     # test 19건. 엔진 초기화가 KV cache 부족으로 실패하면 --max-num-seqs 4 --max-num-batched-tokens 2048
tail -f logs/ugrp-infer.log                                           # 진행 확인
```

→ `results/generation_think_test_vllm.jsonl`, `results/eval_report_generation_test_vllm.json`. 어댑터 여러 개 비교, 샘플링 옵션, 오류는 4.2절과 7절.

**5) 결과 내려받기 → Pod 종료**

```powershell
scp -P <포트> -i $HOME\.ssh\id_ed25519 root@<IP>:/workspace/ugrp/results_*.tar.gz .
tar -xzf results_*.tar.gz               # outputs/, results/, logs/ 복원
```

내려받을 것은 `results_*.tar.gz` 하나다. **Pod는 자동으로 꺼지지 않으므로** 압축이 풀리는지 확인한 뒤 콘솔에서 Terminate한다. 중단됐을 때는 `bash scripts/run_train.sh --detach --from train --resume`, 옵션과 단계별 동작은 4절, 오류는 7절.

## 0. 전체 순서

| # | 어디서 | 명령 | 입력 | 출력 |
| :-: | :--- | :--- | :--- | :--- |
| 1 | 로컬 `UGRP2/data` | `03 merge` → `03 convert` ([../data/README.md](../data/README.md)) | `SFTdata.jsonl`, `reasoning_en.jsonl` | 완성 파일 3개 (1절) |
| 2 | 로컬 | 위 3개 파일을 `data/`에 **수동 복사** | — | `data/SFT_think_en_*.jsonl` |
| 3 | 로컬 | `python scripts/make_bundle.py` | 코드 + `data/`의 3개 파일 | `bundles/ugrp_bundle_{시각}.zip` |
| 4 | Pod | `bash scripts/run_train.sh --detach` | 번들 | `outputs/generation_think/final` (LoRA 어댑터) |
| 5 | Pod | `bash scripts/run_infer.sh --detach` | 어댑터 + `data/SFT_think_en_test.jsonl` | `results/generation_think_test.jsonl` |
| 6 | 로컬/Pod | `python -m ugrp.eval_report <결과.jsonl>` | 5번 출력 | 형식 지표 표 |

- **이 저장소는 데이터를 만들지 않는다.** 지문 정제, 사고 과정 합성·검수·병합, Gemma 4 학습 텍스트 변환은 전부 `UGRP2/data`에서 한다. 여기에는 완성된 파일만 넣는다.
- 지문이나 사고 과정을 고쳤으면 1 → 2 → 3을 다시 한다. 2번을 빠뜨리면 3번이 중단한다.
- 로컬은 GPU가 필요 없다. Pod에서는 학습과 추론만 한다. 4번과 5번은 서로 독립이라 다른 Pod에서 해도 된다.

## 1. 데이터 파일

`UGRP2/data`에서 만들어 `data/`에 **이름 그대로** 복사한다. JSONL, 한 줄에 한 건. id는 `YYYYMMNN`(시행 연월 + 지문 번호)이고 분할은 id(시간)순 8:1:1이다. 사고 과정은 영어판(`_en`)만 쓴다.

| 파일 | 만드는 명령 (`UGRP2/data`) | 형식 | 읽는 곳 |
| :--- | :--- | :--- | :--- |
| `SFT_think_en_train_gemma4.jsonl` | `03 convert` | `{id, target_length, prompt, completion}` | `train`, vLLM `--parity` |
| `SFT_think_en_valid.jsonl` | `03 merge` | `{id, target_length, messages[system, user, assistant{content}]}` | `check_boundary`, 추론 |
| `SFT_think_en_test.jsonl` | `03 merge` | 위와 같음 | 추론 |

```text
prompt     = <bos><|turn>system\n<|think|>\n{system}<turn|>\n<|turn>user\n{user}<turn|>\n<|turn>model\n   (chat template, enable_thinking=True)
completion = <|channel>thought\n{영어 사고 과정}<channel|>{지문}<turn|>\n                             (직접 조립)
```

- system = [prompts.py](src/ugrp/common/prompts.py)의 `GENERATION_SYSTEM`, user = `원본 문서:\n{input_reference}\n\n구조 데이터:\n{input_prompt}`.
- **목표 분량은 지문별 값**이다: 정답 지문의 공백 제외 글자 수를 100자 단위로 반올림한 값(`prompts.target_length_for`, 800~2100)이 `target_length`와 system 턴의 "N자 내외"에 들어간다. 2026-09-21~25의 1500 고정은 프롬프트와 정답이 어긋나 어댑터가 분량 지시를 무시하게 만들어 되돌렸다 ([issue/0925/0925_plan.md](issue/0925/0925_plan.md) 2절). 사고 과정에는 분량 숫자가 없어 재합성은 필요 없었다. 정답이 없는 원본 필드 입력만 `prompts.DEFAULT_TARGET_LENGTH`(1500)를 쓴다.
- valid/test에 사고 과정이 없는 것은 의도다. 모델이 직접 사고하고 쓴 지문을 `content`(정답 지문)와 비교한다.
- completion을 직접 조립하는 근거: Gemma 4 chat template은 assistant 턴의 thought 블록을 지워 버린다. template에 맡기면 사고 과정을 학습할 수 없다.
- `UGRP2/data`와 값이 같아야 하는 것: 집필 지시문, `DEFAULT_TARGET_LENGTH`·`TARGET_LENGTH_STEP`과 `target_length_for`의 반올림 규칙([prompts.py](src/ugrp/common/prompts.py)), completion 형식([gemma_format.py](src/ugrp/common/gemma_format.py)), `max_len`([configs/base.yaml](configs/base.yaml) ↔ 03번의 `MAX_LEN`), 사고 과정의 머리말·종결 문구([eval_report.py](src/ugrp/eval_report.py)). 앞의 둘은 `make_bundle`과 `tests/test_prompts.py`가 검사한다.

## 2. 로컬 명령

### 2.0 환경 (최초 1회)

```bash
python -m venv .venv && .venv/Scripts/activate      # Linux/macOS: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers trl peft accelerate datasets huggingface_hub pyyaml python-dotenv pytest
pip install -e .
cp .env.example .env                                # 선택. HF_TOKEN은 Hub 백업 때만 필수
```

### 2.1 데이터 받기

```powershell
# UGRP2/data 에서 (자세한 옵션은 ../data/README.md)
python 03_build_and_split_dataset.py merge
python 03_build_and_split_dataset.py convert        # 토큰 길이, MAX_LEN 초과 id를 여기서 확인
copy SFT_think_en_train_gemma4.jsonl,SFT_think_en_valid.jsonl,SFT_think_en_test.jsonl ..\agents\data\
```

### 2.2 `make_bundle` — 받은 데이터 검사 + Pod 업로드용 zip

```bash
python scripts/make_bundle.py
```

| 항목 | 내용 |
| :--- | :--- |
| 입력 | `data/`의 3개 파일 (1절). 여기서 만들지 않고 검사만 한다 |
| 검사 (토크나이저 불필요) | train: completion이 `<\|channel>thought\n`으로 시작하고 `<channel\|>`가 정확히 1회, `<turn\|>\n`으로 끝남 / `target_length`가 completion 속 지문 분량의 반올림값(`prompts.target_length_for`) / prompt에 `<\|think\|>`와 `prompts.generation_system(target_length)` 전문이 있고 `<\|turn>model\n`으로 끝남. valid/test: `target_length`가 정답 지문 분량의 반올림값, system 턴이 그 값으로 만든 것, user 턴 꼴, `reasoning` 없음 |
| 복사 누락 검사 | `../data`에 같은 이름의 파일이 있는데 내용이 다르면 중단 (그쪽이 새로 만든 것). `../data`가 없으면 건너뜀 |
| 포함 | `src/`, `configs/`, `scripts/`, `pyproject.toml`, `requirements*.txt`, 위 3개 파일, `MANIFEST.json`(건수, `max_len`, SHA256) |
| 제외 | `legacy/`, 어댑터, 테스트, `.env`, `data/`의 그 밖의 파일 |
| 출력 | `bundles/ugrp_bundle_{YYYYMMDD_HHMM}.zip` (수 MB). `.sh`는 LF로 바꿔 넣는다 |
| 중단 조건 | 파일 없음 / 위 검사 실패(해당 id를 5개까지 출력) / 복사 누락 / 토큰처럼 보이는 문자열(`hf_…` 등) 발견 |

## 3. Pod 준비

| 항목 | 값 | 근거 |
| :--- | :--- | :--- |
| GPU | 80GB 1장 (A100 80GB / H100) | 31B 4-bit + 최대 8.4k 토큰 역전파 |
| 템플릿 | RunPod PyTorch | `requirements.txt`에 torch가 없다 |
| Container Disk (`/root`) | 80GB (vLLM 쓰면 100GB) | 모델 가중치 ~62GB(`/root/model_cache`), vLLM venv ~10GB. Pod를 끄면 사라지지만 다시 받는 데 약 1분 |
| 드라이버 (vLLM만) | CUDA 13.0 이상 | `vllm==0.29.0` wheel이 CUDA 13 빌드(torch cu130). [setup_vllm.sh](scripts/setup_vllm.sh)가 설치 전에 확인하고 미달이면 중단한다 |
| Volume (`/workspace`) | 30GB 이상 | 쓰기가 5~8MB/s로 느리다. 코드·체크포인트·결과만 둔다 |
| 환경 변수 | `HF_TOKEN` (권장) | rate limit 완화, `PUSH_TO_HUB` 사용 시 필수 |

번들 업로드는 JupyterLab 업로드(`/workspace`), `scp -P <포트> bundles\ugrp_bundle_*.zip root@<IP>:/workspace/`, `runpodctl send/receive` 중 하나.

```bash
cd /workspace && unzip ugrp_bundle_*.zip && cd ugrp    # unzip이 없으면: python -m zipfile -e ugrp_bundle_*.zip .
head -20 MANIFEST.json                                  # 건수, max_len 확인
```

이후 명령은 모두 `/workspace/ugrp`에서 실행한다.

## 4. Pod 명령

### 4.1 `run_train.sh` — 학습

```bash
bash scripts/run_train.sh --detach                     # tmux 세션 ugrp-train. 접속이 끊겨도 계속 돈다
bash scripts/run_train.sh --detach --from train        # 해당 단계부터 다시
bash scripts/run_train.sh --detach --from train --resume   # 마지막 체크포인트에서 이어서
PUSH_TO_HUB=<계정>/<repo> bash scripts/run_train.sh --detach   # 끝나면 어댑터를 HF private repo로 백업
```

| 단계 | 실행 | 입력 → 출력 | 시간 | 실패하면 |
| :--- | :--- | :--- | :-: | :--- |
| `setup` | [pod_setup.sh](scripts/pod_setup.sh) | VRAM·디스크(`/workspace` 30GB, 캐시 65GB)·bf16 점검 → 패키지 설치, 모델 다운로드 | 수 분 | 메시지대로 디스크 증설 또는 `HF_TOKEN` 설정 |
| `boundary` | `ugrp.training.check_boundary` | `SFT_think_en_valid.jsonl` 첫 건을 **미세조정 전** 모델로 생성 → 출력이 `<\|channel>thought\n`으로 시작하고 `<channel\|>` 직후 공백 없이 `<turn\|>`로 끝나는지 대조 | 5~10분 | `[MISMATCH]` → 7절 |
| `smoke` | `train --max-steps 3 --limit 4 --longest` | 최장 4건 3 step → `peak_vram_gb`, `max_memory_reserved_gb` 출력 | 5~10분 | 본 학습 전에 멈춘다 |
| `train` | `ugrp.training.train` | `SFT_think_en_train_gemma4.jsonl` → `outputs/generation_think/{checkpoint-*, final}` | 약 4시간 | `--from train --resume` |
| `reference` | `ugrp.inference.parity_reference` | 학습 파일 앞 2건의 정답을 HF 4-bit 베이스와 `final`에 넣어 평균 NLL을 잰다 → `outputs/generation_think/parity_reference.json`. vLLM `--parity`의 기준값. 같은 forward에서 **형식 학습 게이지**(사고·지문 구간 NLL, 사고 첫 토큰 `Task`의 확률)를 찍는다: 확률 0.5 미만이면 사고 형식을 못 배운 것(2차 학습 0.025). 체크포인트별로 보려면 `python -m ugrp.inference.parity_reference --adapter outputs/generation_think/checkpoint-37` | 수 분 | 실패해도 멈추지 않는다. 나중에 직접 실행 |
| `pack` | [pack_results.sh](scripts/pack_results.sh) | `final`, `checkpoint-*`의 어댑터 가중치(`PACK_CHECKPOINTS=0`이면 제외), `trainer_state.json`, `metrics.json`, `parity_reference.json`, `results/`, `logs/` → `results_{번들ID}.tar.gz` (약 0.8GB) | 수 분 | `--from pack` |

학습의 근거 ([configs/base.yaml](configs/base.yaml)):

| 항목 | 값 |
| :--- | :--- |
| 베이스 | 4-bit NF4, double quant, compute bf16. 어댑터는 merge하지 않는다 (4-bit merge는 품질 손실) |
| LoRA | r 16, alpha 32, dropout 0.05, 언어 모델의 q/k/v/o/gate/up/down (vision tower 제외) |
| 최적화 | batch 1 × 누적 **4**, lr **2e-4** cosine, warmup 5%(10 step), grad clip 1.0, `paged_adamw_8bit`, 5 epoch = **185 step**(epoch당 37). 2차 학습(누적 16, 30 step)이 사고 형식을 못 배워 업데이트 횟수를 6배로 늘렸다 ([issue/0927/0927_settings_lora.md](issue/0927/0927_settings_lora.md)) |
| loss | prompt와 completion을 따로 토큰화해 `completion_mask`로 넘긴다 (`completion_only_loss=True`). 사고 과정 + 지문에만 loss |
| 길이 | `max_len` 10240 초과 샘플은 제외 (자르지 않음) |
| 저장 | epoch마다 체크포인트 5개(`checkpoint-37/74/111/148/185`) 모두 보존. `final/`에 어댑터, 토크나이저, `generation_config.json`(eos `[1, 106, 50]`) |
| `eval_loss` | 없음. valid/test에 사고 과정 정답이 없기 때문. 과적합은 체크포인트별 valid 생성으로 보거나 `train_dev_holdout: N`(train 최근 N건을 빼고 epoch마다 측정) |

진행 확인:

```bash
tail -f logs/ugrp-train.log     # Ctrl+C로 빠져나와도 학습은 계속된다
cat logs/STATE                  # 끝난 단계와 소요 시간
tmux attach -t ugrp-train       # 빠져나올 때는 Ctrl+B 다음 D. 여기서 Ctrl+C를 누르면 학습이 죽는다
```

### 4.2 `run_infer.sh` — 추론

```bash
bash scripts/run_infer.sh --detach                     # final 어댑터, test 19건, HF 백엔드
```

| 단계 | 동작 |
| :--- | :--- |
| `setup` | 같은 Pod에서 이미 했으면 건너뛴다. 새 Pod면 모델을 다시 받는다 |
| `vllm_setup` | `--backend vllm`일 때만 [setup_vllm.sh](scripts/setup_vllm.sh)로 `/root/vllm-env` 설치 |
| `adapter` | 어댑터가 있는지 모델을 올리기 전에 확인. 로컬에 없고 경로 꼴이 아니면 HF Hub repo로 본다 |
| `infer` | 배치마다 `results/*.jsonl`에 기록. **다시 실행하면 이미 있는 id는 건너뛴다** |
| `report` | 분할 추론이면 `eval_report`를 돌려 `results/eval_report_*.json` 저장 (`--input/--output`, `--parity`는 제외) |
| `pack` | `results_{번들ID}.tar.gz` 갱신 |

| 항목 | 내용 |
| :--- | :--- |
| 입력 | 기본 `data/SFT_think_en_{split}.jsonl`. 파일의 system/user 턴을 그대로 쓴다 (학습과 같은 `build_prompt`) |
| 직접 만든 입력 | `{"id", "input_reference", "input_prompt"[, "output_passage"]}`도 받는다. 프롬프트는 `prompts.py`로 만들고 목표 분량은 `DEFAULT_TARGET_LENGTH`(1500)다. `--target-length`로 바꾼다 |
| 생성 설정 | `configs/base.yaml`의 `generate`: temperature 1.0, top_p 0.95, top_k 64, `max_new_tokens` 6144 |
| 멈춤 토큰 | `[<eos>(1), <turn\|>(106)]`. 토크나이저의 eos는 `<eos>`뿐이라 직접 넘긴다. `parse_response`는 첫 `<turn\|>` 뒤를 버린다 |
| 출력 파일 | hf `results/generation_think[_base]_{split}.jsonl` / vllm `…_vllm.jsonl` / 다중 어댑터 `…_vllm_{이름}.jsonl` |
| 출력 행 | `{id, target_length, thinking, answer, reference, raw, backend, tokens, finish_reason}`. vllm은 `{adapter, sampling}` 추가 |

| 백엔드 | 방식 | 언제 |
| :--- | :--- | :--- |
| `hf` (기본) | 4-bit 베이스 + PEFT LoRA, 배치 20건(메모리 부족 시 자동으로 절반) | 기본 경로 |
| `vllm` | bf16 원본 + 실행 중 LoRA, 별도 venv. 엔진 기동 5~10분 | 어댑터 여러 개를 한 번에 비교. 검증 계획: [issue/0919/0919_inf.md](issue/0919/0919_inf.md), 환경 수정: [issue/0925/0925.md](issue/0925/0925.md) 부록 A |

자주 쓰는 형태:

```bash
bash scripts/run_infer.sh --detach --limit 3                          # 앞의 3건만
bash scripts/run_infer.sh --detach --split valid                      # valid 18건 (과적합 확인)
bash scripts/run_infer.sh --detach --no-adapter                       # 미세조정 전 베이스 (비교용)
bash scripts/run_infer.sh --detach --adapter outputs/generation_think/checkpoint-37 --output results/think_ep1_test.jsonl   # epoch 1 체크포인트
bash scripts/run_infer.sh --detach --input data/my.jsonl --output results/my.jsonl --target-length 1800

# 새 Pod에서: 번들과 results_*.tar.gz를 올린 뒤
tar -xzf ../results_*.tar.gz && bash scripts/run_infer.sh --detach     # outputs/generation_think/final 복원 후 추론

# vLLM: parity → smoke → 본 추론 순 (KV cache 부족으로 엔진이 뜨지 않으면 --max-num-seqs 4 --max-num-batched-tokens 2048)
bash scripts/run_infer.sh --backend vllm --parity                     # 어댑터가 bf16에서도 붙는지: 학습 2건의 loss를 HF 4-bit 기준값과 비교
bash scripts/run_infer.sh --backend vllm --limit 1 --greedy           # 엔진이 뜨고 사고·지문이 <channel|>·<turn|>로 닫히는지 확인
bash scripts/run_infer.sh --detach --backend vllm --max-num-seqs 4 --max-num-batched-tokens 2048 \
    --adapters final=outputs/generation_think/final ep1=outputs/generation_think/checkpoint-37 ep2=outputs/generation_think/checkpoint-74
bash scripts/run_infer.sh --detach --backend vllm --max-num-seqs 4 --max-num-batched-tokens 2048 --no-adapter   # 같은 조건의 베이스 (비교용)

# 구조 에이전트 → 생성 에이전트 연쇄
export PYTHONPATH=$PWD/src && python -m ugrp.inference.run_pipeline --split test
```

| 옵션 | 기본값 | 설명 |
| :--- | :-: | :--- |
| `--adapter <경로\|HF repo>` | `outputs/{agent}_think/final` | 쓸 어댑터 |
| `--no-adapter` | — | 베이스 모델만. 출력 파일명에 `_base` |
| `--split` | `test` | `data/SFT_think_en_{split}.jsonl` (`valid` \| `test`) |
| `--input` / `--output` | 자동 | 입출력 경로 직접 지정 (report 단계는 건너뜀) |
| `--target-length N` | 파일의 값 (평가 파일은 정답 분량의 100자 단위 반올림값, 원본 필드 입력은 1500) | 모든 샘플의 목표 분량을 N자로 바꿔 프롬프트를 다시 만든다 (예: 1500 고정 조건으로 비교할 때) |
| `--limit N` | 전체 | 앞에서 N건 |
| `--max-new-tokens` | 6144 | 답이 잘리면 올린다 |
| `--backend` | `hf` | `hf` \| `vllm` |
| `--batch-size N` | 20 | [hf] 한 번에 생성할 건수 |
| `--adapters 이름=경로 …` | — | [vllm] 모델을 한 번 올려 여러 어댑터를 차례로 |
| `--greedy`, `--temperature T`, `--seed`, `--repetition-penalty` | configs 값 / 42 / 1.0 | [vllm] 샘플링. 결과의 `sampling`에 기록 |
| `--gpu-mem`, `--max-model-len`, `--max-num-seqs`, `--max-num-batched-tokens` | 0.92 / 12288 / — / — | [vllm] 엔진 메모리. 엔진 초기화가 KV cache 부족으로 실패하면 `--max-num-seqs 4 --max-num-batched-tokens 2048` (활성화 메모리와 CUDA graph 메모리가 줄어 KV에 약 2GB가 더 돌아간다). `--max-model-len`은 줄이지 않는다 (test 프롬프트 최대 5,947 + 생성 6,144) |
| `--structure-from <결과.jsonl>` | — | 구조 에이전트 추론 결과의 `answer`를 구조 데이터로 쓴다 |
| `--parity` | — | [vllm] 생성하지 않고 학습 2건의 loss만 잰다. 판정: base→어댑터 감소폭이 0.3 이상(어댑터가 붙었는지)이고, `outputs/{agent}_think/parity_reference.json`(학습의 `reference` 단계가 만든 HF 4-bit 값)이 있고 같은 데이터·어댑터 이름이면 어댑터 NLL 절대값이 HF 값과 ±0.05 안이어야 OK. 감소폭 비교는 4-bit와 bf16 base의 차이(약 0.05)가 그대로 들어가 허위 FAIL을 내서 2026-09-25에 뺐다. 형식 게이지(사고·지문 NLL, 첫 토큰 확률)도 함께 찍는다 |

## 5. 결과 확인

```text
/workspace/ugrp/
├── results_{번들ID}.tar.gz          내려받을 것은 이것 하나
├── outputs/generation_think/        final/ (어댑터, 토크나이저, generation_config.json), checkpoint-37/74/111 (어댑터만 묶음에 포함), trainer_state.json, metrics.json, parity_reference.json
├── results/                         generation_think_test.jsonl, …_vllm.jsonl, eval_report_generation_test.json
└── logs/                            ugrp-train.log, ugrp-infer.log, STATE, pip_freeze.txt
```

내려받기: JupyterLab에서 우클릭 Download, `scp -P <포트> root@<IP>:/workspace/ugrp/results_*.tar.gz .`, 또는 `runpodctl send`. **Pod는 자동으로 꺼지지 않는다. 압축이 풀리는지 확인한 뒤 콘솔에서 Terminate한다.**

```bash
tar -xzf results_YYYYMMDD_HHMM.tar.gz                 # agents/에서. outputs/, results/, logs/가 복원된다
python -c "import json; print(json.load(open('outputs/generation_think/metrics.json')))"
python -m ugrp.eval_report results/generation_think_test.jsonl [비교할_결과.jsonl ...]
```

### `eval_report` — 형식 지표 (모델 불필요)

입력은 결과 JSONL의 `thinking`, `answer`, `reference`, `target_length`와, 복사율을 위한 평가 입력 파일(결과 파일 이름의 split로 `data/SFT_think_en_{split}.jsonl`을 자동으로 찾고, `--inputs`로 지정할 수도 있다)이다. think 학습이 의도대로 됐는지 보는 형식 점검이며 지문 품질 평가가 아니다. 다만 복사율은 2차 학습에서 정성 평가의 실패 사례(원문 전재, 구조 데이터 나열)를 그대로 걸러냈다 ([issue/0925/0925_plan.md](issue/0925/0925_plan.md) 1.4절).

| 지표 | 계산 | 기대 |
| :--- | :--- | :-: |
| `empty_answer` | 답이 빈 건수 | 0 |
| `tag_leak_in_answer` | 답에 `<\|channel>` 등 제어 태그가 남은 건수 | 0 |
| `thinking_present` | 사고 블록이 파싱된 비율 | 1.0 |
| `thinking_closing_line` | 사고가 `Plan complete. Writing the passage in Korean.`으로 끝난 비율 | 1.0 |
| `thinking_all_headers` | `Task:` `Source Text overview:` `Structure Data check:` `Paragraph plan:` `Final check:`가 모두 있는 비율 | 1.0 |
| `thinking_korean_ratio` | 사고의 한글 비율. 학습 데이터는 영어 사고에 개념어만 한글로 병기한다 (합성 기준 15% 이하) | 0.15 이하 |
| `thinking_chars_mean`, `answer_chars_mean` | 평균 글자 수 | 참고 |
| `length_error_mean` | \|답 글자 수 − 제시한 목표 분량\| / 목표 분량. 평가 파일의 목표는 정답 분량의 반올림값이라 지시 준수를 보여 준다 (베이스 14.6%) | 낮을수록 |
| `length_error_vs_reference_mean` | \|답 글자 수 − 정답 글자 수\| / 정답 글자 수 (공백 제외) | 낮을수록 |
| `paragraphs_match_plan` | 답의 문단 수 = 사고의 `Paragraph N (~M chars)` 개수 | 1.0 |
| `paragraphs_match_reference` | 답의 문단 수 = 정답 지문의 문단 수 | 높을수록 |
| `source_copy_ratio` | 답(공백 제거)이 원본 문서와 30자 이상 연속 일치하는 구간으로 덮이는 비율의 평균. 정답 지문 0.00, 베이스 최대 0.17, 원문을 전재한 출력 0.3~0.9 | 0에 가깝게 |
| `structure_copy_ratio` | 구조 데이터의 `사용된 정보` 명제와 12자 이상 연속 일치하는 비율의 평균. 정답 지문 평균 0.09(최대 0.31) | 0.1 안팎 |
| `copy_flagged` | 원문 ≥ 0.20 또는 구조 ≥ 0.35인 건수 (해당 id는 표 아래 `[COPY]` 줄에). 2차 학습 어댑터 9건, 베이스 0건 | 0 |

## 6. 코드 위치

| 경로 | 역할 | 실행 환경 |
| :--- | :--- | :--- |
| [configs/base.yaml](configs/base.yaml), `generation.yaml`, `structure.yaml` | 모델, `max_len`, 양자화, LoRA, 학습·생성 설정의 단일 출처 | — |
| [common/prompts.py](src/ugrp/common/prompts.py) | 두 에이전트의 system/user 프롬프트, `DEFAULT_TARGET_LENGTH` | 공통 (torch 없음) |
| [common/gemma_format.py](src/ugrp/common/gemma_format.py) | prompt/completion 조립, `parse_response`, 멈춤 토큰 | 공통 |
| [common/data.py](src/ugrp/common/data.py), [common/config.py](src/ugrp/common/config.py) | JSONL 입출력, 파일 이름 규칙, 설정·경로 (`UGRP_ROOT`, `UGRP_DATA_DIR`) | 공통 |
| [training/train.py](src/ugrp/training/train.py), [training/check_boundary.py](src/ugrp/training/check_boundary.py) | 학습, 경계 점검. `train --tiny-debug`는 CPU에서 초소형 모델로 코드 경로만 확인 | Pod |
| [inference/infer.py](src/ugrp/inference/infer.py), [infer_vllm.py](src/ugrp/inference/infer_vllm.py), [parity_reference.py](src/ugrp/inference/parity_reference.py), [inputs.py](src/ugrp/inference/inputs.py), [run_pipeline.py](src/ugrp/inference/run_pipeline.py) | HF 추론, vLLM 추론, vLLM `--parity`용 HF 기준값 측정, 공통 입력·결과 형식, 두 에이전트 연쇄 | Pod |
| [eval_report.py](src/ugrp/eval_report.py) | 5절 | 로컬/Pod |
| [scripts/](scripts/) | `run_train.sh`, `run_infer.sh`, `pod_setup.sh`, `setup_vllm.sh`, `pack_results.sh`, `_common.sh`(tmux, `logs/STATE`, 단계 재개), `make_bundle.py` | — |

- 전처리 코드는 여기에 없다. 예전의 `ugrp.preprocess`(`sync_passages`, `build_sft`)는 [legacy/preprocess_20260921/](legacy/preprocess_20260921/)에 보관만 하고, 하던 일은 `UGRP2/data`의 03번 `merge`·`convert`가 맡는다.
- `--agent structure`(원본 문헌 → 구조도)도 같은 명령 체계를 쓴다. `data/SFT_structure_think_en_{valid,test}.jsonl`과 `…_train_gemma4.jsonl`이 준비되면 된다.
- 경과 기록: 1차 학습 점검 [issue/0919/0919.md](issue/0919/0919.md), 추론 속도·vLLM 계획 [issue/0919/0919_inf.md](issue/0919/0919_inf.md), 2차 학습 결과·vLLM 환경 수정 [issue/0925/0925.md](issue/0925/0925.md), 그에 따른 수정 계획(지문별 목표 분량, 누적 4, 게이지·복사율) [issue/0925/0925_plan.md](issue/0925/0925_plan.md).

## 7. 문제 해결

| 증상 | 조치 |
| :--- | :--- |
| `[ABORT] tmux 세션 'ugrp-train'이 이미 있습니다` | `tmux attach -t ugrp-train`으로 확인. 끝난 세션이면 `tmux kill-session -t ugrp-train` |
| 접속이 끊겨 상태를 모름 | `cat logs/STATE`, `tail -50 logs/ugrp-train.log`, `tmux ls` |
| 모듈을 직접 실행하고 싶음 | `cd /workspace/ugrp && export PYTHONPATH=$PWD/src` 후 `python -m ugrp.training.train …` |
| `$'\r': command not found` | `sed -i 's/\r$//' scripts/*.sh` |
| 로컬 `No module named 'ugrp'` | `pip install -e .` 또는 PowerShell `$env:PYTHONPATH = "src"` |
| `setup`: 디스크 여유 부족 | `/workspace` 30GB 미만 → Volume 증설. `/root/model_cache` 65GB 미만 → Container Disk 80GB 이상 |
| 모델 다운로드가 느림 / HTTP 429 | `export HF_TOKEN=hf_…` 후 `FORCE_SETUP=1 bash scripts/run_train.sh --detach` |
| `boundary`: `[MISMATCH]` | 로그의 `[RAW …]`와 `gemma_format.py`의 `THINK_OPEN`/`COMPLETION_SUFFIX`를 대조해 고친다. `UGRP2/data`의 03번 `convert`도 같은 형식으로 고쳐 다시 실행하고, 복사 → `make_bundle`. 사고가 안 끝났으면 `--max-new-tokens 8192`로 재확인 |
| 학습 중 CUDA OOM | `max_len`을 실측 최대 바로 위로 낮추거나 LoRA `target_modules`를 `(q_proj\|v_proj)`로 축소 |
| `adapter`: 어댑터가 없습니다 | 먼저 학습하거나 `tar -xzf results_*.tar.gz`로 `outputs/generation_think/final` 복원 |
| [hf] `CUDA 메모리 부족 → 배치 크기를 N으로` | 정상. 계속 뜨면 `--batch-size 8` |
| `finish_reason`이 `length` | `가 가 가…` 반복 붕괴면 `--greedy` 또는 `--temperature 0.7`. 정상 사고가 긴 것이면 `--max-new-tokens 8192` |
| 다시 실행했는데 아무것도 생성되지 않음 | 결과 파일에 같은 id가 이미 있다. 파일을 지우거나 `--output`을 바꾼다 |
| [vllm] `--parity`가 "기준값을 쓰지 않습니다" | `parity_reference.json`이 없거나 다른 학습 데이터로 잰 것. 절대 기준(감소폭 0.3)으로만 판정한다. 정밀 비교가 필요하면 학습 환경에서 `python -m ugrp.inference.parity_reference` |
| [vllm] `--parity` FAIL | 감소폭이 0.3 미만이면 regex `target_modules`가 적용되지 않은 것 → [issue/0919/0919_inf.md](issue/0919/0919_inf.md) 대안 1. 감소폭은 충분한데 어댑터 NLL이 HF 값과 0.05 넘게 다르면 대안 2(4-bit 서빙)로 같은 검증 |
| [vllm] `import vllm` 시 `libcudart.so.13` ImportError | torch가 cu126 등 다른 CUDA 빌드로 설치된 것. 수정된 `setup_vllm.sh`(cu130 인덱스 명시)로 `FORCE_SETUP=1 bash scripts/setup_vllm.sh` (옛 venv면 torch 계열만 다시 깐다). 드라이버가 CUDA 13.0 미만인 Pod는 쓸 수 없다 ([issue/0925/0925.md](issue/0925/0925.md) A-1) |
| [vllm] `FileNotFoundError: 'ninja'` (EngineCore 초기화) | FlashInfer가 샘플링 커널을 JIT 빌드할 때 venv의 ninja를 PATH에서 찾는다. `run_infer.sh`가 PATH를 넣어 주므로, 직접 실행할 때는 `PATH=/root/vllm-env/bin:$PATH` (A-2) |
| [vllm] `KV cache is needed, which is larger than the available` (엔진 초기화 실패) | `--max-num-seqs 4 --max-num-batched-tokens 2048`. `--max-model-len`을 줄이면 긴 test 프롬프트가 잘린다 (A-3). 그래도 부족하면 `--gpu-mem 0.95` → 0919/0919_inf.md 대안 2(4-bit 서빙) |
| [vllm] `Maximum concurrency` 4 미만 | 그 자체로는 문제가 아니다. 2.4x에서도 19건이 11분에 끝났다. 엔진이 뜨고 tok/s가 나오면 그대로 쓴다 |
| [vllm] CUDA 드라이버 충돌 | 최신 CUDA 템플릿 Pod를 쓰거나 `requirements-vllm.txt` 버전을 낮춘다 (그때 `setup_vllm.sh`의 `TORCH_INDEX`·`REQUIRED_CUDA`도 맞춘다) |
