# 데이터 명세 및 프롬프트 규격

이 문서는 UGRP2 에이전트 학습·추론에 사용되는 데이터 파일 규격, 프롬프트 조립 규칙 및 데이터 검증 절차를 다룹니다.

---

## 1. 개요 및 데이터 흐름

- **데이터 생성 분리**: 이 저장소(`SFT-for-agents`)는 학습용 데이터를 직접 생성하지 않습니다. 지문 정제, 사고 과정 합성·검수·병합, Gemma 4 학습 텍스트 변환은 모두 `UGRP2/data`에서 수행합니다 ([../data/README.md](../data/README.md) 참고).
- **수동 복사**: `UGRP2/data`에서 완성된 파일 3개를 이 저장소의 `data/` 폴더에 **이름 그대로** 복사하여 사용합니다.

```
[UGRP2/data]
  03_build_and_split_dataset.py merge   -->  SFT_think_en_{valid,test}.jsonl
  03_build_and_split_dataset.py convert -->  SFT_think_en_train_gemma4.jsonl
        │
        ▼ (수동 복사)
[SFT-for-agents/data/]
        │
        ▼
  python scripts/make_bundle.py (형식 검증 및 무결성 확인)
```

---

## 2. 데이터 파일 명세

데이터는 JSON Lines (JSONL) 형식이며 한 줄에 한 건의 샘플을 담고 있습니다.  
- **ID 체계**: `YYYYMMNN` (시행 연월 + 지문 번호)  
- **데이터 분할**: ID(시간순) 기준 `8 : 1 : 1` (Train 148건 / Valid 18건 / Test 19건)  
- **사고 과정 언어**: 영어판(`_en`)을 사용합니다. (한국어 단어/개념어 일부 병기)

| 파일명 | 생성 명령 (`UGRP2/data`) | 형식 (JSON 키) | 읽는 곳 |
| :--- | :--- | :--- | :--- |
| `SFT_think_en_train_gemma4.jsonl` | `03 convert` | `{id, target_length, prompt, completion}` | `train`, vLLM `--parity` |
| `SFT_think_en_valid.jsonl` | `03 merge` | `{id, target_length, messages[system, user, assistant]}` | `check_boundary`, 추론 (valid) |
| `SFT_think_en_test.jsonl` | `03 merge` | 위와 동일 | 추론 (test) |

> [!NOTE]
> `valid`와 `test` 데이터셋의 `assistant` 턴에는 사고 과정(thought)이 없고 정답 지문(`content`)만 들어있습니다.  
> 모델이 입력을 보고 스스로 사고하여 생성한 결과를 원본 정답 지문과 비교하기 위함입니다.

---

## 3. 프롬프트 및 완성문 구조

Gemma 4 31B-it 모델의 think mode 규격에 맞춰 구성됩니다.

```text
prompt     = <bos><|turn>system\n<|think|>\n{system}<turn|>\n<|turn>user\n{user}<turn|>\n<|turn>model\n   (chat template, enable_thinking=True)
completion = <|channel>thought\n{영어 사고 과정}<channel|>{지문}<turn|>\n                             (직접 조립)
```

### 3.1 턴별 구성 요소
- **System 턴**: [`src/ugrp/common/prompts.py`](../src/ugrp/common/prompts.py)의 `GENERATION_SYSTEM` 템플릿 사용. 목표 분량("공백 제외 N자 내외") 지시문 포함.
- **User 턴**:
  ```text
  원본 문서:
  {input_reference}

  구조 데이터:
  {input_prompt}
  ```
- **Model 턴 (Completion)**:
  - `<|channel>thought\n` 태그로 사고 블록을 열고 영어 사고 과정을 작성
  - `<channel|>` 태그로 사고 블록을 닫고 곧바로 지문(한국어) 작성
  - `<turn|>\n` 태그로 턴 종료

### 3.2 Completion을 직접 조립하는 근거
Hugging Face의 Gemma 4 공식 chat template은 assistant 턴에 thought 블록이 들어있으면 템플릿 렌더링 과정에서 이를 제거(strip)합니다. 따라서 템플릿에 전적으로 의존하면 사고 과정을 모델에 지도학습(SFT)시킬 수 없으므로, 완성문을 직접 조립하여 손실을 계산하도록 설계되었습니다 ([`src/ugrp/common/gemma_format.py`](../src/ugrp/common/gemma_format.py)).

---

## 4. 목표 분량 규칙

- **지문별 동적 분량**: 목표 분량은 고정값이 아니며, 정답 지문의 공백 제외 글자 수를 100자 단위로 반올림한 값(`prompts.target_length_for`, 800~2100자)입니다.
- **프롬프트 일치**: `target_length` 수치는 system 턴 프롬프트의 "N자 내외" 지시문에 정확히 반영됩니다.
- *(참고)* 과거 1500자 일괄 고정 방식은 프롬프트 지시와 실제 정답 지문의 분량이 불일치하여 모델이 분량 지시를 무시하는 부작용이 발생했으므로, 현재의 지문별 반올림 방식으로 정상화되었습니다 ([issue/0925/0925_plan.md](../issue/0925/0925_plan.md)).
- 정답 지문이 없는 신규 원본 필드 추론 시에만 `prompts.DEFAULT_TARGET_LENGTH`(1500자)가 적용됩니다.

---

## 5. 저장소 간 일관성 유지 규격

`UGRP2/data`와 `SFT-for-agents` 간에 반드시 동일하게 유지되어야 하는 규격 목록입니다:
1. **집필 지시문**: `GENERATION_SYSTEM`
2. **분량 계산 상수**: `DEFAULT_TARGET_LENGTH`, `TARGET_LENGTH_STEP` 및 `target_length_for` 반올림 규칙
3. **Completion 특수 태그**: `THINK_OPEN`, `THINK_CLOSE`, `TURN_CLOSE` 등
4. **최대 토큰 길이**: `configs/base.yaml`의 `max_len: 10240` (03번 convert의 `MAX_LEN`)
5. **사고 과정 머리말/종결문**: `Task:`, `Plan complete. Writing the passage in Korean.`

위 항목 중 앞의 두 항목은 `scripts/make_bundle.py`와 `tests/test_prompts.py`를 통해 자동으로 검증됩니다.

---

## 6. 번들 검사 로직 (`make_bundle.py`)

`python scripts/make_bundle.py` 실행 시 다음 항목을 자동으로 검사하며, 불일치 시 `[ABORT]`로 중단합니다:

1. **Train 데이터 검사**:
   - `completion`이 `<|channel>thought\n`으로 시작하고 `<channel|>`가 정확히 1회 존재하며 `<turn|>\n`으로 끝나는지 여부
   - `target_length`가 지문 분량의 100자 반올림값과 일치하는지 여부
   - `prompt`에 `<|think|>` 및 해당 분량의 `generation_system` 지시문 전문이 포함되어 있는지 여부
2. **Valid / Test 데이터 검사**:
   - `target_length`가 정답 분량과 일치하는지 여부
   - system/user 턴 형식이 규격대로 들어있는지 여부
   - completion/reasoning이 없는지 여부
3. **복사 누락 검사**:
   - `../data` 디렉토리에 동일 이름의 최신 파일이 있는데 `data/`와 내용이 다를 경우 경고 및 중단
4. **보안 검사**:
   - Hugging Face 토큰(`hf_...`) 등 민감 정보가 포함되어 있는지 검사

