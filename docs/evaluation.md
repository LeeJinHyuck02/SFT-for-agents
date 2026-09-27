# 형식 평가지표(eval_report) 가이드

이 문서는 생성된 추론 결과물(`results/*.jsonl`)을 바탕으로 모델의 사고 구조, 분량 지시 준수율 및 복사율을 평가하는 지표 명세를 다룹니다.

---

## 1. 결과 파일 확인 및 평가 실행

추론이 완료되면 결과물은 RunPod의 `/workspace/ugrp/results_{번들ID}.tar.gz`에 패킹됩니다.  
로컬 머신 또는 Pod에서 압축을 풀고 평가 스크립트를 실행할 수 있습니다 (모델 불필요, CPU 환경 지원).

```bash
# 1. 결과 압축 해제 (outputs/, results/, logs/ 복원)
tar -xzf results_YYYYMMDD_HHMM.tar.gz

# 2. 단일 결과 리포트 출력
python -m ugrp.eval_report results/generation_think_test_vllm_final.jsonl

# 3. 여러 어댑터 결과 및 베이스 모델 결과를 한 표로 비교
python -m ugrp.eval_report \
    results/generation_think_base_test_vllm.jsonl \
    results/generation_think_test_vllm_ep1.jsonl \
    results/generation_think_test_vllm_ep2.jsonl \
    results/generation_think_test_vllm_final.jsonl
```

---

## 2. 평가 지표 상세 명세

`eval_report`는 모델의 문학적 품질이 아닌 **사고 및 작성 형식(Format Compliance)**을 정량 측정합니다.

| 지표명 | 계산 방식 | 기대값 | 설명 및 판정 기준 |
| :--- | :--- | :-: | :--- |
| `empty_answer` | 생성된 본문이 빈 문자열인 건수 | `0` | 답이 누락되면 모델 실패 |
| `tag_leak_in_answer` | 본문에 `<\|channel>`, `<turn\|>` 등 제어 태그가 남아있는 건수 | `0` | 파싱 실패 또는 미닫힘 여부 |
| `thinking_present` | 사고 과정(thought) 블록이 정상 파싱된 비율 | `1.0 (100%)` | think mode 작동 여부 |
| `thinking_closing_line` | 사고 종료 문구(`Plan complete. Writing the passage in Korean.`) 일치율 | `1.0 (100%)` | 사고 완료 규격 준수 여부 |
| `thinking_all_headers` | 필수 5개 헤더가 모두 존재하는 비율 | `1.0 (100%)` | `Task:`, `Source Text overview:`, `Structure Data check:`, `Paragraph plan:`, `Final check:` 포함 여부 |
| `thinking_korean_ratio` | 사고 과정 텍스트 중 한글 문자 비율 | `≤ 0.15 (15% 이하)` | 영어 사고 원칙 준수 여부 (개념어만 한글 병기 허용) |
| `thinking_chars_mean` | 사고 과정 평균 글자 수 | 참고값 | 통상 2,000 ~ 4,000자 내외 |
| `answer_chars_mean` | 완성 지문 평균 글자 수 (공백 제외) | 참고값 | 목표 분량 대비 생성 길이 |
| `length_error_mean` | $\frac{\| \text{생성 글자 수} - \text{목표 분량} \|}{\text{목표 분량}}$ | **낮을수록 우수** | 프롬프트의 분량 지시 준수율 (베이스 모델 약 14.6%) |
| `length_error_vs_ref_mean`| $\frac{\| \text{생성 글자 수} - \text{정답 지문 글자 수} \|}{\text{정답 지문 글자 수}}$ | **낮을수록 우수** | 실제 수능 지문 분량과의 오차율 |
| `paragraphs_match_plan` | 생성 문단 수 = 사고 과정의 `Paragraph N` 계획 수 일치율 | `1.0 (100%)` | 자신이 세운 문단 계획을 그대로 따랐는지 확인 |
| `paragraphs_match_ref` | 생성 문단 수 = 원본 정답 지문의 문단 수 일치율 | 높을수록 우수 | 수능 지문 구조 재현도 |
| `source_copy_ratio` | 원문과 30자 이상 연속 일치하는 구간의 비율 | **`0.00`에 근접** | 원문 단순 전재(베끼기) 여부 |
| `structure_copy_ratio` | 구조 데이터의 명제와 12자 이상 연속 일치하는 비율 | **`0.10` 안팎** | 구조 데이터의 정보 활용 적정선 (정답 지문 평균 약 0.09) |
| `copy_flagged` | 원문 복사율 ≥ 0.20 또는 구조 복사율 ≥ 0.35인 이상 건수 | **`0`** | 과도한 복사가 일어난 샘플 수 (해당 ID 출력) |

---

## 3. 복사율(Copy Ratio) 평가의 중요성

2차 학습 당시 정성 평가에서 실패한 대표적인 사례는 다음과 같았습니다:
- 원본 문헌의 장문을 그대로 복사하여 붙여넣음 (`source_copy_ratio` 0.3~0.9까지 급증)
- 구조 데이터의 문장을 나열식으로 전재함 (`structure_copy_ratio` 과다)

실제 수능 국어 비문학 정답 지문은:
- 원문 문헌과의 30자 연속 일치율(`source_copy_ratio`)이 **0.00**입니다. (전문 용어를 제외하고는 표현을 완전히 재구성함)
- 구조 데이터와의 일치율(`structure_copy_ratio`)은 평균 **0.09** (최대 0.31) 수준입니다.

따라서 `copy_flagged`가 0건인지, `source_copy_ratio`가 베이스 모델 수준(0.0~0.1 미만)으로 억제되는지가 모델의 창작 및 지문 재구성 능력을 검증하는 핵심 척도입니다 ([issue/0925/0925_plan.md](../issue/0925/0925_plan.md) 1.4절).

