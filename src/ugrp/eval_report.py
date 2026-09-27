"""추론 결과의 형식 지표를 집계한다 (모델 불필요, 로컬 실행 가능).

    python -m ugrp.eval_report results/generation_think_test.jsonl
    python -m ugrp.eval_report results/generation_think_test_vllm.jsonl results/generation_think_base_test_vllm.jsonl
    python -m ugrp.eval_report results/my.jsonl --inputs data/my.jsonl        # 복사율의 입력 파일을 직접 지정

결과 파일의 각 행에 들어 있는 target_length(제시한 목표 분량)와 reference(정답 지문)로 형식·분량 지표를 계산한다.
복사율(원문 전재, 구조 데이터 나열)에는 입력 파일의 원본 문서·구조 데이터가 필요하다: 결과 파일 이름의 split(_valid/_test)로
data/SFT_think_en_{split}.jsonl을 찾고, 없으면 그 지표는 None이다.
지표는 think 학습이 의도대로 먹혔는지 보는 형식 점검이며 지문 품질 평가가 아니다. 다만 복사율은 2차 학습의 정성 평가에서
낮은 점수를 받은 사례(원문 전재 5건, 구조 데이터 나열 3건)를 그대로 걸러냈다 (issue/0925/0925_plan.md 1.4절).
"""
import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Dict, Optional

from .common.config import data_dir
from .common.data import load_jsonl, think_messages_path
from .common.prompts import count_chars
from .inference.inputs import normalize

# 영어판 사고 과정의 고정 형식. UGRP2/data의 03_build_and_split_dataset.py(check)와 같은 값이어야 한다
CLOSING_LINE = "Plan complete. Writing the passage in Korean."
THINK_HEADERS = ["Task:", "Source Text overview:", "Structure Data check:", "Paragraph plan:", "Final check:"]
LEAKED_TAGS = ["<|channel>", "<channel|>", "<|think|>", "<|turn>"]

# 복사율: 답(공백 제거)이 입력과 k자 이상 연속 일치하는 구간으로 덮이는 비율. 문턱은 2026-09-25 실측으로 정했다.
#   원문 30자: 정답 지문 0.00, 베이스 최대 0.17, 원문을 전재한 어댑터 출력 0.29~0.90
#   구조 데이터 '사용된 정보' 명제 12자: 정답 지문 평균 0.09·최대 0.31, 베이스 최대 0.12, 구조 데이터를 나열한 어댑터 출력 0.40~0.92
COPY_SOURCE_MIN_CHARS = 30
COPY_STRUCTURE_MIN_CHARS = 12
COPY_SOURCE_FLAG = 0.20
COPY_STRUCTURE_FLAG = 0.35
_SPLIT_IN_NAME = re.compile(r"_(valid|test)(?:_|\.)")


def paragraphs(text: str) -> list:
    return [p for p in re.split(r"\n\s*\n|\n", text) if p.strip()]


def korean_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum("가" <= c <= "힣" for c in letters) / len(letters) if letters else 0.0


def planned_paragraphs(thinking: str) -> int:
    """사고 과정의 'Paragraph plan'에 적힌 문단 수 ('*   Paragraph N (~M chars)' 형식)."""
    return len(re.findall(r"^\s*\*\s+Paragraph \d+ \(~\d+ chars\)$", thinking, flags=re.MULTILINE))


def copy_coverage(text: str, source: str, k: int) -> float:
    """text(공백 제거)의 글자 중 source에도 있는 k자 이상 연속 구간에 속하는 비율. k-gram 집합으로 O(n)."""
    a, s = re.sub(r"\s", "", text), re.sub(r"\s", "", source)
    if len(a) < k or len(s) < k:
        return 0.0
    grams = {s[i:i + k] for i in range(len(s) - k + 1)}
    covered = [False] * len(a)
    for i in range(len(a) - k + 1):
        if a[i:i + k] in grams:
            covered[i:i + k] = [True] * k
    return sum(covered) / len(a)


def structure_propositions(structure: str) -> str:
    """구조 데이터에서 '사용된 정보' 줄의 명제만 모은다 ('핵심 내용' 요약은 지문에 옮겨도 이상하지 않아 제외)."""
    return "\n".join(line.split(":", 1)[1] for line in structure.splitlines() if "사용된 정보" in line and ":" in line)


def load_inputs(path) -> Dict[str, Dict]:
    """{id: {"reference_text", "structure", ...}}. messages 형식과 원본 필드 형식 모두 받는다."""
    return {item["id"]: item for item in (normalize(r) for r in load_jsonl(path))}


def default_inputs_path(result_path: Path) -> Optional[Path]:
    """결과 파일 이름의 split로 평가 입력 파일을 찾는다 (generation_think[_base]_{split}[_vllm][_{이름}].jsonl)."""
    match = _SPLIT_IN_NAME.search(result_path.name)
    if not match:
        return None
    agent = "structure" if "structure" in result_path.name else "generation"
    path = think_messages_path(data_dir(), agent, match.group(1))
    return path if path.is_file() else None


def summarize(path: Path, inputs: Optional[Dict[str, Dict]] = None) -> dict:
    rows = load_jsonl(path)
    think_rows = [r for r in rows if r["thinking"]]
    # length_error는 제시한 목표 분량 대비(평가 파일의 목표는 정답 분량의 반올림값이라 지시 준수를 보여 준다),
    # reference 쪽은 정답 지문의 실제 분량 대비다
    length_errors = [abs(count_chars(r["answer"]) - r["target_length"]) / r["target_length"]
                     for r in rows if r["answer"] and r.get("target_length")]
    reference_length_errors = [abs(count_chars(r["answer"]) - count_chars(r["reference"])) / count_chars(r["reference"])
                               for r in rows if r["answer"] and r.get("reference")]
    reference_matches = [len(paragraphs(r["answer"])) == len(paragraphs(r["reference"]))
                         for r in rows if r["answer"] and r.get("reference")]
    plan_matches = [len(paragraphs(r["answer"])) == planned_paragraphs(r["thinking"])
                    for r in think_rows if r["answer"] and planned_paragraphs(r["thinking"])]
    copy_rows = [(r, inputs[r["id"]]) for r in rows if inputs and r["answer"] and r["id"] in inputs]
    source_copy = [copy_coverage(r["answer"], item["reference_text"], COPY_SOURCE_MIN_CHARS) for r, item in copy_rows]
    structure_copy = [copy_coverage(r["answer"], structure_propositions(item["structure"]), COPY_STRUCTURE_MIN_CHARS)
                      for r, item in copy_rows]
    flagged = [r["id"] for (r, _), s, t in zip(copy_rows, source_copy, structure_copy)
               if s >= COPY_SOURCE_FLAG or t >= COPY_STRUCTURE_FLAG]

    def rate(values):
        return round(sum(values) / len(values), 3) if values else None

    return {
        "file": path.name,
        "samples": len(rows),
        "empty_answer": sum(1 for r in rows if not r["answer"]),
        "tag_leak_in_answer": sum(1 for r in rows if any(tag in r["answer"] for tag in LEAKED_TAGS)),
        "thinking_present": rate([bool(r["thinking"]) for r in rows]),
        "thinking_closing_line": rate([r["thinking"].rstrip().endswith(CLOSING_LINE) for r in think_rows]),
        "thinking_all_headers": rate([all(h in r["thinking"] for h in THINK_HEADERS) for r in think_rows]),
        "thinking_korean_ratio": round(statistics.mean(korean_ratio(r["thinking"]) for r in think_rows), 3) if think_rows else None,
        "thinking_chars_mean": int(statistics.mean(len(r["thinking"]) for r in think_rows)) if think_rows else None,
        "answer_chars_mean": int(statistics.mean(count_chars(r["answer"]) for r in rows)) if rows else None,
        "length_error_mean": round(statistics.mean(length_errors), 3) if length_errors else None,
        "length_error_vs_reference_mean": round(statistics.mean(reference_length_errors), 3) if reference_length_errors else None,
        "paragraphs_match_plan": rate(plan_matches),
        "paragraphs_match_reference": rate(reference_matches),
        "source_copy_ratio": round(statistics.mean(source_copy), 3) if source_copy else None,
        "structure_copy_ratio": round(statistics.mean(structure_copy), 3) if structure_copy else None,
        "copy_flagged": len(flagged) if copy_rows else None,
        "copy_flagged_ids": flagged if copy_rows else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--inputs", type=Path,
                        help="복사율 계산에 쓸 입력 파일 (messages 형식 또는 원본 필드 형식). 기본: 결과 파일 이름의 split로 data/SFT_think_en_{split}.jsonl")
    parser.add_argument("--output", type=Path, help="JSON으로도 저장")
    args = parser.parse_args()

    reports = []
    for path in args.results:
        inputs_path = args.inputs or default_inputs_path(path)
        if inputs_path is None:
            print(f"[WARN] {path.name}: 입력 파일을 찾지 못해 복사율 지표를 건너뜁니다 (--inputs로 지정)")
        reports.append(summarize(path, load_inputs(inputs_path) if inputs_path else None))
    keys = [k for k in reports[0] if k not in ("file", "copy_flagged_ids")]
    width = max(len(k) for k in keys)
    print(" " * width + " | " + " | ".join(r["file"] for r in reports))
    for key in keys:
        print(f"{key:<{width}} | " + " | ".join(f"{str(r[key]):>{len(r['file'])}}" for r in reports))
    for r in reports:
        if r.get("copy_flagged_ids"):
            print(f"[COPY] {r['file']}: 원문 {COPY_SOURCE_FLAG} 또는 구조 {COPY_STRUCTURE_FLAG} 이상 {len(r['copy_flagged_ids'])}건 {r['copy_flagged_ids']}")

    if args.output:
        args.output.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
