"""추론 입력 처리와 결과 행 형식. HF 경로(infer.py)와 vLLM 경로(infer_vllm.py)가 함께 쓴다.

vLLM은 학습 환경과 분리된 venv에서 돌기 때문에, 이 모듈은 torch나 학습 패키지(trl, peft, bitsandbytes)를
import하지 않는다. 의존하는 것은 prompts, data, gemma_format(표준 라이브러리 수준)뿐이다.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..common import prompts
from ..common.data import load_jsonl
from ..common.gemma_format import parse_response

_STRUCTURE_MARK = "\n\n구조 데이터:\n"
_REFERENCE_MARK = "원본 문서:\n"


def normalize(record: Dict) -> Dict:
    """두 입력 형식을 {id, reference_text, structure, target_length, gold, system, user}로 맞춘다."""
    if "messages" in record:
        system, user = record["messages"][0]["content"], record["messages"][1]["content"]
        gold = record["messages"][2]["content"] if len(record["messages"]) > 2 else None
        body, _, structure = user.partition(_STRUCTURE_MARK)
        return {"id": record["id"], "system": system, "user": user, "gold": gold,
                "reference_text": body[len(_REFERENCE_MARK):] if body.startswith(_REFERENCE_MARK) else body,
                "structure": structure, "target_length": record.get("target_length")}
    # 원본 필드 입력에는 정답이 없어 지문별 목표 분량을 만들 수 없다. None이면 build_inputs가 prompts.DEFAULT_TARGET_LENGTH(1500)를 쓴다
    # (평가 파일의 target_length는 정답 지문 분량을 100자 단위로 반올림한 값이다: prompts.target_length_for)
    return {"id": record["id"], "system": None, "user": None, "gold": record.get("output_passage"),
            "reference_text": record["input_reference"], "structure": record.get("input_prompt", ""),
            "target_length": None}


def build_inputs(agent: str, item: Dict, target_length: Optional[int], structure_map: Optional[Dict]) -> tuple:
    """(system, user, 실제로 제시한 목표 분량). 덮어쓸 것이 없으면 파일의 system/user를 그대로 쓴다."""
    if agent == "structure":
        return (item["system"] or prompts.structure_system(),
                item["user"] or prompts.structure_user(item["reference_text"]), None)
    length = target_length or item["target_length"] or prompts.DEFAULT_TARGET_LENGTH
    structure = structure_map.get(item["id"], item["structure"]) if structure_map else item["structure"]
    untouched = item["system"] is not None and target_length is None and structure == item["structure"]
    if untouched:
        return item["system"], item["user"], length
    return prompts.generation_system(length), prompts.generation_user(item["reference_text"], structure), length


def load_items(path, limit: Optional[int] = None) -> List[Dict]:
    items = [normalize(r) for r in load_jsonl(path)]
    return items[:limit] if limit else items


def split_done(items: List[Dict], output_path: Path) -> Tuple[List[Dict], int]:
    """(아직 처리하지 않은 항목, 이미 처리한 건수). 결과 파일에 있는 id는 건너뛴다."""
    done = {r["id"] for r in load_jsonl(output_path)} if output_path.exists() else set()
    todo = [item for item in items if item["id"] not in done]
    if done:
        print(f"[INFO] 기존 결과 {len(items) - len(todo)}건은 건너뜁니다. "
              f"(처음부터 다시 하려면 {output_path.name}을 지우거나 --output 변경)")
    return todo, len(items) - len(todo)


def result_row(item: Dict, length: Optional[int], raw: str, **extra) -> Dict:
    """두 경로가 같은 필드를 쓰도록 결과 행을 한 곳에서 만든다. extra는 경로별 추가 필드."""
    thinking, answer = parse_response(raw)
    return {"id": item["id"], "target_length": length, "thinking": thinking, "answer": answer,
            "reference": item["gold"], "raw": raw, **extra}
