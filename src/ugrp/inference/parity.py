"""vLLM --parity의 기준값 파일 규약과, HF·vLLM 두 경로가 함께 쓰는 형식 학습 게이지 계산.
HF 쪽(parity_reference.py)이 쓰고 vLLM 쪽(infer_vllm.py)이 읽는다.

vLLM venv에서도 import하므로 torch나 학습 패키지를 쓰지 않는다.

    outputs/{agent}_think/parity_reference.json
        {"samples": [{"id", "sha1"}], "base": [NLL...], "adapters": {"final": [NLL...]}, "measured_with": "hf-4bit",
         "details": {"base": [{think, answer, first_prob, first_token, top}, ...], "final": [...]}}

기준값은 데이터와 어댑터에 묶여 있다. samples의 sha1(prompt+completion)이 현재 학습 파일과 다르면 쓰지 않는다.
details는 형식 학습 게이지(2026-09-25, issue/0925/0925_plan.md 5절)이며 parity 판정에는 쓰지 않는다.
    think / answer : completion을 <channel|>에서 나눈 사고 구간·지문 구간의 평균 NLL
    first_prob     : <|channel>thought 개행 직후 첫 내용 토큰(학습 데이터는 모두 'Task')의 확률. 2차 학습 어댑터는 0.025였다
"""
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..common.gemma_format import THINK_CLOSE, THINK_OPEN

PARITY_SAMPLES = 2  # 학습 파일 앞에서 이 건수만 잰다 (건당 8k 토큰 안팎의 forward 한 번)
REFERENCE_NAME = "parity_reference.json"
TOP_CANDIDATES = 3           # 사고 첫 토큰 위치에서 기록할 상위 후보 수
FORMAT_GAUGE_MIN_PROB = 0.5  # 사고 첫 토큰 확률이 이보다 낮으면 사고 형식을 아직 배우지 못한 것으로 본다


def reference_path(run_dir) -> Path:
    return Path(run_dir) / REFERENCE_NAME


def sample_signature(records: List[Dict]) -> List[Dict]:
    return [{"id": r["id"], "sha1": hashlib.sha1((r["prompt"] + r["completion"]).encode("utf-8")).hexdigest()[:12]}
            for r in records]


def save_reference(path, records: List[Dict], base: List[float], adapters: Dict[str, List[float]],
                   details: Optional[Dict[str, List[Dict]]] = None) -> None:
    payload = {"samples": sample_signature(records), "base": base, "adapters": adapters, "measured_with": "hf-4bit"}
    if details:
        payload["details"] = details
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_reference(path, records: List[Dict]) -> Tuple[Optional[Dict], str]:
    """(기준값 또는 None, 쓰지 못하는 이유). 같은 샘플로 잰 것일 때만 돌려준다."""
    path = Path(path)
    if not path.is_file():
        return None, f"{path} 없음"
    reference = json.loads(path.read_text(encoding="utf-8"))
    if reference.get("samples") != sample_signature(records):
        return None, f"{path.name}은 다른 학습 데이터로 잰 것입니다 (샘플 id 또는 내용이 다름)"
    return reference, ""


def think_bounds(tokenizer, completion_ids: List[int]) -> Tuple[Optional[int], Optional[int]]:
    """(사고 첫 내용 토큰의 index, <channel|>의 index). 학습·생성 모두 <|channel> thought 개행 Task 순으로 토큰화된다 (0925.md 3장).

    completion이 THINK_OPEN 토큰열로 시작하지 않으면 first는 None, <channel|>가 없으면 close는 None이다.
    """
    open_ids = tokenizer(THINK_OPEN, add_special_tokens=False)["input_ids"]
    first = len(open_ids) if len(completion_ids) > len(open_ids) and completion_ids[:len(open_ids)] == open_ids else None
    close_id = tokenizer.convert_tokens_to_ids(THINK_CLOSE)
    close = completion_ids.index(close_id) if close_id in completion_ids else None
    return first, close


def summarize_logprobs(logprobs: List[float], first: Optional[int], close: Optional[int]) -> Dict:
    """토큰별 logprob → {nll, think, answer, first_prob}. think는 처음부터 <channel|>까지, answer는 그 뒤(지문 + <turn|>)."""
    def mean_nll(values):
        return round(-sum(values) / len(values), 4) if values else None

    return {"nll": mean_nll(logprobs),
            "think": mean_nll(logprobs[:close + 1]) if close is not None else None,
            "answer": mean_nll(logprobs[close + 1:]) if close is not None else None,
            "first_prob": round(math.exp(logprobs[first]), 4) if first is not None else None}


def describe_gauge(stats: List[Dict]) -> str:
    """형식 학습 게이지 한 줄. stats는 샘플별 summarize_logprobs 결과 (+ 선택: first_token, top)."""
    return (f"사고 NLL {[s.get('think') for s in stats]} / 지문 NLL {[s.get('answer') for s in stats]} / "
            f"P(사고 첫 토큰) {[s.get('first_prob') for s in stats]} / 상위 후보 {[s.get('top') for s in stats]}")


def format_learned(stats: List[Dict]) -> bool:
    """모든 샘플에서 사고 첫 토큰 확률이 FORMAT_GAUGE_MIN_PROB 이상이면 사고 형식을 배운 것으로 본다."""
    return all(s.get("first_prob") is not None and s["first_prob"] >= FORMAT_GAUGE_MIN_PROB for s in stats)
