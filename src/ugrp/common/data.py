"""JSONL 입출력과 데이터 파일 명명 규칙.

데이터는 여기서 만들지 않는다. UGRP2/data(03_build_and_split_dataset.py의 merge, convert)가 만든 완성 파일을
이름 그대로 data/에 복사해 쓴다. 사고 과정은 영어판(_en)만 쓴다.
    SFT_think_en_train_gemma4.jsonl : {id, prompt, completion}. 학습은 이 파일만 읽는다
    SFT_think_en_{valid,test}.jsonl : messages[system, user, assistant{content}]. content는 정답 지문이며
                                      모델이 직접 생성한 결과와 비교하는 용도
"""
import json
from pathlib import Path
from typing import Dict, Iterable, List

EVAL_SPLITS = ("valid", "test")


def load_jsonl(path) -> List[Dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(path, records: Iterable[Dict]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _prefix(agent: str) -> str:
    return "SFT_think_en" if agent == "generation" else f"SFT_{agent}_think_en"


def think_messages_path(data_dir, agent: str, split: str) -> Path:
    """messages 형식: [system, user, assistant]. 추론·평가는 이 파일을 직접 읽는다."""
    return Path(data_dir) / f"{_prefix(agent)}_{split}.jsonl"


def sft_path(data_dir, agent: str) -> Path:
    """UGRP2/data의 03 convert가 train에서 만든 prompt/completion 파일. 학습은 이 파일만 읽는다."""
    return Path(data_dir) / f"{_prefix(agent)}_train_gemma4.jsonl"
