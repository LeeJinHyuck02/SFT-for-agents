"""prompts.py가 데이터 합성·병합 쪽(UGRP2/data)과 같은 문자열·규칙을 쓰는지, 받은 데이터가 기대한 형태인지 검증한다."""
import ast

import pytest

from ugrp.common import prompts
from ugrp.common.config import data_dir, root_dir
from ugrp.common.data import EVAL_SPLITS, load_jsonl, sft_path, think_messages_path
from ugrp.common.gemma_format import COMPLETION_SUFFIX, THINK_CLOSE, THINK_OPEN


def _script_constant(path, name):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == name:
            return ast.literal_eval(node.value)
    raise KeyError(name)


def test_generation_system_formats_target_length():
    text = prompts.generation_system(1800)
    assert "1800자 내외" in text and "{target_length}" not in text


def test_user_turn_layout():
    assert prompts.generation_user("R", "S") == "원본 문서:\nR\n\n구조 데이터:\nS"
    assert prompts.structure_user("R") == "입력 지문:\nR"


def test_target_length_rounding():
    """목표 분량은 정답 지문의 공백 제외 글자 수를 100자 단위로 반올림한다 (반은 올림). 03 merge의 target_length_for와 같은 규칙."""
    assert prompts.TARGET_LENGTH_STEP == 100
    assert prompts.target_length_for("가" * 1249) == 1200
    assert prompts.target_length_for("가" * 1250) == 1300
    assert prompts.target_length_for("가 나\n다" * 500) == 1500  # 공백·줄바꿈은 세지 않는다


@pytest.mark.parametrize("script,name,expected", [
    ("03_build_and_split_dataset.py", "SYSTEM_PROMPT_TEMPLATE", prompts.GENERATION_SYSTEM),
    ("04_synthesize_reasoning_trace.py", "WRITER_INSTRUCTION_TEMPLATE", prompts.GENERATION_SYSTEM),
    ("03_build_and_split_dataset.py", "DEFAULT_TARGET_LENGTH", prompts.DEFAULT_TARGET_LENGTH),
    ("03_build_and_split_dataset.py", "TARGET_LENGTH_STEP", prompts.TARGET_LENGTH_STEP),
    ("04_synthesize_reasoning_trace.py", "DEFAULT_TARGET_LENGTH", prompts.DEFAULT_TARGET_LENGTH),
])
def test_matches_data_scripts(script, name, expected):
    # 스크립트는 합성 작업 폴더(UGRP2/data)에 있다. Pod 번들에는 없다
    path = root_dir().parent / "data" / script
    if not path.exists():
        pytest.skip(f"{script} 없음")
    assert _script_constant(path, name) == expected


@pytest.mark.parametrize("split", EVAL_SPLITS)
def test_matches_eval_data(split):
    """모든 레코드의 system/user 턴이 prompts.py로 재현된다 → 추론 때 파일의 프롬프트를 그대로 써도 안전하다."""
    path = think_messages_path(data_dir(), "generation", split)
    if not path.exists():
        pytest.skip(f"{path.name} 없음")
    for record in load_jsonl(path):
        system, user, assistant = record["messages"]
        # 목표 분량은 정답 지문 분량의 반올림값이고, system 턴은 그 값으로 만든 것이어야 한다 (03 merge와 같은 규칙)
        assert record["target_length"] == prompts.target_length_for(assistant["content"]), record["id"]
        assert system["content"] == prompts.generation_system(record["target_length"]), record["id"]
        assert user["content"].startswith("원본 문서:\n") and "\n\n구조 데이터:\n" in user["content"], record["id"]
        # valid/test는 생성 결과와 비교할 정답 지문만 갖는다
        assert "reasoning" not in assistant, record["id"]


def test_train_data_format():
    path = sft_path(data_dir(), "generation")
    if not path.exists():
        pytest.skip(f"{path.name} 없음")
    for record in load_jsonl(path):
        completion = record["completion"]
        assert completion.startswith(THINK_OPEN) and completion.count(THINK_CLOSE) == 1, record["id"]
        assert completion.endswith(COMPLETION_SUFFIX), record["id"]
        # convert가 넣은 target_length는 completion 속 지문의 분량에서 나온 값이고, prompt의 system 턴에 그 숫자가 들어 있다
        passage = completion.split(THINK_CLOSE, 1)[1][:-len(COMPLETION_SUFFIX)]
        assert record["target_length"] == prompts.target_length_for(passage), record["id"]
        assert prompts.generation_system(record["target_length"]) in record["prompt"], record["id"]
        assert record["prompt"].endswith("<|turn>model\n"), record["id"]


def test_infer_inputs_keep_or_rebuild_prompt():
    from ugrp.inference.inputs import build_inputs, normalize

    record = {"id": "x", "target_length": 1500, "messages": [
        {"role": "system", "content": prompts.generation_system(1500)},
        {"role": "user", "content": prompts.generation_user("원문 A\n\n둘째 문단", "대주제: B")},
        {"role": "assistant", "content": "정답"}]}
    item = normalize(record)
    assert (item["reference_text"], item["structure"], item["gold"]) == ("원문 A\n\n둘째 문단", "대주제: B", "정답")
    # 덮어쓰지 않으면 파일의 프롬프트 그대로
    assert build_inputs("generation", item, None, None) == (
        record["messages"][0]["content"], record["messages"][1]["content"], 1500)
    # 목표 분량·구조도를 덮어쓰면 같은 형식으로 재조립
    system, user, length = build_inputs("generation", item, 1800, {"x": "대주제: C"})
    assert length == 1800 and "1800자 내외" in system
    assert user == prompts.generation_user("원문 A\n\n둘째 문단", "대주제: C")


def test_raw_input_uses_fixed_target_length():
    """정답이 없는 원본 필드 입력은 지문별 값을 만들 수 없으므로 DEFAULT_TARGET_LENGTH를 쓴다 (output_passage가 있어도 마찬가지)."""
    from ugrp.inference.inputs import build_inputs, normalize

    item = normalize({"id": "x", "input_reference": "원문", "input_prompt": "대주제: B", "output_passage": "가" * 2000})
    system, user, length = build_inputs("generation", item, None, None)
    assert length == prompts.DEFAULT_TARGET_LENGTH and system == prompts.generation_system(length)
    assert user == prompts.generation_user("원문", "대주제: B")


def test_file_names():
    assert sft_path("d", "generation").name == "SFT_think_en_train_gemma4.jsonl"
    assert sft_path("d", "structure").name == "SFT_structure_think_en_train_gemma4.jsonl"
    assert think_messages_path("d", "generation", "valid").name == "SFT_think_en_valid.jsonl"
