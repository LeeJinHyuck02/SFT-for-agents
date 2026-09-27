import os

import pytest

from ugrp.common.gemma_format import (COMPLETION_SUFFIX, build_completion, build_prompt, check_control_tokens,
                               parse_response, stop_token_ids)

THINKING = "과제: 4문단, 약 1800자\n문단별 계획:\n    * 1문단 (약 450자)\n구상 완료. 지문을 작성한다."
ANSWER = "첫 문단이다.\n\n둘째 문단이다."


def test_round_trip():
    assert parse_response(build_completion(THINKING, ANSWER)) == (THINKING, ANSWER)


def test_completion_layout():
    completion = build_completion(THINKING, ANSWER)
    assert completion.startswith("<|channel>thought\n과제:")
    assert completion.endswith("<channel|>" + ANSWER + COMPLETION_SUFFIX)


def test_parse_without_channel_keeps_answer():
    assert parse_response(ANSWER + "<turn|>") == ("", ANSWER)


def test_parse_strips_trailing_specials():
    raw = build_completion(THINKING, ANSWER).rstrip("\n") + "<eos>"
    assert parse_response(raw) == (THINKING, ANSWER)


def test_parse_drops_everything_after_first_turn_end():
    # 멈춤 토큰이 빠져 생성이 턴 끝을 넘어간 경우
    raw = build_completion(THINKING, ANSWER) + "<|turn>user\n다음 질문<turn|>\n<|turn>model\n엉뚱한 답"
    assert parse_response(raw) == (THINKING, ANSWER)


def test_parse_unclosed_channel_is_preserved_in_answer():
    # max_new_tokens에 걸려 사고가 끝나지 않은 경우: 잘라내지 말고 그대로 남겨 원인을 볼 수 있게 한다
    raw = "<|channel>thought\n과제: 미완"
    thinking, answer = parse_response(raw)
    assert thinking == "" and "과제: 미완" in answer


# 공식 repo는 gated이므로 HF_TOKEN이 필요하다. 토큰 없이 점검하려면 UGRP_TOKENIZER_ID에 공개 미러를 지정한다.
needs_tokenizer = pytest.mark.skipif(
    not (os.environ.get("HF_TOKEN") or os.environ.get("UGRP_TOKENIZER_ID")),
    reason="HF_TOKEN 또는 UGRP_TOKENIZER_ID 필요",
)


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    from ugrp.common.config import load_config
    model_id = os.environ.get("UGRP_TOKENIZER_ID") or load_config("generation")["model_id"]
    return AutoTokenizer.from_pretrained(model_id, token=os.environ.get("HF_TOKEN"))


@needs_tokenizer
def test_control_tokens_are_single(tokenizer):
    assert all(len(ids) == 1 for ids in check_control_tokens(tokenizer).values())


@needs_tokenizer
def test_stop_token_ids_include_turn_end(tokenizer):
    ids = stop_token_ids(tokenizer)
    assert ids[0] == tokenizer.eos_token_id and ids[1] == tokenizer.convert_tokens_to_ids("<turn|>")
    assert ids[1] != tokenizer.unk_token_id


@needs_tokenizer
def test_prompt_enables_thinking_and_ends_at_model_turn(tokenizer):
    prompt = build_prompt(tokenizer, "SYS", "USER")
    assert "<|think|>" in prompt and prompt.rstrip("\n").endswith("<|turn>model")


@needs_tokenizer
def test_separate_tokenization_matches_joint(tokenizer):
    # train.py는 prompt와 completion을 따로 토큰화해 이어 붙인다. 경계에서 토큰이 달라지면 안 된다.
    prompt = build_prompt(tokenizer, "SYS", "USER")
    completion = build_completion(THINKING, ANSWER)
    encode = lambda text: tokenizer(text, add_special_tokens=False)["input_ids"]
    assert encode(prompt) + encode(completion) == encode(prompt + completion)
    assert encode(prompt).count(tokenizer.bos_token_id) == 1
