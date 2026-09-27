"""Gemma 4 think mode 문자열 포맷의 단일 출처.

Gemma 4의 chat template은 assistant 턴의 thinking 블록을 제거하므로(HF discussions #1/#77/#118),
프롬프트만 template으로 만들고 completion은 여기서 직접 조립한다.
학습과 추론은 모두 이 모듈만 사용한다. 학습 데이터(SFT_think_en_train_gemma4.jsonl)는 UGRP2/data의
03_build_and_split_dataset.py convert가 같은 형식으로 만든다. 여기서 형식을 고치면 그쪽도 함께 고쳐야 하며,
scripts/make_bundle.py가 받은 파일의 형식을 아래 상수로 검사한다.

    prompt     : <bos><|turn>system\n<|think|>{system}<turn|>\n<|turn>user\n{user}<turn|>\n<|turn>model\n
    completion : <|channel>thought\n{thinking}<channel|>{answer}<turn|>\n

경계의 개행 규칙은 ugrp.training.check_boundary로 실제 모델 출력과 대조해 확인한다 (Pod에서 실행).
"""
import re
from typing import Dict, List, Tuple

THINK_OPEN = "<|channel>thought\n"
THINK_CLOSE = "<channel|>"
TURN_END = "<turn|>"
COMPLETION_SUFFIX = TURN_END + "\n"
CONTROL_TOKENS = ["<|think|>", "<|channel>", "<channel|>", "<turn|>"]

_THINK_PATTERN = re.compile(r"<\|channel>thought\n(.*?)<channel\|>", re.DOTALL)
_TRAILING_SPECIALS = re.compile(r"(?:\s*(?:<turn\|>|<eos>|<pad>|<end_of_turn>))+\s*$")


def build_messages(system: str, user: str) -> List[Dict[str, str]]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_prompt(tokenizer, system: str, user: str) -> str:
    """generation prompt까지의 문자열. template의 add_generation_prompt 분기는 정상 동작한다."""
    return tokenizer.apply_chat_template(
        build_messages(system, user),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def build_completion(thinking: str, answer: str) -> str:
    """think 학습용 completion."""
    return f"{THINK_OPEN}{thinking}{THINK_CLOSE}{answer}{COMPLETION_SUFFIX}"


def stop_token_ids(tokenizer) -> List[int]:
    """생성을 멈출 토큰 [<eos>, <turn|>]. 토크나이저의 eos_token은 <eos> 하나뿐이라 이것만 쓰면 턴이 끝나도 멈추지 않는다."""
    return [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids(TURN_END)]


def parse_response(raw: str) -> Tuple[str, str]:
    """skip_special_tokens=False로 디코딩한 모델 출력을 (thinking, answer)로 분리한다.

    첫 <turn|> 이후는 버린다: 멈춤 토큰 설정이 빠져 생성이 턴 끝을 넘어가도 결과가 깨지지 않게 한다.
    """
    text = _TRAILING_SPECIALS.sub("", raw.split(TURN_END, 1)[0])
    match = _THINK_PATTERN.search(text)
    if match:
        thinking = match.group(1).strip()
        answer = text[:match.start()] + text[match.end():]
    else:
        thinking = ""
        answer = text
    return thinking, answer.strip()


def check_control_tokens(tokenizer) -> Dict[str, List[int]]:
    """제어 토큰이 각각 단일 특수 토큰으로 인코딩되는지 확인한다. 아니면 ValueError."""
    result = {}
    broken = []
    for token in CONTROL_TOKENS:
        ids = tokenizer(token, add_special_tokens=False)["input_ids"]
        result[token] = ids
        if len(ids) != 1:
            broken.append(token)
    if broken:
        raise ValueError(f"단일 토큰으로 인코딩되지 않는 제어 토큰: {broken} ({result})")
    return result
