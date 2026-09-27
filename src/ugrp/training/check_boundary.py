"""미세조정 전 모델의 실제 think 출력과 학습 completion 형식의 경계(개행 포함)를 대조한다.

GPU가 있어야만 할 수 있는 유일한 데이터 검증이므로 Pod에서 모델을 받은 직후 가장 먼저 실행한다.
불일치하면 exit code 1 → run_train.sh가 본 학습에 들어가지 않는다. 이 경우 gemma_format.py와 UGRP2/data의
03_build_and_split_dataset.py(convert)를 함께 고치고, 로컬에서 convert를 다시 실행해 복사한 뒤 번들을 새로 올린다.

    python -m ugrp.training.check_boundary --agent generation
"""
import argparse
import sys

from ..common.config import data_dir, hf_token, load_config
from ..common.data import load_jsonl, think_messages_path
from ..common.gemma_format import THINK_CLOSE, THINK_OPEN, TURN_END, build_prompt
from ..inference.infer import generate
from ..inference.inputs import build_inputs, normalize
from .train import load_base_model, load_tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", default="generation", choices=["generation", "structure"])
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    args = parser.parse_args()

    config = load_config(args.agent)
    token = hf_token()
    tokenizer = load_tokenizer(config["model_id"], config["cache_dir"], token)
    model = load_base_model(config, token)
    model.eval()

    record = load_jsonl(think_messages_path(data_dir(), args.agent, "valid"))[0]
    system, user, _ = build_inputs(args.agent, normalize(record), None, None)
    prompt = build_prompt(tokenizer, system, user)
    gen_config = dict(config["generate"], max_new_tokens=args.max_new_tokens)
    raw = generate(model, tokenizer, prompt, gen_config)

    close_at = raw.find(THINK_CLOSE)
    print("=" * 60)
    print(f"[RAW 시작 80자] {raw[:80]!r}")
    if close_at >= 0:
        print(f"[RAW 사고 종료 전후] {raw[max(0, close_at - 40):close_at + len(THINK_CLOSE) + 40]!r}")
    print(f"[RAW 끝 40자] {raw[-40:]!r}")
    print("=" * 60)

    problems = []
    if not raw.startswith(THINK_OPEN):
        problems.append(f"출력이 {THINK_OPEN!r}로 시작하지 않습니다.")
    if close_at < 0:
        problems.append(f"{THINK_CLOSE!r}가 없습니다 (사고가 max_new_tokens 안에 끝나지 않았을 수 있음).")
    else:
        after = raw[close_at + len(THINK_CLOSE):close_at + len(THINK_CLOSE) + 1]
        if after.isspace():
            problems.append(f"{THINK_CLOSE!r} 직후에 공백/개행({after!r})이 있습니다. 학습 completion은 바로 답이 이어집니다.")
    if TURN_END not in raw[-len(TURN_END) - 2:]:
        problems.append(f"출력이 {TURN_END!r}로 끝나지 않습니다.")

    if problems:
        print("[MISMATCH] 학습 completion 형식과 실제 출력이 다릅니다:")
        for problem in problems:
            print(f"  - {problem}")
        sys.exit(1)
    print("[OK] 실제 출력의 경계가 학습 completion 형식과 일치합니다.")


if __name__ == "__main__":
    main()
