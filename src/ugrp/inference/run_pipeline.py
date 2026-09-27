"""기획(structure) → 생성(generation) 연쇄 추론.

두 에이전트의 어댑터가 모두 있을 때 사용한다. 31B 베이스를 두 번 올리지 않도록 단계를 나눠 실행한다.

    python -m ugrp.inference.run_pipeline --split test
"""
import argparse
import subprocess
import sys

from ..common.config import root_dir


def run(module: str, *cli_args: str) -> None:
    command = [sys.executable, "-m", module, *cli_args]
    print(f"[RUN] {' '.join(command)}")
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test")
    parser.add_argument("--target-length", type=int, help="기본: 입력 파일의 목표 분량 (평가 파일은 정답 분량의 100자 단위 반올림값)")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    results = root_dir() / "results"
    structure_output = results / f"pipeline_structure_{args.split}.jsonl"
    passage_output = results / f"pipeline_generation_{args.split}.jsonl"
    limit = ["--limit", str(args.limit)] if args.limit else []
    length = ["--target-length", str(args.target_length)] if args.target_length else []

    run("ugrp.inference.infer", "--agent", "structure", "--split", args.split, "--output", str(structure_output), *limit)
    run("ugrp.inference.infer", "--agent", "generation", "--split", args.split, "--output", str(passage_output),
        "--structure-from", str(structure_output), *length, *limit)
    print(f"연쇄 추론 완료: {passage_output}")


if __name__ == "__main__":
    main()
