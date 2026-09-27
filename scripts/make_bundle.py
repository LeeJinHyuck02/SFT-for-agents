"""Pod 업로드용 번들(zip)을 만든다. 로컬에서 실행.

    python scripts/make_bundle.py

번들에는 코드와 학습용 SFT_think_en_train_gemma4.jsonl, 추론·평가용 SFT_think_en_{valid,test}.jsonl만 들어간다.
세 파일은 UGRP2/data(03 merge → 03 convert)가 만든 것을 data/에 복사해 둔 것이다. 여기서는 만들지 않고 검사만 한다:
형식이 agents의 gemma_format·prompts와 맞는지(목표 분량이 지문 분량의 반올림값인지 포함), UGRP2/data의 같은 이름 파일과 내용이 같은지(복사 누락).
legacy, 어댑터, 노트북, .env는 넣지 않으며, 토큰처럼 보이는 문자열이 있으면 중단한다.
"""
import argparse
import hashlib
import json
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ugrp.common import prompts  # noqa: E402
from ugrp.common.config import data_dir, load_config  # noqa: E402
from ugrp.common.data import EVAL_SPLITS, load_jsonl, sft_path, think_messages_path  # noqa: E402
from ugrp.common.gemma_format import COMPLETION_SUFFIX, THINK_CLOSE, THINK_OPEN  # noqa: E402

CODE_PATHS = ["src", "configs", "scripts", "pyproject.toml", "requirements.txt", "requirements-vllm.txt"]
SECRET_PATTERN = re.compile(rb"hf_[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_\-]{30,}|sk-ant-[A-Za-z0-9_\-]{20,}")
BUNDLE_ROOT = "ugrp"
SOURCE_HINT = "UGRP2/data에서 03_build_and_split_dataset.py merge → convert 후 세 파일을 data/에 복사하십시오."
PROMPT_TAIL = "<|turn>model\n"
MAX_LISTED = 5


def collect_code() -> list:
    files = []
    for entry in CODE_PATHS:
        path = ROOT / entry
        candidates = [path] if path.is_file() else sorted(path.rglob("*"))
        files += [p for p in candidates if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    return files


def _summarize(name: str, what: str, ids: list) -> list:
    return [f"{name}: {what} {len(ids)}건 {ids[:MAX_LISTED]}"] if ids else []


def passage_of(completion: str) -> str:
    """completion(<|channel>thought … <channel|>지문<turn|> 개행)에서 지문만 잘라 낸다."""
    body = completion.split(THINK_CLOSE, 1)[1] if THINK_CLOSE in completion else completion
    return body[:-len(COMPLETION_SUFFIX)] if body.endswith(COMPLETION_SUFFIX) else body


def check_train(path: Path, records: list) -> list:
    """completion의 태그 배치와, prompt에 든 집필 지시문·목표 분량이 agents의 prompts와 같은지 (토크나이저 없이 되는 검사).

    목표 분량은 지문별 값이다: convert가 넣은 target_length가 completion 속 지문 분량의 반올림값(prompts.target_length_for)이고,
    prompt의 system 턴에 그 숫자로 만든 집필 지시문이 있어야 한다.
    """
    bad_completion = [r["id"] for r in records
                      if not (r["completion"].startswith(THINK_OPEN) and r["completion"].count(THINK_CLOSE) == 1
                              and r["completion"].endswith(COMPLETION_SUFFIX))]
    bad_length = [r["id"] for r in records
                  if r.get("target_length") != prompts.target_length_for(passage_of(r["completion"]))]
    bad_prompt = [r["id"] for r in records
                  if not ("<|think|>" in r["prompt"] and prompts.generation_system(r.get("target_length")) in r["prompt"]
                          and r["prompt"].endswith(PROMPT_TAIL))]
    return (_summarize(path.name, "completion 형식이 gemma_format과 다름", bad_completion)
            + _summarize(path.name, "target_length가 없거나 지문 분량의 반올림값(prompts.target_length_for)이 아님 → 03 convert를 다시 실행",
                         bad_length)
            + _summarize(path.name, "prompt에 target_length로 만든 prompts.generation_system이 없거나 think 프롬프트 꼴이 아님", bad_prompt))


def check_eval(path: Path, records: list) -> list:
    bad_length = [r["id"] for r in records
                  if r.get("target_length") != prompts.target_length_for(r["messages"][2]["content"])]
    bad_system = [r["id"] for r in records
                  if r["messages"][0]["content"] != prompts.generation_system(r.get("target_length"))]
    bad_user = [r["id"] for r in records
                if not (r["messages"][1]["content"].startswith("원본 문서:\n") and "\n\n구조 데이터:\n" in r["messages"][1]["content"])]
    leaked = [r["id"] for r in records if "reasoning" in r["messages"][2]]
    return (_summarize(path.name, "target_length가 정답 지문 분량의 반올림값(prompts.target_length_for)이 아님 → 03 merge를 다시 실행", bad_length)
            + _summarize(path.name, "system 턴이 target_length로 만든 prompts.generation_system과 다름", bad_system)
            + _summarize(path.name, "user 턴이 prompts.generation_user 꼴이 아님", bad_user)
            + _summarize(path.name, "평가용인데 reasoning이 들어 있음", leaked))


def check_stale(paths: list) -> list:
    """수동 복사 누락 확인. UGRP2/data에 같은 이름의 파일이 있는데 내용이 다르면 그쪽이 새로 만든 것이다. Pod 등 없으면 건너뛴다."""
    source_dir = ROOT.parent / "data"
    problems = []
    for path in paths:
        source = source_dir / path.name
        if source.resolve() != path.resolve() and source.exists() and source.read_bytes() != path.read_bytes():
            problems.append(f"{path.name}: {source_dir}의 파일과 내용이 다릅니다 → 다시 복사하십시오")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", default="generation", choices=["generation", "structure"])
    args = parser.parse_args()

    base_dir = data_dir()
    config = load_config(args.agent)

    train_sft = sft_path(base_dir, args.agent)
    eval_files = [think_messages_path(base_dir, args.agent, split) for split in EVAL_SPLITS]
    data_files = [train_sft] + eval_files
    counts = {}
    problems = [f"{path.name} 없음 → {SOURCE_HINT}" for path in data_files if not path.exists()]
    if not problems:
        records = {path: load_jsonl(path) for path in data_files}
        counts = {path.name: len(rows) for path, rows in records.items()}
        problems += check_stale(data_files)
        if args.agent == "generation":  # structure 에이전트는 프롬프트·형식 규약이 아직 없다
            problems += check_train(train_sft, records[train_sft])
            for path in eval_files:
                problems += check_eval(path, records[path])

    if problems:
        print("[ABORT] 번들을 만들 수 없습니다:")
        for problem in problems:
            print(f"  - {problem}")
        sys.exit(1)

    entries = [(p, f"{BUNDLE_ROOT}/{p.relative_to(ROOT).as_posix()}") for p in collect_code()]
    entries += [(p, f"{BUNDLE_ROOT}/data/{p.name}") for p in data_files]

    leaked = [str(p) for p, _ in entries if SECRET_PATTERN.search(p.read_bytes())]
    if leaked:
        print("[ABORT] 토큰으로 보이는 문자열이 포함된 파일:")
        for path in leaked:
            print(f"  - {path}")
        sys.exit(1)

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    manifest = {
        "bundle": stamp,
        "agent": args.agent,
        "model_id": config["model_id"],
        "max_len": config["max_len"],
        "counts": counts,
        "sha256": {name: hashlib.sha256(p.read_bytes()).hexdigest() for p, name in entries},
    }

    out_dir = ROOT / "bundles"
    out_dir.mkdir(exist_ok=True)
    bundle_path = out_dir / f"ugrp_bundle_{stamp}.zip"
    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path, name in entries:
            if path.suffix == ".sh":  # Windows에서 CRLF로 저장되면 Pod의 bash가 실행하지 못한다
                bundle.writestr(name, path.read_bytes().replace(b"\r\n", b"\n"))
            else:
                bundle.write(path, name)
        bundle.writestr(f"{BUNDLE_ROOT}/MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))

    print(f"완료: {bundle_path} ({bundle_path.stat().st_size / 1024 ** 2:.1f} MB, 파일 {len(entries) + 1}개)")
    for name, count in counts.items():
        print(f"  {name}: {count}건")
    print("\nPod에서:  unzip ugrp_bundle_*.zip && cd ugrp && export HF_TOKEN=... && bash scripts/run_train.sh --detach")


if __name__ == "__main__":
    main()
