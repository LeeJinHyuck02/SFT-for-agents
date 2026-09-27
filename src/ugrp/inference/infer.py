"""think mode 배치 추론. 학습과 동일한 4-bit 베이스에 어댑터를 붙여(merge 없이) 생성한다.

    python -m ugrp.inference.infer --agent generation                 # outputs/generation_think/final, data/SFT_think_en_test.jsonl
    python -m ugrp.inference.infer --agent generation --split valid
    python -m ugrp.inference.infer --agent generation --no-adapter     # 미세조정 전 베이스 모델
    python -m ugrp.inference.infer --agent generation --input data/my.jsonl --output results/my.jsonl --target-length 1800
    python -m ugrp.inference.infer --agent generation --batch-size 8   # 한 번에 생성할 건수 (기본 20)

입력 레코드는 두 형식을 받는다.
    messages 형식 (SFT_think_en_{split}.jsonl): system/user 턴을 그대로 쓴다. system에는 목표 분량(정답 지문 분량을
        100자 단위로 반올림한 지문별 값)이 이미 들어 있다. messages[2].content(정답 지문)는 결과에 reference로 옮겨 적는다.
    원본 필드 형식 (직접 만든 입력): {id, input_reference, input_prompt[, output_passage]}

여러 건을 왼쪽 패딩으로 묶어 한 번에 생성한다. 병목이 step마다의 파이썬 오버헤드라, 묶은 건수만큼 처리량이 는다.
프롬프트가 긴 것부터 묶어 패딩을 줄이고, 메모리가 부족하면 배치 크기를 절반으로 줄여 다시 시도한다.
결과는 배치마다 기록하며, 다시 실행하면 이미 처리한 id는 건너뛴다.
더 빠른 경로: vLLM(infer_vllm.py, scripts/run_infer.sh --backend vllm)
"""
import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from ..common import prompts
from ..common.config import data_dir, hf_token, load_config, output_dir, root_dir
from ..common.data import load_jsonl, think_messages_path
from ..common.gemma_format import build_prompt, stop_token_ids
from .inputs import build_inputs, load_items, result_row, split_done
from ..training.train import load_base_model, load_tokenizer

DEFAULT_BATCH_SIZE = 20  # test 19건, valid 18건을 한 배치로. 4-bit 가중치 ~18GB + KV ~1.8GB/건이라 80GB에 들어간다


def _trim(ids: List[int], stop_ids: List[int], pad_id: int) -> Tuple[List[int], str]:
    """생성된 id를 첫 멈춤 토큰까지(포함) 자른다. 먼저 끝난 행 뒤에 붙은 패딩도 버린다."""
    kept = []
    for token_id in ids:
        if token_id == pad_id:
            continue
        kept.append(token_id)
        if token_id in stop_ids:
            return kept, "stop"
    return kept, "length"


def generate_batch(model, tokenizer, prompt_texts: List[str], gen_config: dict) -> List[Tuple[str, int, str]]:
    """[(raw, 생성 토큰 수, finish_reason)]. finish_reason은 멈춤 토큰이면 "stop", 최대 길이에서 잘리면 "length"."""
    inputs = tokenizer(prompt_texts, return_tensors="pt", add_special_tokens=False,
                       padding=True, padding_side="left").to(model.device)
    stop_ids = stop_token_ids(tokenizer)
    with torch.no_grad():
        outputs = model.generate(**inputs, eos_token_id=stop_ids, pad_token_id=tokenizer.pad_token_id, **gen_config)
    results = []
    for row in outputs[:, inputs["input_ids"].shape[1]:].tolist():
        ids, finish_reason = _trim(row, stop_ids, tokenizer.pad_token_id)
        results.append((tokenizer.decode(ids, skip_special_tokens=False), len(ids), finish_reason))
    return results


def generate(model, tokenizer, prompt: str, gen_config: dict) -> str:
    """한 건 생성 (check_boundary.py 등)."""
    return generate_batch(model, tokenizer, [prompt], gen_config)[0][0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", default="generation", choices=["generation", "structure"])
    parser.add_argument("--adapter", help="기본값: outputs/{agent}_think/final. HF Hub repo 이름도 가능")
    parser.add_argument("--no-adapter", action="store_true", help="베이스 모델만으로 추론")
    parser.add_argument("--split", default="test")
    parser.add_argument("--input", help="기본값: data/SFT_think_en_{split}.jsonl")
    parser.add_argument("--output", help="기본값: results/{agent}_think_{split}.jsonl")
    parser.add_argument("--structure-from", help="기획 에이전트 추론 결과(jsonl)의 answer를 구조 데이터로 사용")
    parser.add_argument("--target-length", type=int,
                        help="목표 분량(자)을 모든 샘플에 이 값으로 덮어쓴다 (예: 1500 고정 조건 비교). 기본: 입력 파일의 값 "
                             "(평가 파일은 정답 분량의 %d자 단위 반올림값, 원본 필드 입력은 %d)"
                             % (prompts.TARGET_LENGTH_STEP, prompts.DEFAULT_TARGET_LENGTH))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, help="기본값: configs의 generate.max_new_tokens")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="한 번에 생성할 건수. 메모리가 부족하면 자동으로 절반씩 줄인다 (기본 %(default)s)")
    parser.add_argument("--tiny-debug", action="store_true")
    parser.add_argument("--tokenizer-id")
    args = parser.parse_args()

    config = load_config(args.agent)
    token = hf_token()
    run_dir = output_dir(args.agent)
    if args.tiny_debug:
        run_dir = run_dir.with_name(run_dir.name + "_tiny")
    adapter = None if args.no_adapter else args.adapter or str(run_dir / "final")

    tokenizer = load_tokenizer(args.tokenizer_id or config["model_id"], config["cache_dir"], token)
    gen_config = dict(config["generate"])
    if args.max_new_tokens:
        gen_config["max_new_tokens"] = args.max_new_tokens
    if args.tiny_debug:
        model = AutoModelForCausalLM.from_pretrained(run_dir / "tiny_base")
        gen_config["max_new_tokens"] = config["tiny"]["max_new_tokens"]
    else:
        model = load_base_model(config, token)
    if adapter is not None:
        print(f"LoRA 어댑터를 결합합니다: {adapter}")
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    items = load_items(args.input or think_messages_path(data_dir(), args.agent, args.split),
                       args.limit or (1 if args.tiny_debug else None))
    structure_map = {r["id"]: r["answer"] for r in load_jsonl(args.structure_from)} if args.structure_from else None

    suffix = ("_base" if args.no_adapter else "") + ("_tiny" if args.tiny_debug else "")  # 점검 결과가 실제 결과와 섞이지 않게
    output_path = Path(args.output) if args.output else root_dir() / "results" / f"{args.agent}_think{suffix}_{args.split}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    todo, _ = split_done(items, output_path)

    # (항목, 제시한 목표 분량, 프롬프트, 프롬프트 토큰 수). 긴 것부터 묶어야 한 배치 안의 패딩이 적다
    queue = []
    for item in todo:
        system, user, length = build_inputs(args.agent, item, args.target_length, structure_map)
        prompt = build_prompt(tokenizer, system, user)
        queue.append((item, length, prompt, len(tokenizer(prompt, add_special_tokens=False)["input_ids"])))
    queue.sort(key=lambda entry: entry[3], reverse=True)

    batch_size = max(1, args.batch_size)
    started, position, total_tokens = time.time(), 0, 0
    with output_path.open("a", encoding="utf-8", newline="\n") as outfile:
        while position < len(queue):
            batch = queue[position:position + batch_size]
            batch_started = time.time()
            try:
                results = generate_batch(model, tokenizer, [entry[2] for entry in batch], gen_config)
            except torch.cuda.OutOfMemoryError:
                if batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                print(f"[WARN] CUDA 메모리 부족 → 배치 크기를 {batch_size}로 줄여 다시 시도합니다.")
                torch.cuda.empty_cache()
                continue
            for (item, length, _, _), (raw, tokens, finish_reason) in zip(batch, results):
                row = result_row(item, length, raw, backend="hf", tokens=tokens, finish_reason=finish_reason)
                outfile.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"  {item['id']} 사고 {len(row['thinking'])}자 / 답 {len(row['answer'])}자 / {tokens}토큰 ({finish_reason})")
            outfile.flush()
            position += len(batch)
            batch_tokens = sum(tokens for _, tokens, _ in results)
            total_tokens += batch_tokens
            batch_seconds = time.time() - batch_started
            print(f"[{position}/{len(queue)}] 배치 {len(batch)}건 {batch_seconds / 60:.1f}분, "
                  f"{batch_tokens / batch_seconds:.1f} tok/s | 경과 {(time.time() - started) / 60:.0f}분")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    elapsed = time.time() - started
    if queue:
        print(f"[STATS] {len(queue)}건, {elapsed / 60:.1f}분, 생성 {total_tokens}토큰, {total_tokens / max(elapsed, 1e-9):.1f} tok/s")
    print(f"추론 완료: {output_path}")


if __name__ == "__main__":
    main()
