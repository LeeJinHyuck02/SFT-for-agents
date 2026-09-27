"""vLLM --parity가 비교할 기준값을 HF 4-bit 경로(학습과 같은 조건)로 잰다. Pod에서 학습 직후 실행한다.

    python -m ugrp.inference.parity_reference --agent generation                                # outputs/generation_think/final
    python -m ugrp.inference.parity_reference --adapter outputs/generation_think/checkpoint-37   # epoch 1 체크포인트

학습 파일(SFT_think_en_train_gemma4.jsonl) 앞 2건의 정답(completion)을 teacher forcing으로 넣어,
베이스 모델과 어댑터를 붙인 모델의 평균 NLL을 잰다. 생성은 하지 않는다.
같은 forward에서 형식 학습 게이지도 낸다 (2026-09-25, issue/0925/0925_plan.md 5절): 사고 구간·지문 구간의 NLL과
사고 첫 토큰('Task')의 확률·상위 후보. 생성 평가 전에 어댑터가 사고 형식을 배웠는지 수 분 만에 알 수 있다.
    결과: outputs/{agent}_think/parity_reference.json (pack_results.sh가 함께 묶는다)
어댑터 이름은 폴더 이름(final, checkpoint-37 …)이고 infer_vllm.py의 기본 이름과 같다. 다시 실행하면 그 이름의 값만 바뀐다.
"""
import argparse
from pathlib import Path
from typing import Dict, List

import torch
from peft import PeftModel

from ..common.config import data_dir, hf_token, load_config, output_dir
from ..common.data import load_jsonl, sft_path
from ..training.train import load_base_model, load_tokenizer
from .parity import (FORMAT_GAUGE_MIN_PROB, PARITY_SAMPLES, TOP_CANDIDATES, describe_gauge, format_learned,
                     load_reference, reference_path, save_reference, summarize_logprobs, think_bounds)

LOGPROB_CHUNK = 1024  # 어휘 262k개짜리 logits를 fp32로 한꺼번에 softmax하지 않도록 나눠서 계산한다


def completion_stats(model, tokenizer, prompt_ids: List[int], completion_ids: List[int]) -> Dict:
    """teacher forcing으로 completion의 평균 NLL(전체·사고·지문)과 사고 첫 토큰의 확률·상위 후보. infer_vllm.run_parity와 같은 양을 잰다."""
    ids = torch.tensor([prompt_ids + completion_ids], device=model.device)
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0, len(prompt_ids) - 1:-1]  # i번째 토큰은 i-1 위치의 logits가 예측한다
    targets = ids[0, len(prompt_ids):].to(logits.device)
    logprobs = []
    for start in range(0, len(targets), LOGPROB_CHUNK):
        chunk = torch.log_softmax(logits[start:start + LOGPROB_CHUNK].float(), dim=-1)
        logprobs += chunk.gather(1, targets[start:start + LOGPROB_CHUNK, None]).squeeze(1).tolist()
    first, close = think_bounds(tokenizer, completion_ids)
    stats = summarize_logprobs(logprobs, first, close)
    if first is not None:
        top = torch.topk(torch.softmax(logits[first].float(), dim=-1), TOP_CANDIDATES)
        stats["first_token"] = tokenizer.decode([completion_ids[first]])
        stats["top"] = [(tokenizer.decode([int(i)]), round(float(p), 4)) for p, i in zip(top.values, top.indices)]
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", default="generation", choices=["generation", "structure"])
    parser.add_argument("--adapter", help="기본값: outputs/{agent}_think/final")
    args = parser.parse_args()

    config = load_config(args.agent)
    token = hf_token()
    run_dir = output_dir(args.agent)
    adapter = Path(args.adapter) if args.adapter else run_dir / "final"
    if not (adapter / "adapter_config.json").is_file():
        raise SystemExit(f"[ABORT] 어댑터가 없습니다: {adapter}")

    tokenizer = load_tokenizer(config["model_id"], config["cache_dir"], token)
    records = load_jsonl(sft_path(data_dir(), args.agent))[:PARITY_SAMPLES]
    samples = [(tokenizer(r["prompt"], add_special_tokens=False)["input_ids"],
                tokenizer(r["completion"], add_special_tokens=False)["input_ids"]) for r in records]
    print(f"[PARITY-REF] 학습 샘플 {len(samples)}건: {[r['id'] for r in records]}")

    model = load_base_model(config, token)
    model.eval()
    base_stats = [completion_stats(model, tokenizer, p, c) for p, c in samples]
    base = [s["nll"] for s in base_stats]
    print(f"[PARITY-REF] {'base':>14}: {base}")
    print(f"[PARITY-REF] {'':>14}  게이지: {describe_gauge(base_stats)}")

    model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    tuned_stats = [completion_stats(model, tokenizer, p, c) for p, c in samples]
    tuned = [s["nll"] for s in tuned_stats]
    print(f"[PARITY-REF] {adapter.name:>14}: {tuned}  (감소폭 {[round(b - t, 3) for b, t in zip(base, tuned)]})")
    print(f"[PARITY-REF] {'':>14}  게이지: {describe_gauge(tuned_stats)}")
    if format_learned(tuned_stats):
        print(f"[PARITY-REF] 형식 게이지 OK: 사고 첫 토큰 확률이 {FORMAT_GAUGE_MIN_PROB} 이상")
    else:
        print(f"[PARITY-REF] 형식 게이지 미달: 사고 첫 토큰 확률이 {FORMAT_GAUGE_MIN_PROB} 미만 → 사고 형식을 아직 배우지 못했다 "
              "(2차 학습 어댑터 0.025). 생성 평가 전에 학습량(epoch, lr)을 먼저 다시 본다 (issue/0925/0925_plan.md 5절)")

    path = reference_path(run_dir)
    previous, _ = load_reference(path, records)  # 같은 데이터로 잰 다른 어댑터의 값은 남긴다
    adapters = dict(previous["adapters"]) if previous else {}
    details = dict(previous.get("details", {})) if previous else {}
    adapters[adapter.name] = tuned
    details["base"] = base_stats
    details[adapter.name] = tuned_stats
    save_reference(path, records, base, adapters, details)
    print(f"완료: {path}")


if __name__ == "__main__":
    main()
