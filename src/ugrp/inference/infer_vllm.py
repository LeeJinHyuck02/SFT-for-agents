"""vLLM 추론: bf16 원본 모델에 LoRA를 실행 중에 결합해(병합 없이) 여러 건을 동시에 생성한다.

학습 환경과 분리된 vLLM venv(scripts/setup_vllm.sh → /root/vllm-env)에서 실행한다.
입출력 규약(프롬프트, 멈춤 토큰, 결과 필드)은 infer.py와 같다. 계획: issue/0919/0919_inf.md, 환경 수정: issue/0925/0925.md 부록 A

    python -m ugrp.inference.infer_vllm --agent generation                                   # final 어댑터, test
    python -m ugrp.inference.infer_vllm --no-adapter --greedy                                # 베이스 모델
    python -m ugrp.inference.infer_vllm --adapters final=outputs/generation_think/final \\
        ep1=outputs/generation_think/checkpoint-37 --temperature 0.7              # 어댑터 여러 개를 모델 한 번 올려 비교
    python -m ugrp.inference.infer_vllm --max-num-seqs 4 --max-num-batched-tokens 2048       # KV cache 부족(엔진 초기화 실패) 시
    python -m ugrp.inference.infer_vllm --parity                                             # 4-bit 학습 어댑터가 bf16에서도 효과가 있는지 + 형식 게이지

결과: results/{agent}_think[_base]_{split}_vllm.jsonl (어댑터가 여럿이면 _{이름}을 붙인다)
    infer.py와 같은 {id, target_length, thinking, answer, reference, raw}에 {adapter, backend, tokens, finish_reason, sampling}을 더한다.
    다시 실행하면 이미 있는 id는 건너뛴다.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..common import prompts
from ..common.config import data_dir, hf_token, load_config, output_dir, root_dir
from ..common.data import load_jsonl, sft_path, think_messages_path
from ..common.gemma_format import build_prompt, stop_token_ids
from .inputs import build_inputs, load_items, result_row, split_done
from .parity import (PARITY_SAMPLES, TOP_CANDIDATES, describe_gauge, load_reference, reference_path, summarize_logprobs,
                     think_bounds)

# --parity의 기준값은 코드에 두지 않는다. 데이터와 어댑터에 묶인 값이라 parity_reference.py가 HF 4-bit로 재서
# outputs/{agent}_think/parity_reference.json에 저장한 것을 읽는다 (2026-09-25 2차 학습의 참고값: base 1.87 → final 0.96).
# 판정 (2026-09-25 개정, issue/0925/0925_plan.md 7.2절): 예전의 "감소폭이 HF와 ±0.05" 기준은 4-bit base와 bf16 base의 NLL 차이(약 0.05)가
# 감소폭에 그대로 들어가 허위 FAIL을 냈다 (2차 학습: 어댑터 NLL 차이는 +0.008인데 FAIL). 어댑터를 붙인 NLL의 절대값을 비교한다.
PARITY_ADAPTER_TOLERANCE = 0.05  # 주 기준 (기준값이 있을 때): 어댑터 NLL이 HF 4-bit 값과 이 안에서 같아야 한다
PARITY_BASE_TOLERANCE = 0.1      # 보조 확인: base 절대값의 차이 (4-bit vs bf16 차이를 감안). 넘으면 WARN만
PARITY_MIN_DELTA = 0.3           # 항상 확인: base → 어댑터 감소폭이 이보다 작으면 어댑터가 붙지 않은 것으로 본다
PARITY_BATCHED_TOKENS = 2048   # prompt_logprobs는 위치마다 어휘 262k개 logits를 만든다. 8192면 fp32로 ~8.6GB
CHUNK = 64                     # 이 건수마다 결과를 파일에 기록한다 (중간에 끊겨도 이어서 할 수 있게)
VLLM_LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)


def resolve_model_path(config: dict, token: Optional[str]) -> str:
    """캐시의 snapshot 경로. 없으면 받는다 (pod_setup.sh가 이미 받아 두었으면 네트워크를 쓰지 않는다)."""
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(config["model_id"], cache_dir=config["cache_dir"], local_files_only=True)
    except Exception:
        return snapshot_download(config["model_id"], cache_dir=config["cache_dir"], token=token)


def resolve_adapter(path: str, token: Optional[str]) -> Tuple[str, int]:
    """(로컬 경로, LoRA rank). 로컬 폴더가 아니면 HF Hub repo로 보고 받는다."""
    local = Path(path)
    if not (local / "adapter_config.json").is_file():
        from huggingface_hub import snapshot_download

        local = Path(snapshot_download(path, token=token))
    rank = json.loads((local / "adapter_config.json").read_text(encoding="utf-8"))["r"]
    return str(local.resolve()), rank


def adapter_specs(args, run_dir: Path) -> List[Tuple[str, Optional[str]]]:
    """[(이름, 경로 또는 None)]. None은 베이스 모델."""
    if args.no_adapter:
        return [("base", None)]
    if args.adapters:
        specs = []
        for spec in args.adapters:
            name, sep, path = spec.partition("=")
            if not sep or not name or not path:
                raise SystemExit(f"--adapters는 NAME=PATH 형식이어야 합니다: {spec}")
            specs.append((name, path))
        if len({name for name, _ in specs}) != len(specs):
            raise SystemExit("--adapters의 이름이 중복됩니다.")
        return specs
    if args.adapter:
        return [(Path(args.adapter).name or "adapter", args.adapter)]
    return [("final", str(run_dir / "final"))]


def build_llm(model_path: str, max_rank: Optional[int], args, parity: bool):
    from vllm import LLM

    kwargs = dict(model=model_path, tokenizer=model_path, dtype="bfloat16", max_model_len=args.max_model_len,
                  gpu_memory_utilization=args.gpu_mem, seed=args.seed)
    if max_rank:
        # 어댑터는 generate 호출마다 하나씩 쓰므로 GPU에 동시에 올릴 슬롯은 1개면 된다
        kwargs.update(enable_lora=True, max_loras=1,
                      max_lora_rank=next(r for r in VLLM_LORA_RANKS if r >= max_rank))
    if args.max_num_seqs:
        kwargs["max_num_seqs"] = args.max_num_seqs
    if args.max_num_batched_tokens:
        kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    if parity:  # parity 분기가 뒤에 있으므로 --parity일 때는 위 두 옵션과 관계없이 이 값이 우선한다
        kwargs.update(max_num_batched_tokens=PARITY_BATCHED_TOKENS, max_num_seqs=1)
    # 비전/오디오 인코더용 메모리 확보(프로파일링)를 생략한다. 모델이 지원하지 않는 modality 키를 거부하는 버전이면 image만 둔다
    for limits in ({"image": 0, "video": 0, "audio": 0}, {"image": 0}):
        try:
            return LLM(limit_mm_per_prompt=limits, **kwargs)
        except (ValueError, TypeError) as error:
            if limits == {"image": 0} or not any(m in str(error) for m in ("audio", "video")):
                raise
            print(f"[WARN] limit_mm_per_prompt={limits} 거부됨 ({error}) → image만 지정해 다시 시도합니다.")


def report_capacity(llm) -> None:
    try:
        cache = llm.llm_engine.vllm_config.cache_config
        print(f"[KV] num_gpu_blocks={cache.num_gpu_blocks}, block_size={cache.block_size}")
    except Exception:
        pass
    print("[KV] 동시 처리 건수는 위 vLLM 로그의 'Maximum concurrency for ... tokens per request' 줄을 확인하십시오. "
          "이 값은 최대 길이 요청 기준의 하한이라 2.4x에서도 19건이 11분에 끝났다 (2026-09-25). "
          "엔진 초기화가 KV cache 부족으로 실패하면 --max-num-seqs 4 --max-num-batched-tokens 2048, "
          "그래도 부족하면 --gpu-mem 0.95 → issue/0919/0919_inf.md 대안 2(4-bit)")


def sampling_params(config: dict, args, tokenizer):
    """(SamplingParams, 결과에 기록할 설정). --greedy > --temperature > configs의 generate 순으로 정한다."""
    from vllm import SamplingParams

    gen = config["generate"]
    greedy = args.greedy or not gen.get("do_sample", True)
    record = {"temperature": 0.0 if greedy else (args.temperature if args.temperature is not None else gen["temperature"]),
              "top_p": None if greedy else gen.get("top_p"), "top_k": None if greedy else gen.get("top_k"),
              "repetition_penalty": args.repetition_penalty, "seed": args.seed,
              "max_tokens": args.max_new_tokens or gen["max_new_tokens"]}
    params = dict(temperature=record["temperature"], max_tokens=record["max_tokens"], seed=args.seed,
                  stop_token_ids=stop_token_ids(tokenizer),
                  skip_special_tokens=False, spaces_between_special_tokens=False)
    if not greedy:
        params.update({k: record[k] for k in ("top_p", "top_k") if record[k] is not None})
    if args.repetition_penalty:
        params["repetition_penalty"] = args.repetition_penalty
    return SamplingParams(**params), record


def output_path_for(args, name: str, multiple: bool) -> Path:
    if args.output:
        path = Path(args.output)
        return path.with_name(f"{path.stem}_{name}{path.suffix}") if multiple else path
    stem = f"{args.agent}_think{'_base' if args.no_adapter else ''}_{args.split}_vllm"
    return root_dir() / "results" / (f"{stem}_{name}.jsonl" if multiple else f"{stem}.jsonl")


def run_generation(llm, tokenizer, args, name: str, adapter_path: Optional[str], lora_id: int,
                   output_path: Path, params, record: Dict) -> None:
    from vllm.lora.request import LoRARequest

    items = load_items(args.input or think_messages_path(data_dir(), args.agent, args.split), args.limit)
    structure_map = {r["id"]: r["answer"] for r in load_jsonl(args.structure_from)} if args.structure_from else None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    todo, _ = split_done(items, output_path)
    lora = LoRARequest(name, lora_id, adapter_path) if adapter_path else None
    print(f"== [{name}] {len(todo)}건 → {output_path}")

    entries = []
    for item in todo:
        system, user, length = build_inputs(args.agent, item, args.target_length, structure_map)
        # 토큰 id를 직접 넘겨 vLLM이 BOS를 또 붙이거나 chat template을 다시 적용하지 않게 한다
        ids = tokenizer(build_prompt(tokenizer, system, user), add_special_tokens=False)["input_ids"]
        entries.append((item, length, ids))

    started, total_tokens, truncated = time.time(), 0, 0
    with output_path.open("a", encoding="utf-8", newline="\n") as outfile:
        for start in range(0, len(entries), CHUNK):
            chunk = entries[start:start + CHUNK]
            outputs = llm.generate([{"prompt_token_ids": ids} for _, _, ids in chunk], params, lora_request=lora)
            for (item, length, _), output in zip(chunk, outputs):
                completion = output.outputs[0]
                row = result_row(item, length, completion.text, adapter=name, backend="vllm",
                                 tokens=len(completion.token_ids), finish_reason=completion.finish_reason, sampling=record)
                outfile.write(json.dumps(row, ensure_ascii=False) + "\n")
                total_tokens += row["tokens"]
                truncated += completion.finish_reason == "length"
                print(f"  {item['id']} 사고 {len(row['thinking'])}자 / 답 {len(row['answer'])}자 / "
                      f"{row['tokens']}토큰 ({completion.finish_reason})")
            outfile.flush()

    elapsed = time.time() - started
    if entries:
        print(f"[STATS] [{name}] {len(entries)}건, {elapsed / 60:.1f}분, 생성 {total_tokens}토큰, "
              f"{total_tokens / max(elapsed, 1e-9):.1f} tok/s, 최대 길이에서 잘림(finish_reason=length) {truncated}건")


def completion_logprobs(llm, prompt_ids: List[int], completion_ids: List[int], lora, first: Optional[int]):
    """teacher forcing으로 completion 각 토큰의 logprob과, 사고 첫 내용 토큰(first) 위치의 상위 후보 [(token_id, prob)]."""
    from vllm import SamplingParams

    full = prompt_ids + completion_ids
    params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=TOP_CANDIDATES)
    output = llm.generate([{"prompt_token_ids": full}], params, lora_request=lora, use_tqdm=False)[0]
    # prompt_logprobs[i]는 i번째 토큰의 {token_id: Logprob} (상위 후보 + 실제 토큰). 0번은 None이지만 completion 구간은 항상 1 이상이다
    offset = len(prompt_ids)
    logprobs = [output.prompt_logprobs[offset + i][full[offset + i]].logprob for i in range(len(completion_ids))]
    top = []
    if first is not None:
        candidates = output.prompt_logprobs[offset + first]
        top = sorted(((tid, math.exp(lp.logprob)) for tid, lp in candidates.items()), key=lambda t: -t[1])[:TOP_CANDIDATES]
    return logprobs, top


def run_parity(llm, tokenizer, args, adapters: List[Tuple[str, Optional[str]]]) -> bool:
    from vllm.lora.request import LoRARequest

    records = load_jsonl(sft_path(data_dir(), args.agent))[:PARITY_SAMPLES]
    samples = [(tokenizer(r["prompt"], add_special_tokens=False)["input_ids"],
                tokenizer(r["completion"], add_special_tokens=False)["input_ids"]) for r in records]
    bounds = [think_bounds(tokenizer, completion_ids) for _, completion_ids in samples]
    print(f"[PARITY] 학습 샘플 {len(samples)}건: {[r['id'] for r in records]}")

    reference, reason = load_reference(reference_path(output_dir(args.agent)), records)
    if reference is None:
        print(f"[PARITY] HF 4-bit 기준값을 쓰지 않습니다: {reason}. 감소폭 {PARITY_MIN_DELTA} 이상인지만 봅니다.")
        print("         정밀 비교를 하려면 학습 환경에서: python -m ugrp.inference.parity_reference")
    ref_base = reference["base"] if reference else None
    ref_adapters = reference["adapters"] if reference else {}

    def measure(lora):
        stats = []
        for (prompt_ids, completion_ids), (first, close) in zip(samples, bounds):
            logprobs, top = completion_logprobs(llm, prompt_ids, completion_ids, lora, first)
            summary = summarize_logprobs(logprobs, first, close)
            summary["top"] = [(tokenizer.decode([tid]), round(prob, 4)) for tid, prob in top]
            stats.append(summary)
        return stats

    measured = {"base": measure(None)}
    for lora_id, (name, path) in enumerate(adapters, start=1):
        if path is not None:
            measured[name] = measure(LoRARequest(name, lora_id, path))

    for name, stats in measured.items():
        ref = ref_base if name == "base" else ref_adapters.get(name)
        ref_text = f" | HF 4-bit {ref}" if ref else ""
        print(f"[PARITY] {name:>8}: {[s['nll'] for s in stats]}{ref_text}")
        print(f"[PARITY] {'':>8}  게이지: {describe_gauge(stats)}")

    for i, (vllm_base, hf_base) in enumerate(zip([s["nll"] for s in measured["base"]], ref_base or []), start=1):
        if abs(vllm_base - hf_base) > PARITY_BASE_TOLERANCE:
            print(f"[WARN] 샘플 {i}: base {vllm_base:.3f}가 HF {hf_base}와 {PARITY_BASE_TOLERANCE} 넘게 다릅니다. "
                  f"vLLM이 프롬프트를 다르게 처리하는지(BOS, 특수 토큰) 먼저 확인하십시오.")
    if len(measured) == 1:
        print("[PARITY] 어댑터가 없어 판정하지 않았습니다.")
        return True

    ok = True
    for name in [n for n in measured if n != "base"]:
        for i in range(len(samples)):
            nll = measured[name][i]["nll"]
            delta = measured["base"][i]["nll"] - nll
            attached = delta >= PARITY_MIN_DELTA  # 감소폭이 이보다 작으면 어댑터가 붙지 않은 것
            if name in ref_adapters:
                diff = nll - ref_adapters[name][i]
                passed = attached and abs(diff) <= PARITY_ADAPTER_TOLERANCE
                basis = (f"어댑터 NLL {nll:.3f} vs HF {ref_adapters[name][i]} (차이 {diff:+.3f}, 허용 ±{PARITY_ADAPTER_TOLERANCE}), "
                         f"감소폭 {delta:.3f} (최소 {PARITY_MIN_DELTA})")
            else:
                passed = attached
                basis = f"이 어댑터의 HF 기준값 없음, 감소폭 {delta:.3f} (최소 {PARITY_MIN_DELTA})"
            ok &= passed
            print(f"[PARITY] [{name}] 샘플 {i + 1}: {basis} → {'OK' if passed else 'FAIL'}")
    if not ok:
        print(f"[PARITY] 기준 밖입니다. 감소폭이 {PARITY_MIN_DELTA} 미만이면 어댑터가 제대로 붙지 않은 것 → issue/0919/0919_inf.md 대안 1, "
              f"어댑터 NLL이 HF 값과 {PARITY_ADAPTER_TOLERANCE} 넘게 다르면 대안 2(4-bit)로 같은 검증을 해 보십시오.")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", default="generation", choices=["generation", "structure"])
    parser.add_argument("--adapter", help="기본값: outputs/{agent}_think/final. HF Hub repo 이름도 가능")
    parser.add_argument("--adapters", nargs="+", metavar="NAME=PATH", help="어댑터 여러 개를 모델 한 번 올려 차례로 생성")
    parser.add_argument("--no-adapter", action="store_true", help="베이스 모델만으로 추론")
    parser.add_argument("--split", default="test")
    parser.add_argument("--input", help="기본값: data/SFT_think_en_{split}.jsonl")
    parser.add_argument("--output", help="기본값: results/{agent}_think_{split}_vllm.jsonl")
    parser.add_argument("--structure-from", help="기획 에이전트 추론 결과(jsonl)의 answer를 구조 데이터로 사용")
    parser.add_argument("--target-length", type=int,
                        help="목표 분량(자)을 모든 샘플에 이 값으로 덮어쓴다 (예: 1500 고정 조건 비교). 기본: 입력 파일의 값 "
                             "(평가 파일은 정답 분량의 %d자 단위 반올림값, 원본 필드 입력은 %d)"
                             % (prompts.TARGET_LENGTH_STEP, prompts.DEFAULT_TARGET_LENGTH))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, help="기본값: configs의 generate.max_new_tokens")
    parser.add_argument("--greedy", action="store_true", help="temperature 0")
    parser.add_argument("--temperature", type=float, help="기본값: configs의 generate.temperature")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repetition-penalty", type=float, help="기본: 쓰지 않음. 평가 조건이 바뀌므로 결과의 sampling에 기록된다")
    parser.add_argument("--gpu-mem", type=float, default=0.92, help="gpu_memory_utilization")
    parser.add_argument("--max-model-len", type=int, default=12288,
                        help="프롬프트 최대 ~5.9k(test 5,947) + 생성 6144. 줄이면 긴 샘플이 잘리므로 KV cache 부족은 아래 두 옵션으로")
    parser.add_argument("--max-num-seqs", type=int, help="동시 처리 상한. 메모리가 부족하면 4로")
    parser.add_argument("--max-num-batched-tokens", type=int,
                        help="prefill 청크 크기 (vLLM 기본 8192). KV cache가 부족하면 2048로: 활성화 메모리와 CUDA graph 메모리가 줄어든다")
    parser.add_argument("--parity", action="store_true", help="생성 대신 HF 4-bit 기준값과 completion loss를 비교")
    args = parser.parse_args()
    if args.no_adapter and (args.adapter or args.adapters):
        parser.error("--no-adapter와 --adapter/--adapters는 함께 쓸 수 없습니다.")
    if args.adapter and args.adapters:
        parser.error("--adapter와 --adapters 중 하나만 쓰십시오.")

    config = load_config(args.agent)
    token = hf_token()
    model_path = resolve_model_path(config, token)
    adapters = []
    for name, path in adapter_specs(args, output_dir(args.agent)):
        adapters.append((name, *resolve_adapter(path, token)) if path else (name, None, None))
    max_rank = max((rank for _, _, rank in adapters if rank), default=None)
    print(f"모델: {model_path}")
    for name, path, rank in adapters:
        print(f"어댑터: {name} = {path or '(없음: 베이스 모델)'}" + (f" (r={rank})" if rank else ""))

    llm = build_llm(model_path, max_rank, args, parity=args.parity)
    report_capacity(llm)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if args.parity:
        sys.exit(0 if run_parity(llm, tokenizer, args, [(name, path) for name, path, _ in adapters]) else 1)

    params, record = sampling_params(config, args, tokenizer)
    print(f"[SAMPLING] {record}")
    multiple = len(adapters) > 1
    for lora_id, (name, path, _) in enumerate(adapters, start=1):
        run_generation(llm, tokenizer, args, name, path, lora_id,
                       output_path_for(args, name, multiple), params, record)
    print("추론 완료")


if __name__ == "__main__":
    main()
