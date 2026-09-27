"""think mode QLoRA SFT 학습.

    python -m ugrp.training.train --agent generation
    python -m ugrp.training.train --agent generation --max-steps 3 --limit 4   # Pod smoke
    python -m ugrp.training.train --agent generation --tiny-debug              # 로컬 CPU 코드 경로 점검

입력은 UGRP2/data(03 convert)가 만든 SFT_think_en_train_gemma4.jsonl 하나다. valid/test에는 사고 과정 정답이 없어
eval_loss를 계산할 수 없으므로(생성 결과로 평가한다) 기본적으로 평가 없이 학습한다. 과적합이 의심되면
configs의 train_dev_holdout=N으로 train의 마지막 N건을 떼어 eval_loss를 볼 수 있다. TRL에 문자열을 넘기면 BOS/EOS 처리 방식이
버전마다 달라지므로, 여기서 직접 토큰화하여 input_ids + completion_mask로 넘긴다.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from trl import SFTConfig, SFTTrainer

from ..common.config import data_dir, hf_token, load_config, output_dir
from ..common.data import load_jsonl, sft_path
from ..common.gemma_format import stop_token_ids


def load_tokenizer(model_id: str, cache_dir: str, token: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def quantization_config(config: dict) -> BitsAndBytesConfig:
    q = dict(config["quantization"])
    q["bnb_4bit_compute_dtype"] = getattr(torch, q["bnb_4bit_compute_dtype"])
    return BitsAndBytesConfig(**q)


def load_base_model(config: dict, token: str):
    """학습과 추론이 공유하는 4-bit 베이스 로더."""
    return AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        quantization_config=quantization_config(config),
        device_map="auto",
        cache_dir=config["cache_dir"],
        attn_implementation=config["attn_implementation"],
        token=token,
    )


def adapter_generation_config(config: dict, tokenizer, token: str, tiny: bool) -> GenerationConfig:
    """어댑터 폴더에 함께 둘 generation config. 베이스의 원본 파일을 캐시에서 다시 읽고 멈춤 토큰을 보장한다.

    학습 중 트레이너가 메모리 안의 model.generation_config EOS를 토크나이저 값(<eos>)으로 바꿀 수 있으므로
    그 객체는 쓰지 않는다. 이 파일이 있으면 어댑터만 가져가 붙이거나 병합해도 <turn|>에서 멈춘다.
    """
    gen = GenerationConfig() if tiny else GenerationConfig.from_pretrained(
        config["model_id"], cache_dir=config["cache_dir"], token=token)
    eos = gen.eos_token_id
    eos = [] if eos is None else [eos] if isinstance(eos, int) else list(eos)
    gen.eos_token_id = eos + [i for i in stop_token_ids(tokenizer) if i not in eos]
    return gen


def build_tiny_model(config: dict, tokenizer):
    """--tiny-debug용 랜덤 초기화 초소형 모델. 어휘만 실제 토크나이저에 맞춘다."""
    from transformers import LlamaConfig

    tiny = config["tiny"]
    model_config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=tiny["hidden_size"],
        intermediate_size=tiny["intermediate_size"],
        num_hidden_layers=tiny["num_hidden_layers"],
        num_attention_heads=tiny["num_attention_heads"],
        num_key_value_heads=tiny["num_attention_heads"],
        max_position_embeddings=tiny["max_len"],
        pad_token_id=tokenizer.pad_token_id,
    )
    return AutoModelForCausalLM.from_config(model_config)


def tokenize_dataset(tokenizer, records: list, max_len: int, name: str) -> Dataset:
    """prompt/completion을 따로 토큰화해 이어 붙인다. prompt에는 template이 넣은 <bos>가 이미 있다.

    max_len을 넘는 샘플은 자르지 않고 제외한다 (completion이 잘리면 학습 목표가 망가진다).
    """
    rows, dropped = [], []
    for record in records:
        prompt_ids = tokenizer(record["prompt"], add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(record["completion"], add_special_tokens=False)["input_ids"]
        if len(prompt_ids) + len(completion_ids) > max_len:
            dropped.append(record["id"])
            continue
        rows.append({
            "input_ids": prompt_ids + completion_ids,
            "completion_mask": [0] * len(prompt_ids) + [1] * len(completion_ids),
        })
    if dropped:
        print(f"[WARN] {name}: max_len={max_len} 초과로 {len(dropped)}건 제외: {dropped}")
    if not rows:
        raise ValueError(f"{name}: 학습 가능한 샘플이 없습니다.")
    lengths = [len(r["input_ids"]) for r in rows]
    print(f"[INFO] {name}: {len(rows)}건, 토큰 길이 최대 {max(lengths)}")
    first = rows[0]["input_ids"]
    if tokenizer.bos_token_id is not None and first[:2].count(tokenizer.bos_token_id) != 1:
        raise ValueError(f"{name}: 시퀀스 선두의 BOS 개수가 1이 아닙니다: {first[:4]}")
    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", default="generation", choices=["generation", "structure"])
    parser.add_argument("--max-steps", type=int, default=-1, help="smoke용. 지정 시 epoch 설정을 무시")
    parser.add_argument("--limit", type=int, help="smoke용. 앞에서 N건만 사용")
    parser.add_argument("--longest", action="store_true", help="smoke용. --limit과 함께 쓰면 토큰 길이가 가장 긴 N건 선택")
    parser.add_argument("--output-dir", help="기본값: outputs/{agent}_think")
    parser.add_argument("--resume-from-checkpoint", action="store_true")
    parser.add_argument("--push-to-hub", metavar="REPO", help="학습 종료 시 어댑터를 HF private repo로 백업")
    parser.add_argument("--tiny-debug", action="store_true", help="CPU에서 초소형 랜덤 모델로 코드 경로만 검증")
    parser.add_argument("--tokenizer-id", help="기본값: configs의 model_id")
    args = parser.parse_args()

    config = load_config(args.agent)
    token = hf_token()
    tiny = args.tiny_debug
    max_len = config["tiny"]["max_len"] if tiny else config["max_len"]
    out_dir = Path(args.output_dir) if args.output_dir else output_dir(args.agent)
    if tiny and not args.output_dir:
        out_dir = out_dir.with_name(out_dir.name + "_tiny")

    tokenizer = load_tokenizer(args.tokenizer_id or config["model_id"], config["cache_dir"], token)

    base_dir = data_dir()
    train_records = load_jsonl(sft_path(base_dir, args.agent))
    if args.longest:
        train_records.sort(
            key=lambda r: len(r.get("prompt", "")) + len(r.get("completion", "")),
            reverse=True,
        )
        print(f"[INFO] --longest: 가장 긴 {args.limit or 4}건 선택: {[r['id'] for r in train_records[:args.limit or 4]]}")
    if args.limit or tiny:
        train_records = train_records[:args.limit or 4]
    holdout = 0 if (tiny or args.max_steps > 0) else int(config.get("train_dev_holdout", 0))
    dev_records = []
    if holdout > 0:
        # id(연도)순 정렬이므로 마지막 N건 = 가장 최근 기출. 학습에서 빼고 eval_loss 계산에만 쓴다
        train_records, dev_records = train_records[:-holdout], train_records[-holdout:]
        print(f"[INFO] dev holdout {holdout}건: {[r['id'] for r in dev_records]}")
    train_dataset = tokenize_dataset(tokenizer, train_records, max_len, "train")
    eval_dataset = tokenize_dataset(tokenizer, dev_records, max_len, "dev") if dev_records else None

    if tiny:
        model = build_tiny_model(config, tokenizer)
        model.save_pretrained(out_dir / "tiny_base")  # infer.py --tiny-debug가 같은 베이스를 읽는다
        target_modules = config["tiny"]["lora_target_modules"]
    else:
        print("4-bit(NF4)로 양자화하여 베이스 모델을 적재합니다...")
        model = load_base_model(config, token)
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        target_modules = config["lora"]["target_modules"]
    model.config.pad_token_id = tokenizer.pad_token_id

    lora = config["lora"]
    model = get_peft_model(model, LoraConfig(
        r=lora["r"], lora_alpha=lora["lora_alpha"], lora_dropout=lora["lora_dropout"], bias=lora["bias"],
        target_modules=target_modules, task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    train_args = dict(config["train"])
    train_args["eval_strategy"] = "epoch" if eval_dataset is not None else "no"
    if tiny:
        train_args.update(optim="adamw_torch", bf16=False, gradient_checkpointing=False, use_cpu=True,
                          save_strategy="no", gradient_accumulation_steps=1)
    max_steps = config["tiny"]["max_steps"] if tiny and args.max_steps < 0 else args.max_steps
    if max_steps > 0:
        train_args["save_strategy"] = "no"  # smoke: 체크포인트를 남기지 않는다

    sft_config = SFTConfig(
        output_dir=str(out_dir),
        max_length=max_len,
        completion_only_loss=True,
        max_steps=max_steps,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        **train_args,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    print("SFT 훈련을 개시합니다...")
    started = time.time()
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    history = trainer.state.log_history
    step_losses = [(h["epoch"], h["loss"]) for h in history if "loss" in h]
    last_epoch_losses = [loss for epoch, loss in step_losses if epoch > (trainer.state.epoch or 0) - 1]
    metrics = {"train_seconds": round(time.time() - started), "train_samples": len(train_dataset),
               # train_loss: trainer가 기록하는 전체 step 평균. train_loss_last_epoch: 마지막 epoch의 step 평균.
               # (마지막 step 하나의 loss는 샘플 수가 적어 대표성이 없다: 146 = 36 × 4 + 2)
               "train_loss": next((h["train_loss"] for h in reversed(history) if "train_loss" in h), None),
               "train_loss_last_epoch": round(statistics.mean(last_epoch_losses), 4) if last_epoch_losses else None}
    if eval_dataset is not None:
        metrics.update(trainer.evaluate())
    if torch.cuda.is_available():
        metrics["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 1)
        metrics["max_memory_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 1)

    final_dir = out_dir / "final"
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    gen_config = adapter_generation_config(config, tokenizer, token, tiny)
    gen_config.save_pretrained(final_dir)
    print(f"[INFO] generation_config.json 저장 (eos_token_id={gen_config.eos_token_id})")
    trainer.state.save_to_json(str(out_dir / "trainer_state.json"))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"훈련 완료: {final_dir}\n[METRICS] {json.dumps(metrics, ensure_ascii=False)}")

    if args.push_to_hub:
        trainer.model.push_to_hub(args.push_to_hub, private=True, token=token)
        gen_config.push_to_hub(args.push_to_hub, token=token)
        print(f"어댑터를 {args.push_to_hub}에 백업했습니다.")


if __name__ == "__main__":
    main()
