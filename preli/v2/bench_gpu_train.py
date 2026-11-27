import sys
import time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

V2_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(V2_DIR))
from torch_utils_v4 import (
    CACHE_DIR_HF, HeadConditionedTransformer, compute_multitask_loss, free_cuda
)

def bench_train(model_name, bs=16, max_len=144, freeze_k=6, pool_mode="mean", steps=25):
    free_cuda()
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR_HF, local_files_only=True)
    model = HeadConditionedTransformer(model_name, n_feats=58, pool_mode=pool_mode, num_freeze_layers=freeze_k).cuda()
    opt = torch.optim.AdamW(model.param_groups(lr_backbone=2.5e-5, lr_head=1.5e-4))
    scaler = torch.amp.GradScaler("cuda")
    dummy = [("bKash Helpline: Apnar account unusual login detect hoyeche. OTP share korun." * 2)[:max_len]] * bs
    feats = torch.zeros(bs, 58, device="cuda")
    yl = torch.randint(0, 6, (bs,), device="cuda")
    yt = torch.zeros(bs, 6, device="cuda")
    ya = torch.randint(1, 6, (bs,), device="cuda")
    torch.cuda.reset_peak_memory_stats()

    # warmup
    for i in range(steps + 3):
        if i == 3:
            torch.cuda.synchronize()
            t0 = time.time()
        enc = tok(dummy, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        ids = enc["input_ids"].cuda()
        mask = enc["attention_mask"].cuda()
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            ll, lt, la = model(ids, mask, feats)
            loss = compute_multitask_loss(ll, lt, la, yl, yt, ya)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    rps = (steps * bs) / dt
    print(f"TRAIN {model_name} (freeze={freeze_k}, bs={bs}, maxL={max_len}): {rps:.1f} r/s | peak_VRAM={peak:.2f} GB", flush=True)
    del model, opt, scaler
    free_cuda()

def bench_qwen(steps=10):
    free_cuda()
    mname = "Qwen/Qwen2.5-1.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(mname, cache_dir=CACHE_DIR_HF, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(mname, cache_dir=CACHE_DIR_HF, local_files_only=True, dtype=torch.float16).cuda().eval()
    torch.cuda.reset_peak_memory_stats()
    msgs = [{"role": "user", "content": "Classify SMS into JSON {label, action}: Offer letter attached. Joining date 1 Jan. - HR, Daraz"}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    t0 = time.time()
    for _ in range(steps):
        inp = tok(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            _ = model.generate(**inp, max_new_tokens=20, do_sample=False, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"QWEN 1.5B FP16 KV-cache (max_new=20): {steps/dt:.2f} r/s | peak_VRAM={peak:.2f} GB", flush=True)
    del model
    free_cuda()

if __name__ == "__main__":
    bench_train("MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", bs=16, max_len=144, freeze_k=6, pool_mode="mean")
    bench_train("intfloat/multilingual-e5-base", bs=24, max_len=144, freeze_k=4, pool_mode="mean")
    bench_qwen(steps=8)
