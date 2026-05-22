#!/usr/bin/env python3
"""
decode_bench.py - measure ollama decode rate and place it on the roofline.

For a single decode sequence, operational intensity is approximately:
    intensity = bytes_compute / bytes_loaded
              ~ 1 FLOP/byte  (FP32 weights)
              ~ bits_per_weight / 8  FLOP/byte  (quantized)

So decode speed is bottlenecked by memory bandwidth:
    tokens/sec <= peak_BW / model_bytes_per_token

We measure actual decode rate and compute what batch size would be
needed to cross the ridge into compute-bound territory.
"""
import json
import os
import re
import subprocess
import sys
import urllib.request

OLLAMA_URL = "http://localhost:11434"

# hardware constants from sweep.cu measurements
PEAK_BW_GBS    = 228.0   # measured, GB/s
PEAK_FLOPS_TF  = 4.28    # measured, TFLOPS (FP32, at boost)
RIDGE_FLOP_PER_BYTE = PEAK_FLOPS_TF * 1e12 / (PEAK_BW_GBS * 1e9)


def ollama_generate(model, prompt, num_predict=150):
    """Single generate call; returns the full response dict."""
    body = json.dumps({
        "model":   model,
        "prompt":  prompt,
        "stream":  False,
        "options": {"num_predict": num_predict},
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def model_info(model):
    """Return model metadata and blob size in bytes."""
    body = json.dumps({"model": model}).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/show",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())
    details   = data.get("details", {})
    param_str = details.get("parameter_size", "")
    quant     = details.get("quantization_level", "unknown")
    # extract blob path from modelfile FROM line and stat it
    modelfile  = data.get("modelfile", "")
    blob_match = re.search(r"^FROM\s+(\S+)", modelfile, re.MULTILINE)
    blob_bytes = None
    if blob_match:
        try:
            blob_bytes = os.stat(blob_match.group(1)).st_size
        except OSError:
            pass
    if blob_bytes is None:
        # fall back: parse `ollama list` for this model's size
        try:
            out = subprocess.check_output(["ollama", "list"],
                                          text=True, stderr=subprocess.DEVNULL)
            for line in out.splitlines():
                if model.split(":")[0] in line and model.split(":")[-1] in line:
                    # format: NAME  ID  SIZE  MODIFIED
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if p in ("GB", "MB") and i > 0:
                            scale = 1e9 if p == "GB" else 1e6
                            blob_bytes = int(float(parts[i-1]) * scale)
                            break
        except Exception:
            pass
    return param_str, quant, data, blob_bytes


def bench_model(model):
    prompt = (
        "Explain the difference between memory bandwidth and compute "
        "throughput in three sentences."
    )

    print(f"\nModel: {model}")
    param_str, quant, info, blob_bytes = model_info(model)
    print(f"  params: {param_str}   quantization: {quant}")

    # warmup
    print("  warming up...", end=" ", flush=True)
    ollama_generate(model, prompt, num_predict=20)
    print("done")

    # measured run
    print("  measuring...", end=" ", flush=True)
    r = ollama_generate(model, prompt, num_predict=150)
    print("done")

    decode_tokens = r.get("eval_count", 0)
    decode_ns     = r.get("eval_duration", 1)
    prefill_tok   = r.get("prompt_eval_count", 0)
    prefill_ns    = r.get("prompt_eval_duration", 1)

    decode_tps  = decode_tokens  / decode_ns  * 1e9
    prefill_tps = prefill_tok    / prefill_ns * 1e9

    print(f"\n  decode:  {decode_tokens} tokens in {decode_ns/1e9:.2f}s "
          f"= {decode_tps:.1f} tok/s")
    print(f"  prefill: {prefill_tok} tokens in {prefill_ns/1e9:.2f}s "
          f"= {prefill_tps:.1f} tok/s")

    # roofline analysis
    model_bytes = blob_bytes
    if model_bytes:
        bw_ceiling = PEAK_BW_GBS * 1e9 / model_bytes
        efficiency = decode_tps / bw_ceiling
        print(f"\n  model size on disk:  {model_bytes/1e9:.2f} GB")
        print(f"  BW-limited ceiling:  {bw_ceiling:.1f} tok/s  "
              f"(= {PEAK_BW_GBS:.0f} GB/s / {model_bytes/1e9:.2f} GB)")
        print(f"  achieved / ceiling:  {efficiency*100:.0f}%")

        # batch size needed to reach the ridge
        # intensity at batch B ~ B * flops_per_token / bytes_per_token
        # flops_per_token ~ 2 * params (one multiply + one add per weight)
        # bytes_per_token ~ model_bytes (load all weights)
        # ridge when B * 2 * params / model_bytes = RIDGE_FLOP_PER_BYTE
        # B_ridge = RIDGE_FLOP_PER_BYTE * model_bytes / (2 * params)
        # for quantized: flops still ~2*params (dequant to fp16 for matmul)
        # but bytes = model_bytes (quantized)
        param_str2 = param_str
        try:
            scale = 1e9 if param_str2.endswith("B") else 1e6
            params = float(param_str2[:-1]) * scale
            b_ridge = RIDGE_FLOP_PER_BYTE * model_bytes / (2 * params)
            print(f"\n  params:              {params/1e9:.1f}B")
            print(f"  ridge:               {RIDGE_FLOP_PER_BYTE:.1f} FLOP/byte")
            print(f"  batch size to reach ridge: ~{b_ridge:.0f}")
            print(f"  (below that, adding more GPU FLOPS doesn't help)")
        except (ValueError, ZeroDivisionError):
            pass
    else:
        print("  (model size not available from API)")


def main():
    models = sys.argv[1:] or ["qwen2.5:3b-instruct"]
    print(f"Hardware: peak BW {PEAK_BW_GBS} GB/s  "
          f"peak FP32 {PEAK_FLOPS_TF} TFLOPS  "
          f"ridge {RIDGE_FLOP_PER_BYTE:.1f} FLOP/byte")
    for m in models:
        bench_model(m)


if __name__ == "__main__":
    main()
