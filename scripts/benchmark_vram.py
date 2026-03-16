"""Qwen3.5 各モデルサイズの VRAM 使用量と推論速度をベンチマークする."""

import json
import subprocess
import sys

MODELS = [
    "Qwen/Qwen3.5-122B-A10B",
]

BENCH_SCRIPT = r'''
import json, sys, time, torch
from pathlib import Path
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

model_name = sys.argv[1]
image_path = sys.argv[2]

# モデルロード
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForImageTextToText.from_pretrained(
    model_name, dtype=torch.bfloat16, device_map="auto",
)

# 全GPUの合計VRAM計測
num_gpus = torch.cuda.device_count()
vram_after_load = sum(torch.cuda.memory_allocated(i) for i in range(num_gpus)) / 1e9

# 推論
image = Image.open(image_path).convert("RGB")
prompt = "この画像を詳しく説明してください。JSON形式で出力してください。"
messages = [{"role": "user", "content": [
    {"type": "image", "image": image},
    {"type": "text", "text": prompt},
]}]
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

# Warmup
with torch.inference_mode():
    _ = model.generate(**inputs, max_new_tokens=16, do_sample=False)

for i in range(num_gpus):
    torch.cuda.reset_peak_memory_stats(i)

start = time.perf_counter()
with torch.inference_mode():
    output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
elapsed = time.perf_counter() - start

vram_peak = sum(torch.cuda.max_memory_allocated(i) for i in range(num_gpus)) / 1e9
input_len = inputs["input_ids"].shape[1]
gen_tokens = output_ids.shape[1] - input_len

result = {
    "model": model_name,
    "num_gpus": num_gpus,
    "vram_model_gb": round(vram_after_load, 2),
    "vram_peak_gb": round(vram_peak, 2),
    "input_tokens": input_len,
    "generated_tokens": int(gen_tokens),
    "elapsed_seconds": round(elapsed, 2),
    "tokens_per_second": round(gen_tokens / elapsed, 1),
}
print(json.dumps(result))
'''


def run_benchmark(model_name: str, image_path: str) -> dict | None:
    """サブプロセスで1モデルのベンチマークを実行."""
    print(f"\n{'='*60}")
    print(f"ベンチマーク: {model_name}")
    print(f"{'='*60}")

    result = subprocess.run(
        [sys.executable, "-c", BENCH_SCRIPT, model_name, image_path],
        capture_output=True,
        text=True,
        timeout=1800,
    )

    if result.returncode != 0:
        print(f"  エラー: {result.stderr[-500:]}")
        return None

    # 最後の行がJSON結果
    for line in reversed(result.stdout.strip().split("\n")):
        try:
            data = json.loads(line)
            return data
        except json.JSONDecodeError:
            continue
    print(f"  JSON解析失敗")
    print(f"  stdout: {result.stdout[-300:]}")
    return None


def main() -> None:
    image_path = "data/images/sample_home.jpg"

    results = []
    for model_name in MODELS:
        data = run_benchmark(model_name, image_path)
        if data:
            results.append(data)
            print(f"  モデルVRAM: {data['vram_model_gb']:.2f} GB")
            print(f"  ピークVRAM: {data['vram_peak_gb']:.2f} GB")
            print(f"  生成速度:   {data['tokens_per_second']:.1f} toks/s")
            print(f"  生成時間:   {data['elapsed_seconds']:.2f}s ({data['generated_tokens']} tokens)")

    # サマリーテーブル
    print(f"\n{'='*60}")
    print("サマリー")
    print(f"{'='*60}")
    print(f"{'モデル':<28} {'モデルVRAM':>10} {'ピークVRAM':>10} {'速度':>10}")
    print("-" * 60)
    for r in results:
        short = r["model"].split("/")[-1]
        print(
            f"{short:<28} {r['vram_model_gb']:>8.2f}GB "
            f"{r['vram_peak_gb']:>8.2f}GB "
            f"{r['tokens_per_second']:>7.1f}t/s"
        )


if __name__ == "__main__":
    main()
