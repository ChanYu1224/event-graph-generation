"""Vision Understanding: 画像からイベントを構造化抽出するスクリプト (transformers版)."""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}

EVENT_EXTRACTION_PROMPT = """\
あなたは画像からイベント（出来事・行動）を抽出する専門家です。
この画像を注意深く観察し、描写されているイベントを以下のJSON形式で出力してください。
JSON以外は出力しないでください。

{
  "events": [
    {
      "event_id": 1,
      "event_type": "行動/状態/関係/変化",
      "description_ja": "日本語での説明",
      "description_en": "English description",
      "participants": ["人物・物体"],
      "attributes": {
        "location": "場所",
        "time_context": "時間的文脈",
        "emotion": "感情・雰囲気"
      },
      "confidence": 0.95
    }
  ],
  "scene_description": "シーン全体の説明",
  "object_count": 3
}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="画像からイベントを構造化抽出する (transformers版)"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/images/"),
        help="画像フォルダ (default: data/images/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/results.jsonl"),
        help="出力JSONL (default: output/results.jsonl)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3.5-9B",
        help="モデル名 (default: Qwen/Qwen3.5-9B)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="最大生成トークン数 (default: 2048)",
    )
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="思考モードを有効化 (default: False)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="処理済み画像をスキップして再開",
    )
    return parser.parse_args()


def discover_images(input_dir: Path) -> list[Path]:
    """対象拡張子の画像ファイルを探索し、ソートして返す."""
    return [
        p
        for p in sorted(input_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def load_processed_filenames(output_path: Path) -> set[str]:
    """既存のJSONLから処理済みファイル名を読み込む."""
    processed = set()
    if not output_path.exists():
        return processed
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                processed.add(record["image_filename"])
            except (json.JSONDecodeError, KeyError):
                continue
    return processed


def parse_model_output(raw_output: str) -> dict | None:
    """モデル出力からJSONをパースする."""
    text = raw_output.strip()
    # </think> ブロックを除去（Qwen3.5の思考出力対応）
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    # ```json ... ``` ブロックを除去
    if "```json" in text:
        text = text.split("```json", 1)[1]
        if "```" in text:
            text = text.split("```", 1)[0]
        text = text.strip()
    elif text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def process_image(
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
    image_path: Path,
    model_name: str,
    max_new_tokens: int,
    thinking: bool,
) -> dict:
    """1枚の画像を処理し、結果辞書を返す."""
    start_time = time.monotonic()
    result = {
        "image_path": str(image_path),
        "image_filename": image_path.name,
        "model": model_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        result.update(
            status="error",
            error_message=f"画像読み込みエラー: {e}",
            processing_time_seconds=round(time.monotonic() - start_time, 3),
        )
        return result

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": EVENT_EXTRACTION_PROMPT},
            ],
        }
    ]

    try:
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        ).to(model.device)
        inputs.pop("token_type_ids", None)

        gen_start = time.monotonic()
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=True,
            )

        # 入力トークンを除去して生成部分のみデコード
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, input_len:]
        raw_output = processor.decode(generated_ids, skip_special_tokens=True)
        gen_elapsed = time.monotonic() - gen_start
        num_tokens = len(generated_ids)
        print(
            f"  生成: {num_tokens} tokens, {gen_elapsed:.1f}s, "
            f"{num_tokens / gen_elapsed:.1f} toks/s"
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
        result.update(
            status="error",
            error_message=f"推論エラー: {e}",
            processing_time_seconds=round(time.monotonic() - start_time, 3),
        )
        return result
    except Exception as e:
        result.update(
            status="error",
            error_message=f"推論エラー: {e}",
            processing_time_seconds=round(time.monotonic() - start_time, 3),
        )
        return result

    result["raw_output"] = raw_output
    parsed = parse_model_output(raw_output)

    if parsed is not None:
        result["status"] = "success"
        result["events"] = parsed.get("events", [])
        result["scene_description"] = parsed.get("scene_description", "")
        result["object_count"] = parsed.get("object_count", 0)
    else:
        result["status"] = "parse_error"
        result["events"] = []
        result["scene_description"] = ""
        result["object_count"] = 0

    result["processing_time_seconds"] = round(time.monotonic() - start_time, 3)
    return result


def main() -> None:
    args = parse_args()

    # 画像探索
    if not args.input_dir.exists():
        print(f"エラー: 入力ディレクトリが存在しません: {args.input_dir}")
        raise SystemExit(1)

    images = discover_images(args.input_dir)
    if not images:
        print(f"エラー: 画像が見つかりません: {args.input_dir}")
        raise SystemExit(1)

    print(f"発見した画像数: {len(images)}")

    # resume処理
    if args.resume:
        processed = load_processed_filenames(args.output)
        images = [img for img in images if img.name not in processed]
        print(f"スキップ済み: {len(processed)}, 残り: {len(images)}")
        if not images:
            print("すべての画像が処理済みです")
            return

    # 出力ディレクトリ作成
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # モデルロード
    print(f"モデルをロード中: {args.model}")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )

    # 処理ループ
    success_count = 0
    error_count = 0
    total_start = time.monotonic()

    with open(args.output, "a", encoding="utf-8") as f:
        for image_path in tqdm(images, desc="処理中"):
            result = process_image(
                model, processor, image_path, args.model,
                args.max_new_tokens, args.thinking,
            )

            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

            if result["status"] == "success":
                success_count += 1
            else:
                error_count += 1

    total_time = round(time.monotonic() - total_start, 1)
    print(f"\n=== サマリー ===")
    print(f"成功: {success_count}")
    print(f"失敗: {error_count}")
    print(f"合計処理時間: {total_time}秒")


if __name__ == "__main__":
    main()
