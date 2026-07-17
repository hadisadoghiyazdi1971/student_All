from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# اضافه کردن ریشه پروژه به مسیر import
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from llm_service import OllamaService


def read_letter(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"فایل نامه پیدا نشد: {path}")

    text = path.read_text(encoding="utf-8-sig").strip()

    if not text:
        raise ValueError("فایل نامه خالی است.")

    return text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="تست سریع تحلیل نامه فقط با Ollama و بدون embedding یا cache"
    )
    parser.add_argument(
        "--file",
        required=True,
        help="مسیر فایل TXT حاوی متن نامه",
    )
    parser.add_argument(
        "--model",
        default="qwen3:8b",
        help="نام مدل نصب‌شده در Ollama",
    )
    args = parser.parse_args()

    letter_path = Path(args.file).resolve()
    letter_text = read_letter(letter_path)

    service = OllamaService(
        model=args.model,
        timeout_seconds=300,
    )

    available, message, models = service.status()

    print(message)

    if not available:
        if models:
            print("مدل‌های موجود:")
            for model in models:
                print(f"- {model}")
        raise SystemExit(1)

    print(f"\nتعداد نویسه‌های نامه: {len(letter_text):,}")
    print("تحلیل نامه با Ollama آغاز شد...\n")

    started_at = time.perf_counter()

    # هیچ KeyBERT یا taxonomy به مدل داده نمی‌شود.
    analysis = service.analyze_letter(
        letter_text=letter_text,
        keyword_candidates=[],
        field_matches=[],
    )

    elapsed = time.perf_counter() - started_at

    result = analysis.model_dump()

    output_path = PROJECT_ROOT / "runtime" / "quick_test_result.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nزمان پاسخ مدل: {elapsed:.2f} ثانیه")
    print(f"خروجی ذخیره شد: {output_path}")


if __name__ == "__main__":
    main()