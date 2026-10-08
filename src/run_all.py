"""
Chạy tất cả các bước lab theo thứ tự hoặc chỉ một bước cụ thể.

Cách dùng:
    python run_all.py            # chạy tất cả 4 bước
    python run_all.py --step 1   # chỉ chạy Bước 1
    python run_all.py --step 3   # chỉ chạy Bước 3 (RAGAS ~15-30 phút)
"""
import sys
import argparse
import importlib
from pathlib import Path
from collections.abc import Sequence

sys.path.insert(0, str(Path(__file__).parent))


STEPS = {
    1: ("Bước 1: LangSmith RAG Pipeline",   "01_langsmith_rag_pipeline"),
    2: ("Bước 2: Prompt Hub & A/B Routing",  "02_prompt_hub_ab_routing"),
    3: ("Bước 3: RAGAS Evaluation",          "03_ragas_evaluation"),
    4: ("Bước 4: Guardrails AI Validators",  "04_guardrails_validator"),
}


def run_step(step_num: int) -> bool:
    """Import and run a lab step, converting exceptions into an explicit result."""
    title, module_name = STEPS[step_num]
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    try:
        module = importlib.import_module(module_name)
        module.main()
        print(f"\n✅ {title} — HOÀN THÀNH")
        return True
    except SystemExit as e:
        if e.code not in (None, 0):
            print(f"\n❌ {title} — DỪNG (config thiếu hoặc lỗi)")
        return e.code in (None, 0)
    except Exception as e:
        print(f"\n❌ {title} — LỖI: {type(e).__name__}")
        return False


def main(argv: Sequence[str] | None = None) -> int:
    """Return 0 only when every requested step succeeds; stop at the first failure."""
    parser = argparse.ArgumentParser(
        description="Chạy Day22 Lab: LangSmith + Prompt Versioning + RAGAS + Guardrails"
    )
    parser.add_argument(
        "--step", type=int, choices=[1, 2, 3, 4],
        help="Chỉ chạy bước được chỉ định (1-4)"
    )
    args = parser.parse_args(argv)

    steps_to_run = [args.step] if args.step else list(STEPS.keys())

    results = {}
    for step_num in steps_to_run:
        success = run_step(step_num)
        results[step_num] = success
        if not success and not args.step:
            print(f"\n⛔ Dừng lại do Bước {step_num} thất bại.")
            break

    # Tổng kết
    print(f"\n{'=' * 60}")
    print("  Tổng kết")
    print(f"{'=' * 60}")
    for step_num, success in results.items():
        title = STEPS[step_num][0]
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}  {title}")

    return 0 if len(results) == len(steps_to_run) and all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
