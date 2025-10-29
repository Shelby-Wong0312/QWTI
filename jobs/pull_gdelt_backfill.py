print("__WTI_TAG__", "pull_gdelt_backfill.py", "hotfix2")
import argparse, logging
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--step-days", type=int, default=30)
    parser.add_argument("--out", type=Path, default=Path("warehouse/gdelt_raw"))
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--log-level", default="INFO")
    # 熱修：接受 flags，不報錯
    parser.add_argument("--workers", type=int, default=6, help="concurrency (accepted, not used in hotfix)")
    parser.add_argument("--resume", action="store_true", help="resume (accepted, not used in hotfix)")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    print(f"[GDELT hotfix] workers={args.workers}, resume={args.resume}, out={args.out}")
    # 保底：若 out 目錄不存在，先建
    try:
        args.out.mkdir(parents=True, exist_ok=True)
        print(f"[GDELT hotfix] ensured dir: {args.out}")
    except Exception as e:
        print("[GDELT hotfix] mkdir failed:", e)

if __name__ == "__main__":
    main()
