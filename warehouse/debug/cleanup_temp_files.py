"""
Clean up temporary diagnostic and checking scripts
"""
from pathlib import Path
import shutil

# Files to move to warehouse/debug
cleanup_files = [
    'inspect_monthly_files.py',
    'inspect_monthly_structure.py',
    'inspect_monthly_correct.py',
    'check_price_data.py',
    'diagnose_data_gap.py',
    'diagnose_longest_segment.py',
    'evaluate_composite_ic.py',  # v1, keep v2
    'analyze_soft_candidates.py',
    'debug_pmr_zero.py',
    'inspect_gdelt_data.py',
    'inspect_features.py'
]

# Ensure warehouse/debug exists
debug_dir = Path('warehouse/debug')
debug_dir.mkdir(parents=True, exist_ok=True)

moved = 0
for filename in cleanup_files:
    source = Path(filename)
    if source.exists():
        dest = debug_dir / filename
        if dest.exists():
            dest.unlink()
        source.rename(dest)
        print(f"Moved: {filename}")
        moved += 1

print(f"\nTotal files moved: {moved}")

# Files to delete (output logs)
delete_files = [
    'analysis_output.txt',
    'backfill_log.txt',
    'backfill_log_fixed.txt',
    'nul'
]

deleted = 0
for filename in delete_files:
    file_path = Path(filename)
    if file_path.exists():
        file_path.unlink()
        print(f"Deleted: {filename}")
        deleted += 1

print(f"Total files deleted: {deleted}")

print("\nKey scripts retained in root:")
keep_scripts = [
    'rebuild_gdelt_total_from_monthly.py',
    'rebuild_gdelt_and_evaluate.py',
    'evaluate_composite_ic_v2.py',
    'gdelt_gkg_bucket_aggregator.py'
]
for s in keep_scripts:
    if Path(s).exists():
        print(f"  ✓ {s}")
