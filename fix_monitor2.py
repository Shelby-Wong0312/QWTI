path = "warehouse/monitoring/hourly_monitor.py"
with open(path, "r") as f:
    content = f.read()

# Fix 1: cutoff needs timezone
old1 = "cutoff = datetime.now() - timedelta(days=days)"
new1 = "cutoff = datetime.now(timezone.utc) - timedelta(days=days)"

# Fix 2: ensure timezone import exists
import_check = "from datetime import datetime, timedelta, timezone"

changes = 0

if old1 in content:
    content = content.replace(old1, new1)
    changes += 1
    print("FIXED: cutoff now uses timezone.utc")

# Check if timezone is imported
if "from datetime import" in content and "timezone" not in content.split("from datetime import")[1].split("\n")[0]:
    content = content.replace("from datetime import datetime, timedelta", "from datetime import datetime, timedelta, timezone")
    changes += 1
    print("FIXED: added timezone import")

if changes > 0:
    with open(path, "w") as f:
        f.write(content)
    print(f"Total fixes applied: {changes}")
else:
    print("No changes needed or patterns not found")
    # Show relevant lines
    for i, line in enumerate(content.split("\n"), 1):
        if "cutoff" in line or ("from datetime" in line):
            print(f"Line {i}: {line.strip()}")
