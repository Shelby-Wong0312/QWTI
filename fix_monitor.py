import re

path = "warehouse/monitoring/hourly_monitor.py"
with open(path, "r") as f:
    content = f.read()

old = "df_metrics['timestamp'] = pd.to_datetime(df_metrics['timestamp'])"
new = "df_metrics['timestamp'] = pd.to_datetime(df_metrics['timestamp'], format='mixed', utc=True)"

if old in content:
    content = content.replace(old, new)
    with open(path, "w") as f:
        f.write(content)
    print("FIXED: timestamp parsing updated")
else:
    print("Pattern not found, showing existing lines:")
    for i, line in enumerate(content.split("\n"), 1):
        if "to_datetime" in line and "timestamp" in line:
            print(f"Line {i}: {line.strip()}")
