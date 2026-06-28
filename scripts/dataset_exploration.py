import json
from collections import Counter
from datetime import date

TODAY = date.today()

titles = []
countries = []
open_to_work = 0
inactive_count = 0

with open("data/candidates.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        c = json.loads(line)

        p = c["profile"]
        s = c["redrob_signals"]

        titles.append(p["current_title"])
        countries.append(p.get("country", ""))

        if s.get("open_to_work_flag"):
            open_to_work += 1

        last = date.fromisoformat(
            s.get("last_active_date", "2020-01-01")
        )

        if (TODAY - last).days > 180:
            inactive_count += 1

N = len(titles)

print(f"Total: {N}")
print(f"Open to work: {open_to_work / N * 100:.1f}%")
print(f"Inactive >6 months: {inactive_count / N * 100:.1f}%")

print(Counter(titles))

print("\nTop 10 current titles:")

for title, count in Counter(titles).most_common(10):
    print(f"  {count:5d}  {title}")