from pathlib import Path
from rapidfuzz import fuzz

dir_sevan = Path("formatted/sevan")
dir_substack = Path("formatted/substack")

files_sevan = list(dir_sevan.glob("*.md"))
files_substack = list(dir_substack.glob("*.md"))

THRESHOLD = 69.1 # Good threshold for our database
DRY_RUN = False # ← change to False to actually delete

to_delete = []

for a in files_sevan:
    for b in files_substack:
        score = fuzz.token_set_ratio(a.stem.lower(), b.stem.lower())
        if score > THRESHOLD:
            to_delete.append((score, a, b))
            break  # stop after first duplicate match

# Sort by similarity (highest first)
to_delete.sort(reverse=True, key=lambda x: x[0])

print(f"\nFound {len(to_delete)} duplicates to remove from 'sevan':\n")

for score, a, b in to_delete:
    print(f"{score:5.1f}%  DELETE → {a.name}   (matches {b.name})")

# Perform deletion if not dry run
if not DRY_RUN:
    for _, a, _ in to_delete:
        a.unlink()
    print("\nDeletion complete.")
else:
    print("\nDRY RUN ONLY — no files deleted.")
    print("Set DRY_RUN = False to actually remove them.")
