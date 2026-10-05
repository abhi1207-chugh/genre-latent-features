"""Stage 1: check that the project folder structure exists.

Run from the project root:
    python3 experiments/stage01_check_setup.py
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_FOLDERS = [
    "data/raw",
    "data/processed",
    "src/preprocessing",
    "src/models",
    "src/losses",
    "src/training",
    "src/evaluation",
    "src/visualization",
    "experiments",
    "checkpoints",
    "results/figures",
    "notebooks",
    "report",
    "docs",
]

REQUIRED_FILES = [
    "CLAUDE.md",
    "README.md",
    ".gitignore",
    "docs/ML_Project_Final_Writeup.md",
]

missing = []
for folder in REQUIRED_FOLDERS:
    exists = (PROJECT_ROOT / folder).is_dir()
    print(f"[{'OK' if exists else 'MISSING'}] {folder}/")
    if not exists:
        missing.append(folder)

for file in REQUIRED_FILES:
    exists = (PROJECT_ROOT / file).is_file()
    print(f"[{'OK' if exists else 'MISSING'}] {file}")
    if not exists:
        missing.append(file)

if missing:
    print(f"\nSetup INCOMPLETE: {len(missing)} item(s) missing.")
else:
    print("\nSetup complete: all folders and files are present.")
