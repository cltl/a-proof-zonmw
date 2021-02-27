import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

with open("src/config.json", 'r', encoding='utf8') as f:
    CONFIG = json.load(f)

PATHS = CONFIG['paths']
