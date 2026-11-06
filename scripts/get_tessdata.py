"""Download the Tesseract language models the bet-slip OCR uses (spa + eng, tessdata_fast).

They live in data/tessdata/ (gitignored, ~6 MB) so no admin rights are needed and the
system Tesseract install doesn't have to ship Spanish.

Usage:  python scripts/get_tessdata.py
"""

import logging
from pathlib import Path

import requests

from futbol.config import get_settings
from futbol.logging_setup import setup_logging

URL = "https://github.com/tesseract-ocr/tessdata_fast/raw/main/{lang}.traineddata"
LANGS = ["spa", "eng"]

log = logging.getLogger("get_tessdata")


def main() -> None:
    setup_logging()
    folder = Path(get_settings().tessdata_dir)
    folder.mkdir(parents=True, exist_ok=True)
    for lang in LANGS:
        target = folder / f"{lang}.traineddata"
        if target.exists() and target.stat().st_size > 0:
            log.info("%s already present", target.name)
            continue
        resp = requests.get(URL.format(lang=lang), timeout=120)
        resp.raise_for_status()
        target.write_bytes(resp.content)
        log.info("Downloaded %s (%.1f MB)", target.name, len(resp.content) / 1e6)


if __name__ == "__main__":
    main()
