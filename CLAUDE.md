# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Substack article scraper and converter. Downloads articles from a sitemap and converts HTML to clean Markdown.

## Commands

```bash
# Full pipeline (download + convert)
python3 src/main.py

# Limit to N articles
python3 src/main.py -n 33

# Download only / Convert only
python3 src/main.py --skip-convert
python3 src/main.py --skip-download

# Install dependencies
pip install -r requirements.txt
```

## Architecture

**Pipeline:** `sitemap.xml` → `download_substack.py` → HTML → `convert_to_md.py` → Markdown

- `src/main.py` - Orchestrates pipeline with CLI args
- `src/download_substack.py` - Parses sitemap, downloads articles to `./sources/substack/`
- `src/convert_to_md.py` - Extracts title/subtitle/date/content, converts to Markdown in `./formatted/substack/`
