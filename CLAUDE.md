# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Article scraper and converter. Downloads articles from Substack (via sitemap) or SevanNisanyan.com (via API) and converts HTML to clean Markdown.

## Commands

```bash
# Full pipeline - Substack (default)
python3 src/main.py                        # Download batch of 10, convert

# SevanNisanyan.com source
python3 src/main.py --source sevan         # Download from sevannisanyan.com
python3 src/main.py --source sevan --keywords "Pazar Sohbeti"  # With filter

# Custom batch size
python3 src/main.py --batch-size 20

# Delete HTML files after conversion (saves storage)
python3 src/main.py --delete-after

# Download only / Convert only
python3 src/main.py --skip-convert
python3 src/main.py --skip-download

# Install dependencies
pip install -r requirements.txt
```

## Architecture

**Pipeline:**
- Substack: `sitemap.xml` → `download_articles.py` → HTML → `convert_to_md.py` → Markdown
- SevanNisanyan.com: `API (__data.json)` → `download_articles.py` → HTML → `convert_to_md.py` → Markdown

- `src/main.py` - Orchestrates batch loop: download N files → convert → delete (optional) → repeat until done
- `src/download_articles.py` - Downloads articles from Substack (sitemap) or SevanNisanyan.com (API)
- `src/convert_to_md.py` - Extracts title/subtitle/date/content, converts to Markdown
