"""
download_articles.py - Download articles from Substack or SevanNisanyan.com

Usage:
    python3 download_articles.py [--source substack|sevan] [OPTIONS]

Substack mode (default):
    python3 download_articles.py --sitemap sitemap.xml --output DIR

SevanNisanyan.com mode:
    python3 download_articles.py --source sevan --output DIR

This script downloads articles and saves them as raw HTML files.
"""

import requests
import time
import os
import argparse
import xml.etree.ElementTree as ET
import json

# Default path for processed URLs file
DEFAULT_PROCESSED_URLS_FILE = 'processed_urls.txt'

# SevanNisanyan.com API configuration
SEVAN_BASE_URL = 'https://www.sevannisanyan.com'
SEVAN_API_ENDPOINT = '/__data.json'
SEVAN_PAGE_SIZE = 20


def load_processed_urls(filepath=DEFAULT_PROCESSED_URLS_FILE):
    """
    Load set of already-processed URLs from file.

    Args:
        filepath: Path to processed URLs file

    Returns:
        Set of URLs that have been processed
    """
    if not os.path.exists(filepath):
        return set()

    with open(filepath, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())


def add_processed_url(url, filepath=DEFAULT_PROCESSED_URLS_FILE):
    """
    Append a URL to the processed URLs file.

    Args:
        url: URL to mark as processed
        filepath: Path to processed URLs file
    """
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(url + '\n')


# =============================================================================
# Substack Functions (sitemap-based)
# =============================================================================

def parse_sitemap(sitemap_path):
    """
    Parse sitemap.xml and extract article URLs.

    Args:
        sitemap_path: Path to sitemap.xml file

    Returns:
        List of article URLs
    """
    tree = ET.parse(sitemap_path)
    root = tree.getroot()

    namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
    urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]

    # Filter to only post URLs (exclude homepage)
    post_urls = [url for url in urls if url.count('/') > 3]

    return post_urls


def get_filepath_for_url(url, output_dir):
    """
    Get the expected filepath for an article URL.

    Args:
        url: Article URL
        output_dir: Directory where HTML files are saved

    Returns:
        Expected filepath for the article
    """
    slug = url.rstrip('/').split('/')[-1]
    filename = f"{slug}.html"
    return os.path.join(output_dir, filename)


def download_article(url, output_dir, delay=1.0, skip_existing=False):
    """
    Download a single article and save as HTML.

    Args:
        url: Article URL
        output_dir: Directory to save HTML files
        delay: Delay in seconds between requests (be polite to server)
        skip_existing: If True, skip download if file already exists

    Returns:
        Tuple of (status: str, filepath: str or None, error: str or None)
        status can be: 'downloaded', 'skipped', 'failed'
    """
    try:
        filepath = get_filepath_for_url(url, output_dir)

        # Check if file already exists
        if skip_existing and os.path.exists(filepath):
            return 'skipped', filepath, None

        # Download
        response = requests.get(url)
        response.raise_for_status()

        # Save
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(response.text)

        # Be polite to server
        time.sleep(delay)

        return 'downloaded', filepath, None

    except Exception as e:
        return 'failed', None, str(e)


def download_articles_substack(sitemap_path, output_dir='./sources/substack', limit=None, delay=1.0,
                               batch_size=None, skip_existing=False, processed_urls_file=None):
    """
    Download articles from Substack sitemap.

    Args:
        sitemap_path: Path to sitemap.xml
        output_dir: Directory to save HTML files (default: ./sources/substack)
        limit: Maximum number of articles to process (None for all)
        delay: Delay between requests in seconds
        batch_size: If set, stop after downloading this many NEW files (skipped files don't count)
        skip_existing: If True, skip files that already exist (checks HTML file)
        processed_urls_file: If set, skip URLs listed in this file (checks processed URLs list)

    Returns:
        Dictionary with download statistics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load processed URLs if file is specified
    processed_urls = set()
    if processed_urls_file:
        processed_urls = load_processed_urls(processed_urls_file)
        if processed_urls:
            print(f"[PROCESSED] Loaded {len(processed_urls)} already-processed URLs")

    # Parse sitemap
    print(f"[SITEMAP] Reading: {sitemap_path}")
    post_urls = parse_sitemap(sitemap_path)
    total_in_sitemap = len(post_urls)
    print(f"[SITEMAP] Found {total_in_sitemap} articles in sitemap")

    # Apply limit
    if limit:
        post_urls = post_urls[:limit]
        print(f"[LIMIT] Processing first {limit} articles")

    # Download articles
    downloaded = 0
    skipped = 0
    failed = 0
    processed = 0
    downloaded_files = []
    downloaded_urls = []

    for url in post_urls:
        # Check if we've reached batch_size for new downloads
        if batch_size and downloaded >= batch_size:
            print(f"\n[BATCH] Reached batch size of {batch_size} new downloads, stopping")
            break

        # Check if URL is already processed (converted to MD)
        if url in processed_urls:
            print(f"[SKIP] Already processed: {url}")
            skipped += 1
            processed += 1
            continue

        processed += 1
        status, filepath, error = download_article(url, output_dir, delay, skip_existing)

        if status == 'downloaded':
            print(f"[DOWNLOAD] {filepath}")
            downloaded += 1
            downloaded_files.append(filepath)
            downloaded_urls.append(url)
        elif status == 'skipped':
            print(f"[SKIP] HTML exists: {filepath}")
            skipped += 1
        else:
            print(f"[FAILED] {url} - {error}")
            failed += 1

    return {
        'downloaded_count': downloaded,
        'skipped_count': skipped,
        'failed_count': failed,
        'total_processed': processed,
        'downloaded_files': downloaded_files,
        'downloaded_urls': downloaded_urls
    }


# =============================================================================
# SevanNisanyan.com Functions (API-based)
# =============================================================================

def parse_sevan_json_response(data):
    """
    Parse the SvelteKit JSON response format from sevannisanyan.com.

    The response uses a node reference system where values reference other
    positions in the data array.

    Args:
        data: The raw JSON data from __data.json

    Returns:
        Tuple of (articles: list, total: int, has_more: bool)
    """
    try:
        # Navigate to the data nodes
        nodes = data.get('nodes', [])
        if len(nodes) < 3:
            return [], 0, False

        # The main data is in nodes[2]
        main_node = nodes[2]
        if not main_node or main_node.get('type') != 'data':
            return [], 0, False

        node_data = main_node.get('data', [])
        if len(node_data) < 3:
            return [], 0, False

        # node_data[0] contains metadata indices
        # node_data[1] contains entries metadata indices
        # node_data[2] is an array of article reference indices

        meta_indices = node_data[1]  # {'entries': X, 'total': Y, 'hasMore': Z}

        total = node_data[meta_indices.get('total', 0)] if isinstance(meta_indices, dict) else 0
        has_more = node_data[meta_indices.get('hasMore', 0)] if isinstance(meta_indices, dict) else False

        # Get article references
        article_refs = node_data[2] if len(node_data) > 2 else []
        if not isinstance(article_refs, list):
            return [], total, has_more

        articles = []
        for ref_idx in article_refs:
            if not isinstance(ref_idx, int) or ref_idx >= len(node_data):
                continue

            article_meta = node_data[ref_idx]
            if not isinstance(article_meta, dict):
                continue

            # Extract article fields using their reference indices
            article = {}
            for field, field_idx in article_meta.items():
                if isinstance(field_idx, int) and field_idx < len(node_data):
                    article[field] = node_data[field_idx]
                else:
                    article[field] = field_idx

            # Only include if we have a slug
            if article.get('slug'):
                articles.append(article)

        return articles, total, has_more

    except Exception as e:
        print(f"[ERROR] Failed to parse JSON response: {e}")
        return [], 0, False


def fetch_sevan_articles_page(offset=0, keywords=None, order='desc'):
    """
    Fetch a page of articles from sevannisanyan.com API.

    Args:
        offset: Pagination offset
        keywords: Optional keywords filter (list of strings)
        order: Sort order ('asc' or 'desc')

    Returns:
        Tuple of (articles: list, total: int, has_more: bool)
    """
    params = {'offset': offset, 'o': order}

    if keywords:
        # Keywords are passed as JSON-encoded array
        params['k'] = json.dumps(keywords)

    url = f"{SEVAN_BASE_URL}{SEVAN_API_ENDPOINT}"

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return parse_sevan_json_response(data)
    except Exception as e:
        print(f"[ERROR] Failed to fetch from API: {e}")
        return [], 0, False


def fetch_all_sevan_article_urls(keywords=None, order='desc', delay=0.5):
    """
    Fetch all article URLs from sevannisanyan.com.

    Args:
        keywords: Optional keywords filter
        order: Sort order
        delay: Delay between API requests

    Returns:
        List of (url, slug) tuples
    """
    all_articles = []
    offset = 0
    total = None

    while True:
        print(f"[API] Fetching articles at offset {offset}...")
        articles, fetched_total, has_more = fetch_sevan_articles_page(offset, keywords, order)

        if total is None:
            total = fetched_total
            print(f"[API] Total articles available: {total}")

        if not articles:
            break

        for article in articles:
            slug = article.get('slug', '')
            if slug:
                url = f"{SEVAN_BASE_URL}/metin/{slug}"
                all_articles.append((url, slug))

        if not has_more:
            break

        offset += SEVAN_PAGE_SIZE
        time.sleep(delay)

    return all_articles


def download_sevan_article(url, slug, output_dir, delay=1.0, skip_existing=False):
    """
    Download a single article from sevannisanyan.com.

    Args:
        url: Article URL
        slug: Article slug (used for filename)
        output_dir: Directory to save HTML files
        delay: Delay after download
        skip_existing: If True, skip if file exists

    Returns:
        Tuple of (status, filepath, error)
    """
    try:
        filepath = os.path.join(output_dir, f"{slug}.html")

        if skip_existing and os.path.exists(filepath):
            return 'skipped', filepath, None

        response = requests.get(url)
        response.raise_for_status()

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(response.text)

        time.sleep(delay)
        return 'downloaded', filepath, None

    except Exception as e:
        return 'failed', None, str(e)


def download_articles_sevan(output_dir='./sources/sevan', limit=None, delay=1.0,
                            batch_size=None, skip_existing=False, processed_urls_file=None,
                            keywords=None, order='desc'):
    """
    Download articles from sevannisanyan.com.

    Args:
        output_dir: Directory to save HTML files
        limit: Maximum number of articles to process
        delay: Delay between requests
        batch_size: Stop after downloading this many NEW files
        skip_existing: Skip files that already exist
        processed_urls_file: File containing already-processed URLs
        keywords: Keywords filter (list of strings)
        order: Sort order

    Returns:
        Dictionary with download statistics
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load processed URLs
    processed_urls = set()
    if processed_urls_file:
        processed_urls = load_processed_urls(processed_urls_file)
        if processed_urls:
            print(f"[PROCESSED] Loaded {len(processed_urls)} already-processed URLs")

    # Fetch all article URLs
    print(f"[SEVAN] Fetching article list from API...")
    if keywords:
        print(f"[SEVAN] Keywords filter: {keywords}")

    article_list = fetch_all_sevan_article_urls(keywords, order, delay=0.5)
    total_available = len(article_list)
    print(f"[SEVAN] Found {total_available} articles")

    # Apply limit
    if limit:
        article_list = article_list[:limit]
        print(f"[LIMIT] Processing first {limit} articles")

    # Download articles
    downloaded = 0
    skipped = 0
    failed = 0
    processed = 0
    downloaded_files = []
    downloaded_urls = []

    for url, slug in article_list:
        if batch_size and downloaded >= batch_size:
            print(f"\n[BATCH] Reached batch size of {batch_size} new downloads, stopping")
            break

        if url in processed_urls:
            print(f"[SKIP] Already processed: {url}")
            skipped += 1
            processed += 1
            continue

        processed += 1
        status, filepath, error = download_sevan_article(url, slug, output_dir, delay, skip_existing)

        if status == 'downloaded':
            print(f"[DOWNLOAD] {filepath}")
            downloaded += 1
            downloaded_files.append(filepath)
            downloaded_urls.append(url)
        elif status == 'skipped':
            print(f"[SKIP] HTML exists: {filepath}")
            skipped += 1
        else:
            print(f"[FAILED] {url} - {error}")
            failed += 1

    return {
        'downloaded_count': downloaded,
        'skipped_count': skipped,
        'failed_count': failed,
        'total_processed': processed,
        'downloaded_files': downloaded_files,
        'downloaded_urls': downloaded_urls
    }


# =============================================================================
# Unified Download Function
# =============================================================================

def download_articles(source='substack', sitemap_path=None, output_dir=None, limit=None,
                      delay=1.0, batch_size=None, skip_existing=False, processed_urls_file=None,
                      keywords=None, order='desc'):
    """
    Download articles from specified source.

    Args:
        source: 'substack' or 'sevan'
        sitemap_path: Path to sitemap.xml (for substack)
        output_dir: Directory to save HTML files
        limit: Maximum number of articles
        delay: Delay between requests
        batch_size: Stop after this many new downloads
        skip_existing: Skip existing files
        processed_urls_file: File tracking processed URLs
        keywords: Keywords filter (for sevan)
        order: Sort order (for sevan)

    Returns:
        Dictionary with download statistics
    """
    if source == 'sevan':
        if output_dir is None:
            output_dir = './sources/sevan'
        return download_articles_sevan(
            output_dir=output_dir,
            limit=limit,
            delay=delay,
            batch_size=batch_size,
            skip_existing=skip_existing,
            processed_urls_file=processed_urls_file,
            keywords=keywords,
            order=order
        )
    else:
        # Default to substack
        if output_dir is None:
            output_dir = './sources/substack'
        if sitemap_path is None:
            sitemap_path = 'sitemap.xml'
        return download_articles_substack(
            sitemap_path=sitemap_path,
            output_dir=output_dir,
            limit=limit,
            delay=delay,
            batch_size=batch_size,
            skip_existing=skip_existing,
            processed_urls_file=processed_urls_file
        )


def main():
    parser = argparse.ArgumentParser(
        description='Download articles from Substack or SevanNisanyan.com'
    )
    parser.add_argument(
        '--source',
        choices=['substack', 'sevan'],
        default='substack',
        help='Article source (default: substack)'
    )
    parser.add_argument(
        '--sitemap',
        default='sitemap.xml',
        help='Path to sitemap.xml for Substack (default: sitemap.xml)'
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Output directory (default: ./sources/substack or ./sources/sevan)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of articles to process (default: all)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Stop after downloading this many NEW files (default: no limit)'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip files that already exist'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=1.0,
        help='Delay between requests in seconds (default: 1.0)'
    )
    parser.add_argument(
        '--processed-urls-file',
        default=None,
        help='Path to file containing already-processed URLs to skip'
    )
    parser.add_argument(
        '--keywords',
        nargs='+',
        default=None,
        help='Keywords filter for SevanNisanyan.com (e.g., "Pazar Sohbeti")'
    )
    parser.add_argument(
        '--order',
        choices=['asc', 'desc'],
        default='desc',
        help='Sort order for SevanNisanyan.com (default: desc)'
    )

    args = parser.parse_args()

    print("="*60)
    print(f"Article Downloader - Source: {args.source.upper()}")
    print("="*60)
    print()

    result = download_articles(
        source=args.source,
        sitemap_path=args.sitemap,
        output_dir=args.output,
        limit=args.limit,
        delay=args.delay,
        batch_size=args.batch_size,
        skip_existing=args.skip_existing,
        processed_urls_file=args.processed_urls_file,
        keywords=args.keywords,
        order=args.order
    )

    print()
    print("="*60)
    print(f"Download complete!")
    print(f"  Downloaded: {result['downloaded_count']}")
    print(f"  Skipped: {result['skipped_count']}")
    print(f"  Failed: {result['failed_count']}")
    if args.output:
        print(f"  Output directory: {args.output}")
    print("="*60)


if __name__ == '__main__':
    main()
