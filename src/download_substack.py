"""
download_substack.py - Download articles from Substack

Usage:
    python3 download_substack.py [--sitemap PATH] [--output DIR] [--limit N]

This script downloads Substack articles and saves them as raw HTML files.
"""

import requests
import time
import os
import argparse
import xml.etree.ElementTree as ET

# Default path for processed URLs file
DEFAULT_PROCESSED_URLS_FILE = 'processed_urls.txt'


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


def download_articles(sitemap_path, output_dir='./sources/substack', limit=None, delay=1.0,
                      batch_size=None, skip_existing=False, processed_urls_file=None):
    """
    Download articles from sitemap.

    Args:
        sitemap_path: Path to sitemap.xml
        output_dir: Directory to save HTML files (default: ./sources/substack)
        limit: Maximum number of articles to process (None for all)
        delay: Delay between requests in seconds
        batch_size: If set, stop after downloading this many NEW files (skipped files don't count)
        skip_existing: If True, skip files that already exist (checks HTML file)
        processed_urls_file: If set, skip URLs listed in this file (checks processed URLs list)

    Returns:
        Dictionary with:
            - downloaded_count: Number of newly downloaded files
            - skipped_count: Number of skipped (existing) files
            - failed_count: Number of failed downloads
            - total_processed: Total URLs processed
            - downloaded_files: List of paths to newly downloaded files
            - downloaded_urls: List of URLs that were downloaded (parallel to downloaded_files)
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


def main():
    parser = argparse.ArgumentParser(
        description='Download Substack articles from sitemap.xml'
    )
    parser.add_argument(
        '--sitemap',
        default='sitemap.xml',
        help='Path to sitemap.xml (default: sitemap.xml)'
    )
    parser.add_argument(
        '--output',
        default='./sources/substack',
        help='Output directory (default: ./sources/substack)'
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

    args = parser.parse_args()

    print("="*60)
    print("Substack Article Downloader")
    print("="*60)
    print()

    result = download_articles(
        args.sitemap,
        args.output,
        args.limit,
        args.delay,
        args.batch_size,
        args.skip_existing,
        args.processed_urls_file
    )

    print()
    print("="*60)
    print(f"Download complete!")
    print(f"  Downloaded: {result['downloaded_count']}")
    print(f"  Skipped: {result['skipped_count']}")
    print(f"  Failed: {result['failed_count']}")
    print(f"  Output directory: {args.output}")
    print("="*60)


if __name__ == '__main__':
    main()