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


def download_article(url, output_dir, delay=1.0):
    """
    Download a single article and save as HTML.
    
    Args:
        url: Article URL
        output_dir: Directory to save HTML files
        delay: Delay in seconds between requests (be polite to server)
        
    Returns:
        Tuple of (success: bool, filepath: str or None, error: str or None)
    """
    try:
        # Extract article slug from URL for filename
        slug = url.rstrip('/').split('/')[-1]
        filename = f"{slug}.html"
        filepath = os.path.join(output_dir, filename)
        
        # Download
        response = requests.get(url)
        response.raise_for_status()
        
        # Save
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        # Be polite to server
        time.sleep(delay)
        
        return True, filepath, None
        
    except Exception as e:
        return False, None, str(e)


def download_articles(sitemap_path, output_dir='./sources/substack', limit=None, delay=1.0):
    """
    Download all articles from sitemap.
    
    Args:
        sitemap_path: Path to sitemap.xml
        output_dir: Directory to save HTML files (default: ./sources/substack)
        limit: Maximum number of articles to download (None for all)
        delay: Delay between requests in seconds
        
    Returns:
        Tuple of (successful_count, failed_count, total_count)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse sitemap
    print(f"Reading sitemap: {sitemap_path}")
    post_urls = parse_sitemap(sitemap_path)
    total_count = len(post_urls)
    
    # Apply limit
    if limit:
        post_urls = post_urls[:limit]
        print(f"Limiting to {limit} articles (out of {total_count} total)")
    
    print(f"Found {len(post_urls)} articles to download\n")
    
    # Download each article
    successful = 0
    failed = 0
    
    for i, url in enumerate(post_urls, 1):
        print(f"[{i}/{len(post_urls)}] Downloading: {url}")
        
        success, filepath, error = download_article(url, output_dir, delay)
        
        if success:
            print(f"  ✓ Saved to {filepath}")
            successful += 1
        else:
            print(f"  ✗ Error: {error}")
            failed += 1
    
    return successful, failed, len(post_urls)


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
        help='Maximum number of articles to download (default: all)'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=1.0,
        help='Delay between requests in seconds (default: 1.0)'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Substack Article Downloader")
    print("="*60)
    print()
    
    successful, failed, total = download_articles(
        args.sitemap,
        args.output,
        args.limit,
        args.delay
    )
    
    print()
    print("="*60)
    print(f"Download complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total: {total}")
    print(f"  Output directory: {args.output}")
    print("="*60)


if __name__ == '__main__':
    main()