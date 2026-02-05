"""
main.py - Complete Substack scraping and conversion pipeline

Usage:
    python3 main.py                    # Download and convert all articles
    python3 main.py -n 33              # Download and convert first 33 articles
    python3 main.py --skip-download    # Only convert existing HTML files
    python3 main.py --skip-convert     # Only download, don't convert

This script orchestrates the complete pipeline:
1. Download articles from sitemap.xml as HTML
2. Convert HTML files to clean Markdown
"""

import argparse
import os
from download_substack import download_articles
from convert_to_md import convert_multiple_files


def main():
    parser = argparse.ArgumentParser(
        description='Download and convert Substack articles to Markdown',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 main.py                    # Process all articles
  python3 main.py -n 33              # Process first 33 articles
  python3 main.py --skip-download    # Only convert existing HTML
  python3 main.py --skip-convert     # Only download HTML
        """
    )
    
    parser.add_argument(
        '-n', '--limit',
        type=int,
        default=None,
        help='Maximum number of articles to process (default: all)'
    )
    parser.add_argument(
        '--sitemap',
        default='sitemap.xml',
        help='Path to sitemap.xml (default: sitemap.xml)'
    )
    parser.add_argument(
        '--source-dir',
        default='./sources/substack',
        help='Directory for downloaded HTML files (default: ./sources/substack)'
    )
    parser.add_argument(
        '--output-dir',
        default='./formatted/substack',
        help='Directory for converted Markdown files (default: ./formatted/substack)'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip download step, only convert existing files'
    )
    parser.add_argument(
        '--skip-convert',
        action='store_true',
        help='Skip conversion step, only download files'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=1.0,
        help='Delay between download requests in seconds (default: 1.0)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print(" "*20 + "SUBSTACK SCRAPER PIPELINE")
    print("="*70)
    print()
    
    download_success = 0
    download_failed = 0
    download_total = 0
    
    convert_success = 0
    convert_failed = 0
    convert_total = 0
    
    # Step 1: Download articles
    if not args.skip_download:
        print("STEP 1: Downloading Articles")
        print("-" * 70)
        print()
        
        download_success, download_failed, download_total = download_articles(
            sitemap_path=args.sitemap,
            output_dir=args.source_dir,
            limit=args.limit,
            delay=args.delay
        )
        
        print()
        print(f"Download Summary: {download_success}/{download_total} successful")
        print()
    else:
        print("STEP 1: Downloading Articles - SKIPPED")
        print()
    
    # Step 2: Convert to Markdown
    if not args.skip_convert:
        print("="*70)
        print("STEP 2: Converting to Markdown")
        print("-" * 70)
        print()
        
        # Check if source directory exists
        if not os.path.exists(args.source_dir):
            print(f"Error: Source directory '{args.source_dir}' does not exist.")
            print("Run without --skip-download first, or specify correct --source-dir")
            return
        
        convert_success, convert_failed, convert_total = convert_multiple_files(
            input_dir=args.source_dir,
            output_dir=args.output_dir,
            limit=args.limit
        )
        
        print()
        print(f"Conversion Summary: {convert_success}/{convert_total} successful")
        print()
    else:
        print("="*70)
        print("STEP 2: Converting to Markdown - SKIPPED")
        print()
    
    # Final Summary
    print("="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    
    if not args.skip_download:
        print(f"Downloaded:  {download_success}/{download_total} articles")
        print(f"             Location: {args.source_dir}")
    
    if not args.skip_convert:
        print(f"Converted:   {convert_success}/{convert_total} articles")
        print(f"             Location: {args.output_dir}")
    
    # Overall success rate
    if not args.skip_download and not args.skip_convert:
        overall_success = min(download_success, convert_success)
        overall_total = download_total
        print()
        print(f"Overall:     {overall_success}/{overall_total} articles fully processed")
        
        if overall_success < overall_total:
            print()
            print("Note: Some articles failed. Check logs above for details.")
    
    print("="*70)


if __name__ == '__main__':
    main()