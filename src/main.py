"""
main.py - Complete Substack scraping and conversion pipeline

Usage:
    python3 main.py                    # Download batch of 10, convert, keep files
    python3 main.py --batch-size 20    # Download batch of 20 articles
    python3 main.py --delete-after     # Delete HTML files after conversion
    python3 main.py --skip-download    # Only convert existing HTML files
    python3 main.py --skip-convert     # Only download, don't convert

This script orchestrates the complete pipeline:
1. Download a batch of articles from sitemap.xml as HTML (skipping existing files)
2. Convert the downloaded HTML files to clean Markdown
3. Optionally delete the HTML files after conversion
"""

import argparse
import os
from download_substack import download_articles
from convert_to_md import convert_html_to_markdown


def main():
    parser = argparse.ArgumentParser(
        description='Download and convert Substack articles to Markdown',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 main.py                    # Process batch of 10 articles
  python3 main.py --batch-size 20    # Process batch of 20 articles
  python3 main.py --delete-after     # Delete HTML after conversion
  python3 main.py --skip-download    # Only convert existing HTML
  python3 main.py --skip-convert     # Only download HTML
        """
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Number of NEW articles to download per batch (default: 10)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum total number of articles to download (default: all)'
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
        '--delete-after',
        action='store_true',
        help='Delete HTML files after successful conversion (default: False)'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=1.0,
        help='Delay between download requests in seconds (default: 1.0)'
    )

    args = parser.parse_args()

    print("="*70)
    print("SUBSTACK SCRAPER PIPELINE")
    print("="*70)

    # Handle --skip-download mode separately (one-time conversion of existing files)
    if args.skip_download:
        print("\n[INFO] Download skipped - converting existing HTML files")
        print("-" * 70)

        if not os.path.exists(args.source_dir):
            print(f"[ERROR] Source directory '{args.source_dir}' does not exist.")
            return

        files_to_convert = [
            os.path.join(args.source_dir, f)
            for f in os.listdir(args.source_dir)
            if f.endswith('.html')
        ]
        print(f"[INFO] Found {len(files_to_convert)} HTML files to convert")

        for filepath in files_to_convert:
            success, output_path, error = convert_html_to_markdown(filepath, args.output_dir)
            if success:
                print(f"[CONVERT] {os.path.basename(filepath)} -> {output_path}")
            else:
                print(f"[FAILED] {os.path.basename(filepath)} - {error}")

        print("\n" + "="*70)
        print("PIPELINE COMPLETE!")
        print("="*70)
        return

    # Main batch processing loop
    total_downloaded = 0
    total_converted = 0
    total_deleted = 0
    batch_num = 0
    limit = args.limit

    if limit:
        print(f"\n[INFO] Limit set: will download up to {limit} articles total")

    while True:
        # Check if we've reached the limit
        if limit and total_downloaded >= limit:
            print(f"\n[INFO] Reached limit of {limit} articles.")
            break

        batch_num += 1
        print(f"\n{'='*70}")
        print(f"BATCH {batch_num}")
        print("="*70)

        # Calculate batch size for this iteration
        if limit:
            remaining = limit - total_downloaded
            current_batch_size = min(args.batch_size, remaining)
        else:
            current_batch_size = args.batch_size

        # Step 1: Download batch
        print(f"\n[BATCH {batch_num}] Downloading up to {current_batch_size} new articles...")
        print("-" * 70)

        result = download_articles(
            sitemap_path=args.sitemap,
            output_dir=args.source_dir,
            batch_size=current_batch_size,
            skip_existing=True,
            delay=args.delay
        )

        downloaded_files = result['downloaded_files']
        total_downloaded += result['downloaded_count']

        print("-" * 70)
        print(f"[DOWNLOAD COMPLETE] Downloaded: {result['downloaded_count']}, "
              f"Skipped: {result['skipped_count']}, Failed: {result['failed_count']}")

        # Check if we're done (no new files to download)
        if not downloaded_files:
            print(f"\n[INFO] No new files to download. All articles processed.")
            break

        # Step 2: Convert batch (unless --skip-convert)
        if not args.skip_convert:
            print(f"\n[BATCH {batch_num}] Converting {len(downloaded_files)} files...")
            print("-" * 70)

            converted_count = 0
            files_to_delete = []

            for filepath in downloaded_files:
                success, output_path, error = convert_html_to_markdown(filepath, args.output_dir)

                if success:
                    print(f"[CONVERT] {os.path.basename(filepath)}")
                    converted_count += 1
                    files_to_delete.append(filepath)
                else:
                    print(f"[FAILED] {os.path.basename(filepath)} - {error}")

            total_converted += converted_count
            print("-" * 70)
            print(f"[CONVERT COMPLETE] Converted: {converted_count}/{len(downloaded_files)}")

            # Step 3: Delete HTML files if requested
            if args.delete_after and files_to_delete:
                print(f"\n[BATCH {batch_num}] Deleting HTML files...")
                print("-" * 70)

                for filepath in files_to_delete:
                    try:
                        os.remove(filepath)
                        print(f"[DELETE] {os.path.basename(filepath)}")
                        total_deleted += 1
                    except Exception as e:
                        print(f"[FAILED] Could not delete {filepath}: {e}")

                print("-" * 70)
                print(f"[DELETE COMPLETE] Deleted: {len(files_to_delete)} files")

    # Final Summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print(f"Total batches:    {batch_num}")
    print(f"Total downloaded: {total_downloaded}" + (f" (limit: {limit})" if limit else ""))
    if not args.skip_convert:
        print(f"Total converted:  {total_converted}")
    if args.delete_after:
        print(f"Total deleted:    {total_deleted}")
    print("="*70)


if __name__ == '__main__':
    main()