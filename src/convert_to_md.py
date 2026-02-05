"""
convert_to_md.py - Convert Substack HTML articles to clean Markdown

Usage:
    python3 convert_to_md.py <input_html> [output_dir]
    python3 convert_to_md.py input.html                    # outputs to ./formatted/substack/
    python3 convert_to_md.py input.html /custom/path/      # outputs to custom path

This script converts raw Substack HTML to clean, RAG-ready Markdown.
"""

import sys
import os
import re
import argparse
from bs4 import BeautifulSoup
from markdownify import markdownify as md


def clean_html_whitespace(soup):
    """
    Clean whitespace within HTML elements before markdown conversion.
    This prevents random newlines within paragraphs while preserving
    structural breaks between elements and spaces around inline tags.
    
    Args:
        soup: BeautifulSoup object
        
    Returns:
        Cleaned BeautifulSoup object
    """
    # Block-level elements where we normalize whitespace
    block_elements = ['p', 'li', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 
                      'blockquote', 'td', 'th']
    
    for element in soup.find_all(block_elements):
        # Process all text nodes within this element
        for content in element.descendants:
            if isinstance(content, str):
                # Replace multiple whitespace (including newlines) with single space
                cleaned = re.sub(r'\s+', ' ', content)
                if cleaned != content:
                    content.replace_with(cleaned)
    
    return soup


def clean_markdown_final(text):
    """
    Final cleanup of markdown - minimal since HTML preprocessing handles most issues.
    
    Args:
        text: Markdown text
        
    Returns:
        Cleaned markdown text
    """
    # Remove excessive blank lines (more than 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove escaped asterisks that are standalone (separator artifacts)
    text = re.sub(r'\n\\\*\s*\n', '\n\n---\n\n', text)
    
    # Remove trailing whitespace from each line
    text = '\n'.join(line.rstrip() for line in text.split('\n'))
    
    return text.strip()


def extract_article_content(soup):
    """
    Extract article components from Substack HTML.
    
    Args:
        soup: BeautifulSoup object of the HTML page
        
    Returns:
        Dictionary with title, subtitle, date, and content_elem
    """
    title_elem = soup.find('h1', class_='post-title')
    title = title_elem.get_text(strip=True) if title_elem else 'Untitled'
    
    subtitle_elem = soup.find('h3', class_='subtitle')
    subtitle = subtitle_elem.get_text(strip=True) if subtitle_elem else ''
    
    # Date - look for date patterns
    date = ''
    date_elem = soup.find('div', class_='pencraft pc-reset', string=lambda x: x and ',' in str(x))
    if date_elem:
        date = date_elem.get_text(strip=True)
    else:
        # Alternative: look for meta divs with month names
        all_divs = soup.find_all('div', class_=lambda x: x and 'meta' in str(x))
        for div in all_divs:
            text = div.get_text(strip=True)
            if any(month in text for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']):
                date = text
                break
    
    # Main content
    content_elem = soup.find('div', class_='available-content')
    if not content_elem:
        content_elem = soup.find('div', class_='body markup')
    
    # Get source URL if available
    canonical = soup.find('link', rel='canonical')
    source_url = canonical.get('href') if canonical else ''
    
    return {
        'title': title,
        'subtitle': subtitle,
        'date': date,
        'source_url': source_url,
        'content_elem': content_elem
    }


def convert_html_to_markdown(html_path, output_dir='./formatted/substack'):
    """
    Convert a single HTML file to Markdown.

    Args:
        html_path: Path to HTML file
        output_dir: Directory to save markdown file

    Returns:
        Tuple of (success: bool, output_path: str or None, error: str or None, source_url: str or None)
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Read HTML
        with open(html_path, 'r', encoding='utf-8') as f:
            html = f.read()
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract article components
        article = extract_article_content(soup)
        
        if not article['content_elem']:
            return False, None, "No content found in HTML", None
        
        # KEY STEP: Clean HTML whitespace BEFORE markdown conversion
        content_elem_cleaned = clean_html_whitespace(article['content_elem'])
        
        # Convert cleaned HTML to Markdown
        content_md = md(str(content_elem_cleaned), 
                       heading_style="ATX",
                       bullets="-",
                       strip=['script', 'style'])
        
        # Final minimal cleanup
        content_md = clean_markdown_final(content_md)
        
        # Build final markdown
        markdown_content = f"# {article['title']}\n\n"
        
        if article['subtitle']:
            markdown_content += f"*{article['subtitle']}*\n\n"
        
        if article['date']:
            markdown_content += f"{article['date']}\n\n"
        
        if article['source_url']:
            markdown_content += f"**Source:** {article['source_url']}\n\n"
        
        markdown_content += "---\n\n"
        markdown_content += content_md
        
        # Create output filename (same as input but .md)
        input_basename = os.path.basename(html_path)
        output_basename = os.path.splitext(input_basename)[0] + '.md'
        output_path = os.path.join(output_dir, output_basename)
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        return True, output_path, None, article['source_url']

    except Exception as e:
        return False, None, str(e), None


def convert_multiple_files(input_dir, output_dir='./formatted/substack', limit=None):
    """
    Convert all HTML files in a directory to Markdown.
    
    Args:
        input_dir: Directory containing HTML files
        output_dir: Directory to save markdown files
        limit: Maximum number of files to convert (None for all)
        
    Returns:
        Tuple of (successful_count, failed_count, total_count)
    """
    # Get all HTML files
    html_files = [f for f in os.listdir(input_dir) if f.endswith('.html')]
    
    if limit:
        html_files = html_files[:limit]
    
    print(f"Found {len(html_files)} HTML files to convert\n")
    
    successful = 0
    failed = 0
    
    for i, filename in enumerate(html_files, 1):
        html_path = os.path.join(input_dir, filename)
        print(f"[{i}/{len(html_files)}] Converting: {filename}")
        
        success, output_path, error, _source_url = convert_html_to_markdown(html_path, output_dir)

        if success:
            print(f"  ✓ Saved to {output_path}")
            successful += 1
        else:
            print(f"  ✗ Error: {error}")
            failed += 1
    
    return successful, failed, len(html_files)


def main():
    parser = argparse.ArgumentParser(
        description='Convert Substack HTML to clean Markdown',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 convert_to_md.py article.html
  python3 convert_to_md.py article.html /custom/output/
  python3 convert_to_md.py --input-dir ./sources/substack --limit 10
        """
    )
    
    # Single file mode
    parser.add_argument(
        'input_path',
        nargs='?',
        help='Input HTML file or directory'
    )
    parser.add_argument(
        'output_dir',
        nargs='?',
        default='./formatted/substack',
        help='Output directory (default: ./formatted/substack)'
    )
    
    # Batch mode
    parser.add_argument(
        '--input-dir',
        help='Convert all HTML files in this directory'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of files to convert (batch mode only)'
    )
    
    args = parser.parse_args()
    
    # Determine mode
    if args.input_dir:
        # Batch mode
        print("="*60)
        print("Batch HTML to Markdown Converter")
        print("="*60)
        print()
        
        successful, failed, total = convert_multiple_files(
            args.input_dir,
            args.output_dir,
            args.limit
        )
        
        print()
        print("="*60)
        print(f"Conversion complete!")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Total: {total}")
        print(f"  Output directory: {args.output_dir}")
        print("="*60)
        
    elif args.input_path:
        # Single file mode
        if os.path.isdir(args.input_path):
            # User provided directory as first arg
            print("="*60)
            print("Batch HTML to Markdown Converter")
            print("="*60)
            print()
            
            successful, failed, total = convert_multiple_files(
                args.input_path,
                args.output_dir,
                None
            )
            
            print()
            print("="*60)
            print(f"Conversion complete!")
            print(f"  Successful: {successful}")
            print(f"  Failed: {failed}")
            print(f"  Total: {total}")
            print(f"  Output directory: {args.output_dir}")
            print("="*60)
        else:
            # Single file
            print(f"Converting: {args.input_path}")
            success, output_path, error, _source_url = convert_html_to_markdown(
                args.input_path,
                args.output_dir
            )

            if success:
                print(f"✓ Success! Saved to: {output_path}")
            else:
                print(f"✗ Error: {error}")
                sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()