"""
convert_to_md.py - Convert HTML articles to clean Markdown

Supports:
- Substack articles (HTML with post-title, subtitle, available-content classes)
- SevanNisanyan.com articles (SvelteKit with embedded JSON data)

Usage:
    python3 convert_to_md.py <input_html> [output_dir]
    python3 convert_to_md.py input.html                    # outputs to ./formatted/substack/
    python3 convert_to_md.py input.html /custom/path/      # outputs to custom path

This script converts raw HTML to clean, RAG-ready Markdown.
"""

import sys
import os
import re
import json
import argparse
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from datetime import datetime


def fix_turkish_encoding(text):
    """
    Fix mojibake (double-encoded UTF-8) in Turkish text.

    This fixes text that was originally UTF-8 but was misinterpreted as Latin-1
    and then re-encoded as UTF-8.

    Args:
        text: Potentially mojibake text

    Returns:
        Properly decoded text
    """
    if not text:
        return text

    try:
        # Try to fix the double-encoding: encode as Latin-1, decode as UTF-8
        fixed = text.encode('latin-1').decode('utf-8')
        return fixed
    except (UnicodeDecodeError, UnicodeEncodeError):
        # If it fails, the text is probably already correct
        return text


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


def detect_source_type(html):
    """
    Detect whether HTML is from Substack or SevanNisanyan.com.

    Args:
        html: Raw HTML string

    Returns:
        'sevan' or 'substack'
    """
    if '__sveltekit' in html and 'sevannisanyan' in html.lower():
        return 'sevan'
    return 'substack'


def extract_sevan_json_data(html):
    """
    Extract the embedded JSON data from SevanNisanyan.com SvelteKit HTML.

    Args:
        html: Raw HTML string

    Returns:
        Dictionary with article data or None if not found
    """
    # Look for the kit.start() call with embedded data
    # Pattern: data: [null,null,{type:"data",data:{entry:{...}}}]
    pattern = r'data:\s*\[null,null,\{type:"data",data:\{entry:(\{.*?\})\},uses:'
    match = re.search(pattern, html, re.DOTALL)

    if not match:
        return None

    try:
        # The matched content is JavaScript object notation, not pure JSON
        # We need to convert it: handle unquoted keys, new Date(), etc.
        js_obj = match.group(1)

        # Convert JavaScript object to valid JSON:
        # 1. Quote unquoted keys
        js_obj = re.sub(r'(\{|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', js_obj)

        # 2. Replace new Date(timestamp) with the timestamp
        js_obj = re.sub(r'new Date\((\d+)\)', r'\1', js_obj)

        # Parse as JSON
        data = json.loads(js_obj)
        return data

    except (json.JSONDecodeError, Exception):
        return None


def extract_sevan_article_content(html):
    """
    Extract article components from SevanNisanyan.com HTML.

    Args:
        html: Raw HTML string

    Returns:
        Dictionary with title, keywords, source, date, text_content, source_url
    """
    data = extract_sevan_json_data(html)

    if data:
        # Extract from JSON data
        title = data.get('title', {}).get('tr', 'Untitled')
        keywords = data.get('keywords', {}).get('tr', [])
        source = data.get('source', {}).get('tr', '')
        slug = data.get('slug', '')

        # Handle dates (timestamps in milliseconds)
        dates = data.get('dates', [])
        date_str = ''
        if dates and isinstance(dates[0], (int, float)):
            try:
                dt = datetime.fromtimestamp(dates[0] / 1000)
                date_str = dt.strftime('%d %B %Y')
            except:
                pass

        # Extract text content (questions and answers)
        text_sections = data.get('text', [])
        content_parts = []

        for section in text_sections:
            questions = section.get('question') or []
            answers = section.get('answer') or []

            for q in questions:
                if q:
                    content_parts.append(f"**{q}**")

            for a in answers:
                if a:
                    content_parts.append(a)

        text_content = '\n\n'.join(content_parts)
        source_url = f"https://www.sevannisanyan.com/metin/{slug}" if slug else ''

        # Fix Turkish encoding issues (mojibake)
        title = fix_turkish_encoding(title)
        keywords = [fix_turkish_encoding(k) for k in keywords]
        source = fix_turkish_encoding(source)
        text_content = fix_turkish_encoding(text_content)

        return {
            'title': title,
            'keywords': keywords,
            'source': source,
            'date': date_str,
            'text_content': text_content,
            'source_url': source_url,
            'source_type': 'sevan'
        }

    # Fallback: parse from HTML DOM
    soup = BeautifulSoup(html, 'html.parser')

    # Title from the header div
    title_elem = soup.find('div', class_=lambda x: x and 'text-xl' in str(x) and 'font-semibold' in str(x))
    title = title_elem.get_text(strip=True) if title_elem else 'Untitled'

    # Source/category
    source = ''
    source_elem = soup.find('div', class_=lambda x: x and 'truncate' in str(x) and 'text-start' in str(x))
    if source_elem:
        source = source_elem.get_text(strip=True)

    # Date
    date_str = ''
    date_elem = soup.find('div', class_=lambda x: x and 'tabular-nums' in str(x))
    if date_elem:
        date_str = date_elem.get_text(strip=True)

    # Keywords
    keywords = []
    keyword_links = soup.find_all('a', class_=lambda x: x and 'rounded-md' in str(x) and 'from-neutral-500' in str(x))
    for link in keyword_links:
        keywords.append(link.get_text(strip=True))

    # Main text content
    content_div = soup.find('div', class_=lambda x: x and 'whitespace-pre-wrap' in str(x))
    text_content = ''
    if content_div:
        # Get all text divs
        text_divs = content_div.find_all('div', recursive=True)
        parts = []
        for div in text_divs:
            # Check if it's a question (bold/semibold)
            if 'font-semibold' in str(div.get('class', [])):
                text = div.get_text(strip=True)
                if text:
                    parts.append(f"**{text}**")
            else:
                # Regular paragraph
                text = div.get_text(strip=True)
                if text and text not in [p.strip('*') for p in parts]:
                    parts.append(text)
        text_content = '\n\n'.join(parts)

    # Source URL from canonical link
    canonical = soup.find('link', rel='canonical')
    source_url = canonical.get('href') if canonical else ''

    return {
        'title': title,
        'keywords': keywords,
        'source': source,
        'date': date_str,
        'text_content': text_content,
        'source_url': source_url,
        'source_type': 'sevan'
    }


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
        'content_elem': content_elem,
        'source_type': 'substack'
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

        # Detect source type
        source_type = detect_source_type(html)

        if source_type == 'sevan':
            # SevanNisanyan.com format
            article = extract_sevan_article_content(html)

            if not article['text_content']:
                return False, None, "No content found in HTML", None

            # Build markdown for SevanNisanyan
            markdown_content = f"# {article['title']}\n\n"

            if article['source']:
                markdown_content += f"*{article['source']}*\n\n"

            if article['date']:
                markdown_content += f"{article['date']}\n\n"

            if article['keywords']:
                keywords_str = ', '.join(article['keywords'])
                markdown_content += f"**Anahtar Kelimeler:** {keywords_str}\n\n"

            if article['source_url']:
                markdown_content += f"**Kaynak:** {article['source_url']}\n\n"

            markdown_content += "---\n\n"
            markdown_content += article['text_content']

        else:
            # Substack format
            soup = BeautifulSoup(html, 'html.parser')
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

            if article.get('subtitle'):
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