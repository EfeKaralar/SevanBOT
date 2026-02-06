#!/usr/bin/env python3
"""
Remove comments from Substack markdown files.
Comments appear under a header pattern: #### X yorum:
where X is the number of comments.
"""

import re
from pathlib import Path


def remove_comments_from_file(file_path: Path) -> bool:
    """
    Remove comments section from a markdown file.

    Args:
        file_path: Path to the markdown file

    Returns:
        True if comments were found and removed, False otherwise
    """
    content = file_path.read_text(encoding='utf-8')

    # Pattern to match: #### X yorum: where X is one or more digits
    # Match from this header to the end of the file
    pattern = r'#### \d+ yorum:.*'

    # Check if pattern exists
    if not re.search(pattern, content, re.DOTALL):
        return False

    # Remove everything from the comment header onwards
    new_content = re.sub(pattern, '', content, flags=re.DOTALL)

    # Remove trailing whitespace and ensure single newline at end
    new_content = new_content.rstrip() + '\n'

    # Write back to file
    file_path.write_text(new_content, encoding='utf-8')
    return True


def main():
    """Process all markdown files in formatted/substack directory."""
    substack_dir = Path('formatted/substack')

    if not substack_dir.exists():
        print(f"Error: Directory {substack_dir} not found")
        return

    # Find all markdown files
    md_files = list(substack_dir.glob('*.md'))

    if not md_files:
        print(f"No markdown files found in {substack_dir}")
        return

    print(f"Found {len(md_files)} markdown files")

    # Process each file
    files_with_comments = 0
    files_without_comments = 0

    for md_file in md_files:
        if remove_comments_from_file(md_file):
            files_with_comments += 1
            print(f"✓ Removed comments from: {md_file.name}")
        else:
            files_without_comments += 1

    print(f"\nSummary:")
    print(f"  Files with comments removed: {files_with_comments}")
    print(f"  Files without comments: {files_without_comments}")
    print(f"  Total files processed: {len(md_files)}")


if __name__ == '__main__':
    main()
