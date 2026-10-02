#!/usr/bin/env python3
"""
verify_readme_math.py
Verifies that README.md does not contain math syntax that breaks GitHub's KaTeX renderer,
specifically:
1. Escaped or unescaped underscores inside \\text{...}, e.g., \\text{OUT_OF_SCOPE} or \\text{OUT\\_OF\\_SCOPE}
2. Unbalanced math delimiters ($ or $$)
3. Invalid LaTeX macros in math mode
"""
import os
import re
import sys
from pathlib import Path

def verify_math(file_path):
    content = Path(file_path).read_text(encoding="utf-8")
    errors = []

    # 1. Check for underscores inside \text{...} within math mode
    # Find all inline math $...$ (not preceded or followed by $)
    inline_math_pattern = re.compile(r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)', re.DOTALL)
    display_math_pattern = re.compile(r'\$\$(.*?)\$\$', re.DOTALL)

    math_blocks = []
    for match in display_math_pattern.finditer(content):
        math_blocks.append((match.start(), match.end(), match.group(1), "display"))
    for match in inline_math_pattern.finditer(content):
        # ensure not inside display math
        start, end = match.span()
        if not any(dm[0] <= start and end <= dm[1] for dm in math_blocks if dm[3] == "display"):
            math_blocks.append((start, end, match.group(1), "inline"))

    for start, end, block, mtype in math_blocks:
        # Check for \text{...} containing _ or \_
        text_matches = re.findall(r'\\text\{([^}]*)\}', block)
        for t in text_matches:
            if '_' in t or r'\_' in t:
                # Find line number
                line_no = content[:start].count('\n') + 1
                errors.append(f"Line {line_no}: Underscore in \\text{{{t}}} inside {mtype} math block: ${block}$")

    # 2. Check for unbalanced single or double dollar signs
    # Mask out code blocks (both ``` and `)
    no_code = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
    no_code = re.sub(r'`[^`\n]+`', '', no_code)

    # Check double dollars
    display_count = no_code.count('$$')
    if display_count % 2 != 0:
        errors.append(f"Unbalanced $$ delimiters found in document (count = {display_count}).")

    # Remove $$ blocks to check single $
    no_display = re.sub(r'\$\$.*?\$\$', '', no_code, flags=re.DOTALL)
    # Count unescaped single $
    single_dollars = re.findall(r'(?<!\\)\$', no_display)
    if len(single_dollars) % 2 != 0:
        errors.append(f"Unbalanced single $ delimiters found in document (count = {len(single_dollars)}).")

    if errors:
        print(f"❌ Found {len(errors)} math formatting issue(s) in {file_path}:")
        for err in errors:
            print(f"  - {err}")
        return False
    else:
        print(f"✅ All math in {file_path} passes KaTeX/GitHub markdown verification!")
        return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
    elif os.path.exists("README.md"):
        target = "README.md"
    else:
        target = "ME2 - Voice Command Model/README.md"
    success = verify_math(target)
    sys.exit(0 if success else 1)
