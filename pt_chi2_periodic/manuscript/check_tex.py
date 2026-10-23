#!/usr/bin/env python3
"""
check_tex.py -- static checks of main.tex for an environment without TeX (catches what usually
breaks a first compile).  It does not replace compiling: run `make` where TeX Live is installed.

  braces        unescaped { } balance (comments stripped)
  math          unescaped $ count is even
  environments  every \\begin{x} closed by \\end{x}, properly nested
  conditionals  \\if... / \\fi balance (\\newif lines excluded)
  labels        every \\ref / \\eqref target has a \\label; duplicate labels
  citations     every \\cite key is in refs.bib; bib entries never cited (warning)
  files         every \\input and \\includegraphics target exists (.eps AND .pdf for figures)
  ascii         non-ASCII characters (warning; fine with utf8 input, but arXiv's latex is safer without)
usage:  python3 check_tex.py [main.tex]       exit status 1 if an error was found
"""
import os
import re
import sys

tex_path = sys.argv[1] if len(sys.argv) > 1 else 'main.tex'
root = os.path.dirname(os.path.abspath(tex_path))
src = open(tex_path, encoding='utf-8').read()
errors, warnings = [], []


def strip_comments(s):
    """drop everything after an unescaped % on each line"""
    return '\n'.join(re.sub(r'(?<!\\)%.*', '', line) for line in s.split('\n'))


body = strip_comments(src)
# \input'ed tables are checked too (they are part of the document)
for m in re.finditer(r'\\input\{([^}]+)\}', body):
    p = os.path.join(root, m.group(1) if m.group(1).endswith('.tex') else m.group(1) + '.tex')
    if not os.path.exists(p):
        errors.append(f"\\input file missing: {m.group(1)}")
    else:
        body += '\n' + strip_comments(open(p, encoding='utf-8').read())

# braces (ignore \{ \} and \\ followed by brace is still a brace -> handle \\ first)
b = re.sub(r'\\\\', '', body)
b = re.sub(r'\\[{}]', '', b)
depth, line = 0, 1
for ch in b:
    if ch == '\n':
        line += 1
    elif ch == '{':
        depth += 1
    elif ch == '}':
        depth -= 1
        if depth < 0:
            errors.append(f"unmatched '}}' near line {line}")
            depth = 0
if depth:
    errors.append(f"{depth} unclosed '{{'")

# math shifts
d = len(re.findall(r'(?<!\\)\$', re.sub(r'\\\\', '', body)))
if d % 2:
    errors.append(f"odd number of $ ({d})")

# environments
stack = []
for m in re.finditer(r'\\(begin|end)\{([^}]+)\}', body):
    kind, env = m.groups()
    ln = body[:m.start()].count('\n') + 1
    if kind == 'begin':
        stack.append((env, ln))
    elif not stack or stack[-1][0] != env:
        errors.append(f"\\end{{{env}}} (line ~{ln}) does not close {stack[-1] if stack else 'anything'}")
    else:
        stack.pop()
for env, ln in stack:
    errors.append(f"\\begin{{{env}}} at line ~{ln} never closed")

# TeX conditionals
nif = len(re.findall(r'\\if[a-zA-Z]+', re.sub(r'\\newif\\if[a-zA-Z]+', '', body)))
nfi = len(re.findall(r'\\fi\b', body))
if nif != nfi:
    errors.append(f"\\if... ({nif}) and \\fi ({nfi}) do not balance")

# labels and references
labels = re.findall(r'\\label\{([^}]+)\}', body)
for l in set(labels):
    if labels.count(l) > 1:
        errors.append(f"duplicate label {l}")
for r in re.findall(r'\\(?:eq)?ref\{([^}]+)\}', body):
    if r not in labels:
        errors.append(f"\\ref to undefined label {r}")

# citations
bib = open(os.path.join(root, 'refs.bib'), encoding='utf-8').read()
keys = re.findall(r'@\w+\{([^,\s]+),', bib)
cited = set()
for m in re.finditer(r'\\cite\w*\{([^}]+)\}', body):
    for k in m.group(1).split(','):
        cited.add(k.strip())
for k in sorted(cited - set(keys)):
    errors.append(f"\\cite key not in refs.bib: {k}")
for k in sorted(set(keys) - cited):
    warnings.append(f"bib entry never cited: {k}")

# graphics: both formats, so latex (eps) and pdflatex (pdf) both work
for f in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', body):
    for ext in ('.eps', '.pdf'):
        if not os.path.exists(os.path.join(root, f + ext)):
            errors.append(f"figure missing: {f}{ext}")

# macros: every camelCase command (the generated-number style, e.g. \tEigOne) must be defined by
# \newcommand in main.tex or in an \input file such as numbers.tex
defined = set(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}', body))
for m in re.finditer(r'\\input\{([^}]+)\}', body):
    p = os.path.join(root, m.group(1) if m.group(1).endswith('.tex') else m.group(1) + '.tex')
    if os.path.exists(p):
        defined |= set(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}', open(p, encoding='utf-8').read()))
for tok in sorted(set(re.findall(r'\\([a-z]+[A-Z][A-Za-z]*)', body)) - defined):
    errors.append(f"undefined macro \\{tok}")

# non-ASCII
for i, l in enumerate(src.split('\n'), 1):
    bad = sorted({c for c in l if ord(c) > 127})
    if bad:
        warnings.append(f"non-ASCII on line {i}: {''.join(bad)}")

print(f"{tex_path}: {len(labels)} labels, {len(cited)} cited keys / {len(keys)} bib entries, "
      f"{len(re.findall(r'includegraphics', body))} figures")
for w in warnings:
    print("  warning:", w)
for e in errors:
    print("  ERROR:", e)
print("OK" if not errors else f"{len(errors)} error(s)")
sys.exit(1 if errors else 0)
