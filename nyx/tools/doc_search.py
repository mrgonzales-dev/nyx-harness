"""Doc search tool — fuzzy keyword search across installed docsets.

Searches the DevDocs docsets already installed via /docs, using fuzzy
matching on entry names.  Extracts only the relevant section from each
matching page (not the whole page), strips HTML, and trims results to
keep context usage small.

Algorithm:
  1. Tokenize the query into keywords (splits camelCase, snake_case, etc).
  2. Pre-filter: quick substring scan to narrow candidates before
     expensive fuzzy matching.
  3. Score entries with weighted matching:
       - exact word match  > substring match > fuzzy match
       - name match        > path match
       - multi-keyword hits boost score
  4. Extract the relevant HTML section per match (heading to heading).
  5. Strip HTML, trim each section to MAX_SECTION_CHARS.
  6. Content re-score: boost results where more keywords appear in
     the extracted text; drop false positives.
  7. Return formatted results within a total character budget.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from nyx.tools import docs as docset_manager

# ── tuning constants ────────────────────────────────────────────

MAX_SECTION_CHARS = 1500
MAX_TOTAL_CHARS = 4000
MAX_RESULTS = 3

# Minimum fuzzy score (0.0–1.0) to keep an entry after pre-filter.
MIN_MATCH_SCORE = 0.3

# How many candidates to extract after scoring (before content filtering).
CANDIDATE_POOL = 12

# ── tokenization ────────────────────────────────────────────────

# Splits camelCase, PascalCase, snake_case, kebab-case, dot.paths.
_SPLIT_RE = re.compile(r"(?<=[a-z])(?=[A-Z])|[\s/._\-]+")


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase keyword tokens.

    Handles camelCase (HasPrefix → has, prefix), snake_case, dot paths,
    and kebab-case.  Filters out tokens shorter than 2 chars.
    """
    parts = _SPLIT_RE.split(text)
    return [p.lower() for p in parts if len(p) >= 2]


def _is_subseq(needle: str, haystack: str) -> bool:
    """Check if *needle* is a subsequence of *haystack* (all chars in order)."""
    it = iter(haystack)
    return all(c in it for c in needle)


def _split_concatenated(token: str, known_tokens: set[str]) -> list[str]:
    """Try to split a concatenated token using known entry tokens.

    If the token wasn't split by _tokenize (e.g. "stringsprefix"), try
    to break it using known_tokens as a dictionary.  For example, if
    "strings" and "prefix" are known tokens, "stringsprefix" splits
    into ["strings", "prefix"].

    Returns the original token in a list if no split is found.
    """
    if len(token) < 6:
        return [token]

    # Try to find known tokens that are prefixes of the token.
    for known in known_tokens:
        if len(known) >= 3 and token.startswith(known):
            remainder = token[len(known):]
            if len(remainder) >= 2:
                # Recursively split the remainder.
                rest_parts = _split_concatenated(remainder, known_tokens)
                if rest_parts != [remainder]:
                    return [known] + rest_parts
                # Also check if remainder itself is a known token.
                if remainder in known_tokens:
                    return [known, remainder]
                # Even if remainder isn't known, the split is useful.
                return [known, remainder]

    return [token]


# ── HTML utilities ──────────────────────────────────────────────

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_CODE_RE = re.compile(r"<pre[^>]*>(.*?)</pre>", re.DOTALL)


def _strip_html(html: str) -> str:
    """Convert HTML to plain text, preserving code blocks."""
    def _code_repl(m: re.Match) -> str:
        inner = _TAG_RE.sub("", m.group(1))
        return f"\n```\n{inner.strip()}\n```\n"

    html = _CODE_RE.sub(_code_repl, html)
    text = _TAG_RE.sub("", html)
    text = _WS_RE.sub(" ", text).strip()
    text = re.sub(r"\s*```\s*", "\n```\n", text)
    return text


def _extract_section(html: str, anchor: str) -> str:
    """Extract the HTML from the heading with id=anchor to the next heading."""
    pattern = re.compile(
        rf'<h([234])[^>]*id=["\']?{re.escape(anchor)}["\']?', re.IGNORECASE
    )
    m = pattern.search(html)
    if not m:
        # Fallback: find anchor anywhere and grab surrounding content.
        pos = html.find(f'id="{anchor}"')
        if pos == -1:
            pos = html.find(anchor)
        if pos == -1:
            return ""
        chunk = html[pos : pos + MAX_SECTION_CHARS * 3]
        return _strip_html(chunk)[:MAX_SECTION_CHARS]

    heading_level = int(m.group(1))
    start = m.start()

    # Find the next heading of same or higher level (h2 >= h3 >= h4).
    next_heading = re.compile(rf"<h[2-{heading_level}][^>]*>", re.IGNORECASE)
    rest = html[start + 1 :]
    m2 = next_heading.search(rest)
    if m2:
        section_html = html[start : start + 1 + m2.start()]
    else:
        section_html = html[start : start + MAX_SECTION_CHARS * 3]

    text = _strip_html(section_html)
    return text[:MAX_SECTION_CHARS]


# ── scoring ─────────────────────────────────────────────────────

def _score_entry(
    keywords: list[str], name: str, path: str
) -> float:
    """Score an entry against query keywords.

    Weighted scoring:
      - exact word match:    1.0 per keyword
      - substring match:     0.7 per keyword
      - fuzzy match (>0.6):  ratio per keyword
      - no match:            0.0

    Name matches are weighted 2x over path matches.
    Multi-keyword coverage gives a bonus.
    """
    name_tokens = set(_tokenize(name))
    path_tokens = set(_tokenize(path))
    name_lower = name.lower()
    path_lower = path.lower()

    total = 0.0
    hits = 0

    for kw in keywords:
        kw_score = 0.0

        # Exact word match in name (highest weight).
        if kw in name_tokens:
            kw_score = 1.0
        # Substring match in name.
        elif kw in name_lower:
            kw_score = 0.7
        # Exact word match in path.
        elif kw in path_tokens:
            kw_score = 0.5
        # Substring match in path.
        elif kw in path_lower:
            kw_score = 0.35
        else:
            # Fuzzy match against name words.
            best_fuzzy = 0.0
            for token in name_tokens:
                ratio = SequenceMatcher(None, kw, token).ratio()
                if ratio > best_fuzzy:
                    best_fuzzy = ratio
            # Also fuzzy against full name.
            ratio = SequenceMatcher(None, kw, name_lower).ratio()
            if ratio > best_fuzzy:
                best_fuzzy = ratio
            if best_fuzzy >= 0.6:
                kw_score = best_fuzzy * 0.6  # fuzzy is worth less than substring

        if kw_score > 0:
            hits += 1
        total += kw_score

    # Average per-keyword score, weighted by coverage.
    avg = total / len(keywords) if keywords else 0.0
    coverage_bonus = (hits / len(keywords)) * 0.15 if keywords else 0.0

    return min(1.0, avg + coverage_bonus)


# ── content verification ────────────────────────────────────────

def _content_keyword_hits(text: str, keywords: list[str]) -> int:
    """Count how many query keywords appear in the extracted text.

    A keyword counts as a hit if it appears as a substring, or as a
    subsequence (handles concatenated query tokens like "stringsprefix"
    matching content that has "strings.HasPrefix").
    """
    text_lower = text.lower()
    hits = 0
    for kw in keywords:
        if kw in text_lower:
            hits += 1
        elif len(kw) >= 6 and _is_subseq(kw, text_lower):
            hits += 1
    return hits


# ── docset name mapping ─────────────────────────────────────────

# Map common language names to docset slug prefixes.
_DOCSET_ALIASES: dict[str, str] = {
    "go": "go",
    "golang": "go",
    "python": "python",
    "py": "python",
    "javascript": "javascript",
    "js": "javascript",
    "typescript": "typescript",
    "ts": "typescript",
    "rust": "rust",
    "ruby": "ruby",
    "rb": "ruby",
    "java": "java",
    "c": "c",
    "cpp": "cpp",
    "c++": "cpp",
    "php": "php",
    "html": "html",
    "css": "css",
    "react": "react",
    "vue": "vue",
    "dart": "dart",
    "kotlin": "kotlin",
    "swift": "swift",
    "node": "node",
    "nodejs": "node",
    "bash": "bash",
    "shell": "bash",
    "sh": "bash",
    "sql": "postgresql",
    "postgres": "postgresql",
    "sqlite": "sqlite",
    "mongodb": "mongodb",
}


def _resolve_docset(filter_str: str) -> str | None:
    """Map a user-provided docset name to a slug prefix.

    Returns None if no mapping is found (search all docsets).
    """
    return _DOCSET_ALIASES.get(filter_str.lower())


# ── main search function ────────────────────────────────────────

def search_docs(query: str, docset: str = "") -> str:
    """Search installed docsets for *query* and return relevant snippets.

    Args:
        query: Search keywords — use specific function or package names
               for best results, e.g. "strings HasPrefix" or "json dumps".
        docset: Optional docset filter — a language name like "go",
                "python", "javascript" to narrow the search.

    Returns formatted snippets or a no-results message.
    """
    installed = docset_manager.list_installed()
    if not installed:
        return "No docsets installed. Use /docs install <name> to add one."

    keywords = _tokenize(query)
    if not keywords:
        return "Please provide search keywords (e.g. 'strings HasPrefix')."

    # Resolve optional docset filter.
    slug_filter = _resolve_docset(docset) if docset else None
    if slug_filter:
        installed = [d for d in installed if d.slug.lower().startswith(slug_filter)]
        if not installed:
            return f"No docset matching '{docset}' is installed. Use /docs list to see installed docsets."

    # ── 1. Collect known tokens from entry names (for splitting concatenated queries) ──
    all_entries: list[tuple[str, dict]] = []  # (slug, entry)
    known_tokens: set[str] = set()

    for ds in installed:
        index = docset_manager.load_index(ds.slug)
        if not index:
            continue
        for entry in index.get("entries", []):
            all_entries.append((ds.slug, entry))
            for tok in _tokenize(entry.get("name", "")):
                known_tokens.add(tok)

    # Try to split concatenated query tokens using known entry tokens.
    # e.g. "stringsprefix" → ["strings", "prefix"] if both are known.
    expanded_keywords: list[str] = []
    for kw in keywords:
        split = _split_concatenated(kw, known_tokens)
        expanded_keywords.extend(split)
    keywords = expanded_keywords if len(expanded_keywords) > len(keywords) else keywords

    # ── 2. Score all entries ──
    candidates: list[tuple[float, str, dict, str]] = []

    for slug, entry in all_entries:
        name = entry.get("name", "")
        path = entry.get("path", "")
        page_key = path.split("#")[0]

        # Quick pre-filter: at least one keyword should match somehow.
        combined = (name + " " + path).lower()
        matched = False
        for kw in keywords:
            if kw in combined:
                matched = True
                break
            if len(kw) >= 6 and _is_subseq(kw, combined):
                matched = True
                break
        if not matched:
            # Fuzzy fallback for near-misses.
            if not any(
                SequenceMatcher(None, kw, name.lower()).ratio() >= 0.6
                for kw in keywords
            ):
                continue

        score = _score_entry(keywords, name, path)
        if score >= MIN_MATCH_SCORE:
            candidates.append((score, slug, entry, page_key))

    if not candidates:
        return f"No results for '{query}'" + (f" in {docset}." if docset else ".")

    # Sort by score, take top candidates.
    candidates.sort(key=lambda c: c[0], reverse=True)
    candidates = candidates[:CANDIDATE_POOL]

    # ── 2. Extract sections, verify, and re-score ──
    results: list[tuple[float, str, str, str]] = []
    total_chars = 0
    db_cache: dict[str, dict] = {}

    for score, slug, entry, page_key in candidates:
        if len(results) >= MAX_RESULTS:
            break

        path = entry.get("path", "")
        anchor = path.split("#", 1)[1] if "#" in path else ""
        name = entry.get("name", path)

        # Load db (cached).
        if slug not in db_cache:
            db_cache[slug] = docset_manager.load_db(slug) or {}
        db = db_cache[slug]

        if page_key not in db:
            continue

        html = db[page_key]
        section_text = (
            _extract_section(html, anchor)
            if anchor
            else _strip_html(html)[:MAX_SECTION_CHARS]
        )

        if not section_text:
            continue

        # Content re-score: boost if keywords appear in extracted text.
        content_hits = _content_keyword_hits(section_text, keywords)
        if content_hits == 0:
            # Drop false positives — name matched but content has none of the keywords.
            continue

        # Boost: each content hit adds up to 0.1 to the score.
        content_boost = min(0.2, content_hits * 0.1)
        final_score = min(1.0, score + content_boost)

        # Budget check.
        header = f"[{len(results)+1}] {name}  ({slug})"
        overhead = len(header) + 2
        remaining = MAX_TOTAL_CHARS - total_chars - overhead
        if remaining <= 100:
            break
        if len(section_text) > remaining:
            section_text = section_text[: max(0, remaining - 3)].rsplit(" ", 1)[0] + "..."

        results.append((final_score, slug, name, section_text))
        total_chars += len(section_text) + overhead

    if not results:
        return (
            f"No relevant results for '{query}'"
            + (f" in {docset}" if docset else "")
            + " — matched entry names but content didn't contain the keywords."
        )

    # ── 3. Sort by final score and format ──
    results.sort(key=lambda r: r[0], reverse=True)

    parts: list[str] = []
    for i, (score, slug, name, text) in enumerate(results, 1):
        parts.append(f"[{i}] {name}  ({slug})")
        parts.append(text)
        parts.append("")

    # Trailing instruction so the model knows to answer, not re-search.
    parts.append("---")
    parts.append("Use the above documentation to answer the user's question. Do not call search_docs again for the same query.")

    return "\n".join(parts).strip()


# ── Ollama/OpenAI tool definition ───────────────────────────────

TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "search_docs",
        "description": (
            "Search installed documentation docsets for a specific function, "
            "class, method, or package. Returns concise code documentation "
            "snippets with syntax and examples.\n\n"
            "TIPS for best results:\n"
            "- Use specific function or class names: 'strings HasPrefix', not 'how to check string start'\n"
            "- Include the package/module when known: 'json dumps', 'crypto aes', 'Array map'\n"
            "- Separate multiple keywords with spaces: 'strings Replace' not 'stringsReplace'\n"
            "- Use the docset parameter to narrow to a language: 'go', 'python', 'javascript'"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Specific function/class/package name to search for. "
                        "Use space-separated keywords: 'strings HasPrefix', 'json dumps', 'Array flatMap'. "
                        "Do NOT concatenate words: use 'strings HasPrefix' not 'stringsHasPrefix'."
                    ),
                },
                "docset": {
                    "type": "string",
                    "description": (
                        "Optional: language/docset to narrow search — 'go', 'python', 'javascript'. "
                        "Omit to search all installed docsets."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}
