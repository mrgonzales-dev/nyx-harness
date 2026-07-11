"""Doc search tool — fuzzy keyword search across installed docsets.

Searches the DevDocs docsets already installed via /docs, using fuzzy
matching on entry names.  Extracts only the relevant section from each
matching page (not the whole page), strips HTML, and trims results to
keep context usage small.

Algorithm:
  1. Tokenize the query into keywords.
  2. Score every entry in every installed docset by fuzzy-matching
     keywords against the entry name (and path).
  3. Take the top N candidates by score.
  4. For each candidate, extract the HTML section identified by the
     anchor in the entry path (from the heading to the next heading).
  5. Strip HTML tags, trim each section to MAX_SECTION_CHARS.
  6. Verify the query terms actually appear in the extracted text —
     drop false positives where the name matched but the content
     doesn't mention the query.
  7. Return formatted results within a total character budget.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from nyx.tools import docs as docset_manager

# ── tuning constants ────────────────────────────────────────────

# Max characters per extracted section.
MAX_SECTION_CHARS = 1500

# Max total characters across all returned results.
MAX_TOTAL_CHARS = 4000

# Maximum number of results to return.
MAX_RESULTS = 3

# Minimum fuzzy match score (0.0–1.0) to consider an entry.
MIN_MATCH_SCORE = 0.35

# Minimum number of query keywords that must appear in the extracted
# content for a result to be kept (filters false positives).
MIN_CONTENT_HITS = 1


# ── HTML utilities ──────────────────────────────────────────────

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_CODE_RE = re.compile(r"<pre[^>]*>(.*?)</pre>", re.DOTALL)
_HEADING_RE = re.compile(r"<h([234])[^>]*>", re.DOTALL)


def _strip_html(html: str) -> str:
    """Convert HTML to plain text, preserving code blocks."""
    # Preserve code blocks with fences.
    def _code_repl(m: re.Match) -> str:
        inner = _TAG_RE.sub("", m.group(1))
        return f"\n```\n{inner.strip()}\n```\n"

    html = _CODE_RE.sub(_code_repl, html)
    # Remove all remaining tags.
    text = _TAG_RE.sub("", html)
    # Collapse whitespace.
    text = _WS_RE.sub(" ", text).strip()
    # Clean up spaces around code fences.
    text = re.sub(r"\s*```\s*", "\n```\n", text)
    return text


def _extract_section(html: str, anchor: str) -> str:
    """Extract the HTML from the heading with id=anchor to the next heading."""
    # Find the heading with the given anchor id.
    pattern = re.compile(
        rf'<h([234])[^>]*id=["\']?{re.escape(anchor)}["\']?', re.IGNORECASE
    )
    m = pattern.search(html)
    if not m:
        # Fallback: try to find the anchor anywhere and grab surrounding content.
        pos = html.find(f'id="{anchor}"')
        if pos == -1:
            pos = html.find(anchor)
        if pos == -1:
            return ""
        # Grab a chunk starting from the anchor.
        chunk = html[pos : pos + MAX_SECTION_CHARS * 3]
        return _strip_html(chunk)[:MAX_SECTION_CHARS]

    heading_level = int(m.group(1))
    start = m.start()

    # Find the next heading of the same or higher level (h2 >= h3 >= h4).
    next_heading = re.compile(rf"<h[2-{heading_level}][^>]*>", re.IGNORECASE)
    rest = html[start + 1 :]
    m2 = next_heading.search(rest)
    if m2:
        section_html = html[start : start + 1 + m2.start()]
    else:
        # No next heading — take to end of page, but cap it.
        section_html = html[start : start + MAX_SECTION_CHARS * 3]

    text = _strip_html(section_html)
    return text[:MAX_SECTION_CHARS]


# ── fuzzy matching ──────────────────────────────────────────────

def _tokenize(query: str) -> list[str]:
    """Split a query into lowercase keyword tokens."""
    return [t for t in re.split(r"[\s/._-]+", query.lower()) if len(t) >= 2]


def _fuzzy_score(keywords: list[str], target: str) -> float:
    """Score how well *target* matches the *keywords*.

    Uses SequenceMatcher for each keyword against the full target and
    each word in the target.  Returns the best average match across
    keywords.
    """
    target_lower = target.lower()
    target_words = re.split(r"[\s/._-]+", target_lower)

    scores: list[float] = []
    for kw in keywords:
        # Best match for this keyword: against full target or any word.
        best = SequenceMatcher(None, kw, target_lower).ratio()
        for word in target_words:
            if len(word) >= 2:
                ratio = SequenceMatcher(None, kw, word).ratio()
                if ratio > best:
                    best = ratio
        scores.append(best)

    return sum(scores) / len(scores) if scores else 0.0


# ── content verification ────────────────────────────────────────

def _content_has_keywords(text: str, keywords: list[str]) -> int:
    """Count how many query keywords appear in the extracted text."""
    text_lower = text.lower()
    hits = 0
    for kw in keywords:
        if kw in text_lower:
            hits += 1
    return hits


# ── main search function ────────────────────────────────────────

def search_docs(query: str) -> str:
    """Search installed docsets for *query* and return relevant snippets.

    Returns a formatted string with the top results, or a message if
    nothing was found or no docsets are installed.
    """
    installed = docset_manager.list_installed()
    if not installed:
        return "No docsets installed. Use /docs install <name> to add one."

    keywords = _tokenize(query)
    if not keywords:
        return "Please provide search keywords."

    # ── 1. Score all entries across all docsets ──
    candidates: list[tuple[float, str, dict, str]] = []  # (score, slug, entry, page_key)

    for ds in installed:
        index = docset_manager.load_index(ds.slug)
        db = docset_manager.load_db(ds.slug)
        if not index or not db:
            continue

        entries = index.get("entries", [])
        for entry in entries:
            name = entry.get("name", "")
            path = entry.get("path", "")
            # The page key is the path before the # anchor.
            page_key = path.split("#")[0]

            # Skip if the page isn't in the db.
            if page_key not in db:
                continue

            score = _fuzzy_score(keywords, name)
            # Boost exact substring matches.
            if all(kw in name.lower() for kw in keywords):
                score = min(1.0, score + 0.3)

            if score >= MIN_MATCH_SCORE:
                candidates.append((score, ds.slug, entry, page_key))

    if not candidates:
        return f"No results for '{query}'."

    # Sort by score descending, take top candidates.
    candidates.sort(key=lambda c: c[0], reverse=True)
    candidates = candidates[: MAX_RESULTS * 3]  # over-fetch, we'll filter

    # ── 2. Extract sections and verify relevance ──
    results: list[tuple[float, str, str, str]] = []  # (score, slug, name, text)
    total_chars = 0
    db_cache: dict[str, dict] = {}  # slug -> db, loaded lazily

    for score, slug, entry, page_key in candidates:
        if len(results) >= MAX_RESULTS:
            break

        path = entry.get("path", "")
        anchor = path.split("#", 1)[1] if "#" in path else ""
        name = entry.get("name", path)

        # Load the db for this docset (cached).
        if slug not in db_cache:
            db_cache[slug] = docset_manager.load_db(slug) or {}
        db = db_cache[slug]

        if page_key not in db:
            continue

        html = db[page_key]
        section_text = _extract_section(html, anchor) if anchor else _strip_html(html)[:MAX_SECTION_CHARS]

        if not section_text:
            continue

        # Verify query terms appear in the content.
        content_hits = _content_has_keywords(section_text, keywords)
        if content_hits < MIN_CONTENT_HITS:
            continue

        # Trim if we'd exceed the total budget (account for header + separator).
        header = f"[{len(results)+1}] {name}  ({slug})"
        overhead = len(header) + 2  # header + newline + blank line separator
        remaining = MAX_TOTAL_CHARS - total_chars - overhead
        if remaining <= 100:
            break
        if len(section_text) > remaining:
            section_text = section_text[:max(0, remaining - 3)].rsplit(" ", 1)[0] + "..."

        results.append((score, slug, name, section_text))
        total_chars += len(section_text) + overhead

    if not results:
        return f"No relevant results for '{query}' (matched entry names but content didn't contain the keywords)."

    # ── 3. Format output ──
    parts: list[str] = []
    for i, (score, slug, name, text) in enumerate(results, 1):
        parts.append(f"[{i}] {name}  ({slug})")
        parts.append(text)
        parts.append("")  # blank line between results

    return "\n".join(parts).strip()


# ── Ollama/OpenAI tool definition ───────────────────────────────

TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "search_docs",
        "description": (
            "Search installed documentation docsets for a keyword or topic. "
            "Returns relevant code documentation snippets. "
            "Use this when the user asks about a library, API, function, or language feature."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords, e.g. 'strings HasPrefix' or 'array map filter'",
                }
            },
            "required": ["query"],
        },
    },
}
