"""Documentation manager — download and cache DevDocs docsets.

Fetches docsets from the DevDocs CDN (documents.devdocs.io) and stores
them in a persistent local cache.  Each docset has:
  - index.json  (search index: entries with name, path, type)
  - db.json     (page content: {path: html_string})
  - meta.json   (slug, name, version, installed date)
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

_CATALOG_URL = "https://devdocs.io/docs.json"
_CDN_BASE = "https://documents.devdocs.io"
_USER_AGENT = "nyx/0.1.0"
_CACHE_DIR = Path.home() / ".local" / "share" / "nyx" / "docs"

# Shared client with retries and Windows-friendly DNS handling.
# Connection pooling avoids repeated getaddrinfo calls and retries
# handles transient DNS failures common on Windows.
_HTTP = httpx.Client(
    headers={"User-Agent": _USER_AGENT},
    timeout=httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0),
    follow_redirects=True,
    transport=httpx.HTTPTransport(retries=2),
)


def _cache_dir() -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


@dataclass
class DocsetMeta:
    """Metadata for an installed docset."""
    slug: str
    name: str
    version: str
    release: str
    installed_at: float
    entry_count: int
    db_size: int  # bytes


def _fetch_catalog() -> list[dict]:
    """Fetch the DevDocs catalog. Returns list of docset dicts."""
    r = _HTTP.get(_CATALOG_URL)
    r.raise_for_status()
    return r.json()


def _fetch_json(url: str) -> dict | list:
    r = _HTTP.get(url)
    r.raise_for_status()
    return r.json()


def install(slug: str) -> tuple[bool, str]:
    """Download a docset to the local cache.

    Returns (success, message).
    """
    # Find the docset in the catalog.
    try:
        catalog = _fetch_catalog()
    except Exception as e:
        return False, f"could not fetch catalog: {e}"

    entry = next((d for d in catalog if d["slug"] == slug), None)
    if entry is None:
        # Try partial match.
        matches = [d for d in catalog if slug in d["slug"]]
        if len(matches) == 1:
            entry = matches[0]
            slug = entry["slug"]
        elif matches:
            names = ", ".join(d["slug"] for d in matches[:10])
            return False, f"ambiguous — did you mean: {names}"
        else:
            return False, f"no docset found for '{slug}'"

    ds_dir = _cache_dir() / slug
    ds_dir.mkdir(parents=True, exist_ok=True)

    # Download index.json.
    try:
        index = _fetch_json(f"{_CDN_BASE}/{slug}/index.json")
    except Exception as e:
        return False, f"could not download index: {e}"

    index_path = ds_dir / "index.json"
    index_path.write_text(json.dumps(index))

    entry_count = len(index.get("entries", [])) if isinstance(index, dict) else 0

    # Download db.json — this is the big one.
    try:
        r = _HTTP.get(f"{_CDN_BASE}/{slug}/db.json")
        r.raise_for_status()
    except Exception as e:
        # Clean up partial state — index.json was already written.
        shutil.rmtree(ds_dir, ignore_errors=True)
        return False, f"could not download docs database: {e}"

    db_path = ds_dir / "db.json"
    db_path.write_bytes(r.content)
    db_size = len(r.content)

    # Write metadata.
    meta = DocsetMeta(
        slug=slug,
        name=entry.get("name", slug),
        version=str(entry.get("version", "")),
        release=str(entry.get("release", "")),
        installed_at=time.time(),
        entry_count=entry_count,
        db_size=db_size,
    )
    meta_path = ds_dir / "meta.json"
    meta_path.write_text(json.dumps({
        "slug": meta.slug,
        "name": meta.name,
        "version": meta.version,
        "release": meta.release,
        "installed_at": meta.installed_at,
        "entry_count": meta.entry_count,
        "db_size": meta.db_size,
    }, indent=2))

    size_mb = db_size / 1024 / 1024
    return True, f"{meta.name} {meta.version} — {entry_count} entries, {size_mb:.1f}MB"


def uninstall(slug: str) -> tuple[bool, str]:
    """Remove a docset from the local cache."""
    ds_dir = _cache_dir() / slug
    if not ds_dir.exists():
        return False, f"'{slug}' is not installed"
    shutil.rmtree(ds_dir)
    return True, f"removed {slug}"


def list_installed() -> list[DocsetMeta]:
    """Return metadata for all installed docsets."""
    result: list[DocsetMeta] = []
    for ds_dir in sorted(_cache_dir().iterdir()):
        if not ds_dir.is_dir():
            continue
        meta_path = ds_dir / "meta.json"
        if not meta_path.exists():
            continue
        try:
            data = json.loads(meta_path.read_text())
            result.append(DocsetMeta(**data))
        except Exception:
            continue
    return result


def list_available(filter: str = "") -> list[dict]:
    """Fetch the catalog and return docsets not yet installed.

    If *filter* is given, only return docsets whose slug contains it.
    """
    installed_slugs = {d.slug for d in list_installed()}
    try:
        catalog = _fetch_catalog()
    except Exception:
        return []

    # Deduplicate by slug — keep only the latest version of each.
    seen: dict[str, dict] = {}
    for d in catalog:
        slug = d["slug"]
        if filter and filter not in slug:
            continue
        if slug in installed_slugs:
            continue
        seen[slug] = d

    return sorted(seen.values(), key=lambda d: d["slug"])


def is_installed(slug: str) -> bool:
    return (_cache_dir() / slug).exists()


def load_index(slug: str) -> dict | None:
    """Load a docset's search index from cache."""
    path = _cache_dir() / slug / "index.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def load_db(slug: str) -> dict | None:
    """Load a docset's page database from cache."""
    path = _cache_dir() / slug / "db.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def open_docset(slug: str) -> tuple[dict, dict] | None:
    """Load index and db for an installed docset from cache.

    Returns (index, db) or None if not installed.
    """
    index = load_index(slug)
    db = load_db(slug)
    if index is None or db is None:
        return None
    return index, db
