"""Git-based update checker for Nyx.

Fetches the remote and compares the local HEAD against the remote
tracking branch.  If the local branch is behind, reports how many
commits behind and the commit subjects.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

# Walk up from this file to find the git root.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class UpdateInfo:
    """Result of an update check."""
    up_to_date: bool
    behind_count: int
    new_commits: list[str]  # commit subjects
    local_sha: str
    remote_sha: str
    error: str | None = None


def _git(*args: str) -> str:
    """Run a git command in the repo root and return stdout."""
    result = subprocess.run(
    ["git", *args],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def check_for_updates() -> UpdateInfo:
    """Fetch remote and compare local vs remote branch.

    Returns an UpdateInfo describing the state.  If anything goes wrong
    (no git, no remote, offline), returns an UpdateInfo with error set.
    """
    try:
        # Get current branch name.
        branch = _git("rev-parse", "--abbrev-ref", "HEAD")

        # Fetch the remote (quiet, no tags).
        _git("fetch", "--quiet")

        # Get local and remote SHAs.
        local_sha = _git("rev-parse", "HEAD")
        remote_sha = _git("rev-parse", f"origin/{branch}")

        if local_sha == remote_sha:
            return UpdateInfo(
                up_to_date=True,
                behind_count=0,
                new_commits=[],
                local_sha=local_sha[:8],
                remote_sha=remote_sha[:8],
            )

        # Count how many commits behind.
        behind = _git("rev-list", "--count", f"HEAD..origin/{branch}")
        behind_count = int(behind) if behind.isdigit() else 0

        # Get the commit subjects we're missing.
        log = _git("log", "--oneline", f"HEAD..origin/{branch}")
        new_commits = log.splitlines() if log else []

        return UpdateInfo(
            up_to_date=False,
            behind_count=behind_count,
            new_commits=new_commits,
            local_sha=local_sha[:8],
            remote_sha=remote_sha[:8],
        )
    except Exception as e:
        return UpdateInfo(
            up_to_date=True,
            behind_count=0,
            new_commits=[],
            local_sha="",
            remote_sha="",
            error=str(e),
        )
