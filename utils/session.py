"""
Session History
===============
Lightweight JSON-backed store for past code generation runs.
Enables --refine (improve last generation) and --history (list past runs).
"""
import json
import os
from datetime import datetime
from typing import Optional, Dict, Any, List
from utils.logger import logger

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SESSIONS_FILE = os.path.join(_BASE_DIR, "workspace", "sessions.json")

# Maximum number of sessions to keep (oldest are pruned)
_MAX_SESSIONS = 50


def _load_all() -> List[Dict[str, Any]]:
    """Load all sessions from disk. Returns empty list on any error."""
    if not os.path.exists(_SESSIONS_FILE):
        return []
    try:
        with open(_SESSIONS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception as e:
        logger.warning(f"Could not read sessions file: {e}")
        return []


def _save_all(sessions: List[Dict[str, Any]]) -> None:
    """Persist session list to disk."""
    os.makedirs(os.path.dirname(_SESSIONS_FILE), exist_ok=True)
    try:
        with open(_SESSIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(sessions, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Could not write sessions file: {e}")


def save_session(
    prompt: str,
    enhanced_prompt: Optional[str],
    plan: Optional[str],
    code: str,
    output_path: str,
) -> None:
    """
    Append a completed generation run to the session store.

    Args:
        prompt: Original user prompt.
        enhanced_prompt: Engineered prompt from Interpreter agent (None in fast mode).
        plan: Implementation plan from Planner agent (None in fast mode).
        code: Final generated Python code.
        output_path: Path where the code was saved.
    """
    sessions = _load_all()
    entry: Dict[str, Any] = {
        "id": len(sessions) + 1,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "prompt": prompt,
        "enhanced_prompt": enhanced_prompt,
        "plan": plan,
        "code": code,
        "output_path": output_path,
    }
    sessions.append(entry)
    # Prune oldest if over limit
    if len(sessions) > _MAX_SESSIONS:
        sessions = sessions[-_MAX_SESSIONS:]
    _save_all(sessions)
    logger.info(f"Session #{entry['id']} saved to {_SESSIONS_FILE}")


def load_last_session() -> Optional[Dict[str, Any]]:
    """
    Return the most recent session, or None if no sessions exist.
    """
    sessions = _load_all()
    if not sessions:
        logger.warning("No previous sessions found.")
        return None
    return sessions[-1]


def print_history() -> None:
    """
    Print a formatted list of all past sessions to stdout.
    """
    sessions = _load_all()
    if not sessions:
        print("No generation history found.")
        return

    print("\n" + "=" * 60)
    print(f"  Generation History ({len(sessions)} session(s))")
    print("=" * 60)
    for s in reversed(sessions):
        prompt_preview = s.get("prompt", "")[:60].replace("\n", " ")
        path = s.get("output_path", "N/A")
        print(f"  #{s['id']:>3}  [{s['timestamp']}]  {prompt_preview!r}")
        print(f"        Saved to: {path}")
        print()
    print("=" * 60 + "\n")
