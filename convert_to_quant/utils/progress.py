"""
Throttled tqdm postfix updates.

tqdm's set_postfix() defaults to refresh=True, which forces a display refresh on
every call and bypasses the progress bar's mininterval throttle. Routing postfix
updates through refresh=False lets the normal iteration-driven refresh (which
respects mininterval) render them, cutting per-iteration terminal I/O.
"""


def set_postfix_throttled(pbar, metrics: dict) -> None:
    """Update a tqdm bar's postfix without forcing an immediate display refresh."""
    pbar.set_postfix(metrics, refresh=False)
