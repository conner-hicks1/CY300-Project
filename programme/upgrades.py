from collections.abc import Callable

from core.logger import Logger


# =========================================================
# Old Saves, Brought Up to Date
# =========================================================
#
# Each saved file says its format ("format": n). As the game
# changes what it keeps, the format goes up, and a step here
# brings a file of the format before up to it; a file is
# taken through every step from its own format to today's as
# it is read (programme/career.py, science/record.py,
# programme/fleet.py, programme/saves.py). A file from a
# newer game than this one is read as well as it can be.
#
# The steps so far:
#
#   career 1 -> 2   careers from before the tech tree had
#                   every part; read as they were, they came
#                   in with nothing researched: those with no
#                   "researched" key, or none researched and
#                   none spent yet having flown parts the tree
#                   holds back, get the whole tree

FORMATS = {
    "career": 2,
    "science": 1,
    "fleet": 1,
    "slot": 1,
}


def _career_tree(
    data: dict,
    context: dict
) -> dict:

    tree = context.get("tree")

    if tree is None:

        from programme.research import load_tree

        tree = load_tree()

    from programme.research import locked_parts

    flown = set(dict(data.get("reliability") or {}))

    before_tree = "researched" not in data or (
        not data.get("researched")
        and not data.get("research_spent")
        and bool(flown & set(locked_parts(tree, set())))
    )

    if before_tree:

        data["researched"] = sorted(node.id for node in tree.nodes)

        Logger.info("[Saves] A career from before the tech tree: every node researched (it had every part).")

    return data


# kind -> {format: its step to the next}
STEPS: dict[str, dict[int, Callable[[dict, dict], dict]]] = {
    "career": {1: _career_tree},
}


def upgrade(
    kind: str,
    data: dict,
    context: dict | None = None
) -> dict:
    """A saved file's data brought to today's format (a copy; context: what the steps may need, e.g. the tech tree)."""

    if not isinstance(data, dict):
        return data

    data = dict(data)

    current = FORMATS[kind]

    try:
        version = int(data.get("format", 1))
    except (TypeError, ValueError):
        version = 1

    if version > current:

        Logger.warning("[Saves] A %s saved by a newer game (format %d, this one knows %d): read as well as it can be.", kind, version, current)

        return data

    steps = STEPS.get(kind, {})

    while version < current:

        step = steps.get(version)

        if step is not None:
            data = step(data, context or {})

        version += 1

    data["format"] = current

    return data
