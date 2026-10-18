import json

from dataclasses import dataclass
from pathlib import Path


# =========================================================
# Research
# =========================================================
#
# The tech tree (assets/programme/tech.json): what the
# programme knows how to build. Each node costs science
# (earned by experiments and missions, science/record.py)
# and needs the nodes before it; researched, its parts can
# be built. The first nodes cost nothing: a crewed rocket to
# orbit. Science spent on research is not lost to the
# missions that count science banked: they count what was
# earned.
#
# A sandbox knows everything.

TECH_PATH = Path("assets/programme/tech.json")


class TechError(ValueError):
    """A tech tree that cannot be used."""


@dataclass(frozen=True, slots=True)
class TechNode:

    id: str
    name: str
    about: str
    cost: float                     # science
    requires: tuple[str, ...]
    parts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TechTree:

    nodes: tuple[TechNode, ...]

    def node(
        self,
        node_id: str
    ) -> TechNode | None:

        return next((node for node in self.nodes if node.id == node_id), None)

    def node_of(
        self,
        part_id: str
    ) -> TechNode | None:
        """The node that unlocks a part (None: no node: always buildable)."""

        return next((node for node in self.nodes if part_id in node.parts), None)

    def depth(
        self,
        node_id: str
    ) -> int:
        """Steps from the roots (the longest way there): its column in the tree."""

        node = self.node(node_id)

        if node is None or not node.requires:
            return 0

        return 1 + max(self.depth(other) for other in node.requires)

    @property
    def free(
        self
    ) -> set[str]:
        """Nodes known from the start: free, needing nothing."""

        return {node.id for node in self.nodes if node.cost <= 0.0 and not node.requires}


def load_tree(
    path: Path = TECH_PATH,
    parts: dict | None = None
) -> TechTree:
    """The tree; with the part catalog given, every part in it must be one."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))

    nodes = []

    seen_parts: dict[str, str] = {}

    for raw in data.get("nodes", []):

        node_id = raw.get("id")

        if not isinstance(node_id, str) or not node_id:
            raise TechError("a tech node needs an 'id'")

        cost = raw.get("cost", 0.0)

        if isinstance(cost, bool) or not isinstance(cost, (int, float)) or cost < 0.0:
            raise TechError(f"tech node '{node_id}': cost must be a number >= 0")

        node = TechNode(
            id=node_id,
            name=str(raw.get("name", node_id)),
            about=str(raw.get("about", "")),
            cost=float(cost),
            requires=tuple(raw.get("requires", [])),
            parts=tuple(raw.get("parts", []))
        )

        for part_id in node.parts:

            if part_id in seen_parts:
                raise TechError(f"part '{part_id}' is in both '{seen_parts[part_id]}' and '{node_id}'")

            if parts is not None and part_id not in parts:
                raise TechError(f"tech node '{node_id}': part '{part_id}' is not in the catalog")

            seen_parts[part_id] = node_id

        nodes.append(node)

    ids = [node.id for node in nodes]

    if len(set(ids)) != len(ids):
        raise TechError("a tech node is defined twice")

    for node in nodes:
        for other in node.requires:
            if other not in ids:
                raise TechError(f"tech node '{node.id}' needs '{other}', which is not a node")

    tree = TechTree(tuple(nodes))

    # No loops: every node reaches a root.
    def visit(node_id, path):

        if node_id in path:
            raise TechError(f"the tech tree loops through '{node_id}'")

        for other in tree.node(node_id).requires:
            visit(other, path | {node_id})

    for node in nodes:
        visit(node.id, frozenset())

    return tree


# ---------------------------------------------------------
# What is known
# ---------------------------------------------------------

def known(
    tree: TechTree,
    researched
) -> set[str]:
    """Every node known: researched, and the free ones."""

    return set(researched) | tree.free


def state(
    tree: TechTree,
    node: TechNode,
    researched
) -> str:
    """"known", "open" (what it needs is known), or "locked"."""

    have = known(tree, researched)

    if node.id in have:
        return "known"

    if all(other in have for other in node.requires):
        return "open"

    return "locked"


def locked_parts(
    tree: TechTree,
    researched
) -> dict[str, TechNode]:
    """Part id -> the node it waits on, for every part not yet buildable."""

    have = known(tree, researched)

    return {
        part_id: node
        for node in tree.nodes
        if node.id not in have
        for part_id in node.parts
    }


def research(
    tree: TechTree,
    node_id: str,
    researched: set,
    science: float
) -> str:
    """
    Research a node with `science` to spend: "" done (added
    to `researched`), else why not.
    """

    node = tree.node(node_id)

    if node is None:
        return "No such research"

    status = state(tree, node, researched)

    if status == "known":
        return f"{node.name} is known already"

    if status == "locked":

        missing = [tree.node(other).name for other in node.requires if other not in known(tree, researched)]

        return f"{node.name} needs {', '.join(missing)} first"

    if science < node.cost - 1e-9:
        return f"{node.name} needs {node.cost:,.0f} science ({science:,.1f} to spend)"

    researched.add(node.id)

    return ""
