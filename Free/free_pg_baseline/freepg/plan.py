from dataclasses import dataclass, field
from typing import List, Optional

from .regex_literals import LiteralGroup


@dataclass
class PlanNode:
    node_type: str  # "AND", "OR", "LEAF", "TRUE"
    value: Optional[str] = None
    children: List["PlanNode"] = field(default_factory=list)

    def __repr__(self) -> str:
        if self.node_type == "LEAF":
            return f"LEAF({self.value})"
        if self.node_type == "TRUE":
            return "TRUE"
        return f"{self.node_type}({self.children})"


def build_plan(literals: List[LiteralGroup]) -> PlanNode:
    if not literals:
        return PlanNode("TRUE")
    nodes: List[PlanNode] = []
    for group in literals:
        if isinstance(group, list):
            alts = [PlanNode("LEAF", v) for v in group]
            if not alts:
                continue
            nodes.append(PlanNode("OR", children=alts))
        else:
            nodes.append(PlanNode("LEAF", group))
    if not nodes:
        return PlanNode("TRUE")
    if len(nodes) == 1:
        return nodes[0]
    return simplify(PlanNode("AND", children=nodes))


def simplify(node: PlanNode) -> PlanNode:
    if node.node_type in ("LEAF", "TRUE"):
        return node
    simplified_children = [simplify(c) for c in node.children if c is not None]
    flat: List[PlanNode] = []
    for child in simplified_children:
        if node.node_type == child.node_type and child.node_type in ("AND", "OR"):
            flat.extend(child.children)
        else:
            flat.append(child)
    if node.node_type == "AND":
        flat = [c for c in flat if c.node_type != "TRUE"]
        if not flat:
            return PlanNode("TRUE")
        if len(flat) == 1:
            return flat[0]
    if node.node_type == "OR":
        if any(c.node_type == "TRUE" for c in flat):
            return PlanNode("TRUE")
        if len(flat) == 1:
            return flat[0]
    return PlanNode(node.node_type, children=flat)


def plan_to_str(node: PlanNode) -> str:
    if node.node_type == "LEAF":
        return f"\"{node.value}\""
    if node.node_type == "TRUE":
        return "TRUE"
    sep = " AND " if node.node_type == "AND" else " OR "
    return "(" + sep.join(plan_to_str(c) for c in node.children) + ")"
