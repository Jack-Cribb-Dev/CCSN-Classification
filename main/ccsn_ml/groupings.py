"""
Definitions of CCSN class labels and grouping schemes.

This centralizes:
- The full 20-class mechanism list (plus noise).
- Grouped-class schemes (e.g. the 9-class grouping from the dissertation).
- Any future alternative groupings you want to explore.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


# Canonical 20-class ordering used in your FINALS code and paper
MECHANISMS_20: List[str] = [
    "C15-3D",
    "FR",
    "NR",
    "SR",
    "m39",
    "noise",
    "s11",
    "s12",
    "s13",
    "s14",
    "s15",
    "s15fr",
    "s15nr",
    "s15r",
    "s18",
    "s19",
    "s25",
    "s60",
    "s9",
    "y20",
]


@dataclass(frozen=True)
class Grouping:
    """Mapping from group label to list of constituent mechanisms."""

    name: str
    groups: Dict[str, List[str]]


# 9-class grouping from Dir_main/100k/1D_CNN/datasets/gen_grouped.py (C3_GROUPS)
# plus the noise class appended by datasets/add.py.
# Label index is dict order: noise is last (index 8), and s25 sits with the
# other non-rotating Radice models. The saved 1D grouped network has 9 outputs.
GROUPING_9 = Grouping(
    name="grouped_9",
    groups={
        "C15-3D": ["C15-3D"],
        "sR_9_19": ["s9", "s11", "s12", "s13", "s14", "s15", "s19", "s25"],
        "NR": ["NR"],
        "pan_rotating": ["SR", "FR"],
        "Powell": ["m39", "y20", "s18"],
        "s15nr": ["s15nr"],
        "s60": ["s60"],
        "sA_rotating": ["s15r", "s15fr"],
        "noise": ["noise"],
    },
)

N_CLASSES_FULL = 20
N_CLASSES_GROUPED = 9


@dataclass(frozen=True)
class ClassScheme:
    """Resolved label space for either the 20-class or the 9-class task."""

    mode: str
    classes: List[str]
    mechanism_to_class: Dict[str, str]

    @property
    def num_classes(self) -> int:
        return len(self.classes)


def class_scheme(class_mode: str) -> ClassScheme:
    """
    Map a config class_mode onto labels.

    Grouped mode always returns 9 classes. Full mode always returns 20.
    """
    if class_mode in ("20", "full"):
        return ClassScheme(
            mode="20",
            classes=list(MECHANISMS_20),
            mechanism_to_class={name: name for name in MECHANISMS_20},
        )
    if class_mode in ("grouped_9", "9"):
        classes = list(GROUPING_9.groups.keys())
        if len(classes) != N_CLASSES_GROUPED:
            raise RuntimeError(f"Grouped scheme has {len(classes)} labels, expected {N_CLASSES_GROUPED}.")
        return ClassScheme(
            mode="grouped_9",
            classes=classes,
            mechanism_to_class=make_mechanism_to_group_map(MECHANISMS_20, GROUPING_9),
        )
    raise ValueError(f"Unsupported class_mode={class_mode!r}. Use '20' or 'grouped_9'.")


def mechanism_index_map(mechanisms: List[str]) -> Dict[str, int]:
    """Return a mapping from mechanism label to integer index."""
    return {name: i for i, name in enumerate(mechanisms)}


def make_mechanism_to_group_map(mechanisms: List[str], grouping: Grouping) -> Dict[str, str]:
    """
    Build a mechanism -> group-label mapping for a given `Grouping`.

    Any mechanism not present in the grouping will raise a KeyError. This is
    intentional: it forces consistency between label definitions and data.
    """
    mech_to_group: Dict[str, str] = {}
    for group_label, mech_list in grouping.groups.items():
        for mech in mech_list:
            mech_to_group[mech] = group_label

    missing = [m for m in mechanisms if m not in mech_to_group]
    if missing:
        raise KeyError(f"Grouping {grouping.name!r} is missing mechanisms: {missing}")

    return {m: mech_to_group[m] for m in mechanisms}

