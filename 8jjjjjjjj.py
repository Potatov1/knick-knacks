#!/usr/bin/env python3
"""Calculate electronic configurations and shell quantum numbers.

This module intentionally uses only the Python standard library.  The electron
configuration engine performs a baseline Aufbau fill and then applies a small,
weighted state comparison to model common half-filled and filled subshell
stability effects without storing per-element configuration exceptions.
"""

import argparse
import math
import sys


SUBSHELL_LABELS = ("s", "p", "d", "f")
EXTENDED_SUBSHELL_LABELS = "ghiklmnopqrstuvwxyz"

DIGIT_ROOTS = {
    "0": "nil",
    "1": "un",
    "2": "bi",
    "3": "tri",
    "4": "quad",
    "5": "pent",
    "6": "hex",
    "7": "sept",
    "8": "oct",
    "9": "enn",
}

# Required Aufbau / Madelung ordering through the currently known elements.
AUFBAU_ORDER = (
    (1, 0),
    (2, 0),
    (2, 1),
    (3, 0),
    (3, 1),
    (4, 0),
    (3, 2),
    (4, 1),
    (5, 0),
    (4, 2),
    (5, 1),
    (6, 0),
    (4, 3),
    (5, 2),
    (6, 1),
    (7, 0),
    (5, 3),
    (6, 2),
    (7, 1),
)

ELEMENT_SYMBOLS = (
    "",
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
)

NOBLE_GASES = (
    (2, "He"),
    (10, "Ne"),
    (18, "Ar"),
    (36, "Kr"),
    (54, "Xe"),
    (86, "Rn"),
    (118, "Og"),
)

SUPERSCRIPTS = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


class ConfigurationError(ValueError):
    """Raised when a requested quantum chemistry calculation is invalid."""


def subshell_capacity(l_value):
    """Return the Pauli capacity 2(2l + 1) for a subshell."""
    return 2 * (2 * l_value + 1)


def subshell_label(l_value):
    """Return the spectroscopic label for an angular momentum value."""
    if l_value < len(SUBSHELL_LABELS):
        return SUBSHELL_LABELS[l_value]

    index = l_value - len(SUBSHELL_LABELS)
    base = len(EXTENDED_SUBSHELL_LABELS)
    label = ""
    while True:
        label = EXTENDED_SUBSHELL_LABELS[index % base] + label
        index = index // base - 1
        if index < 0:
            return label


def generate_aufbau_order():
    """Yield orbitals in Madelung order for arbitrarily high shells."""
    diagonal = 1
    while True:
        for n_value in range(1, diagonal + 1):
            l_value = diagonal - n_value
            if 0 <= l_value < n_value:
                yield n_value, l_value
        diagonal += 1


def calculate_aufbau(z_value):
    """Return a baseline Aufbau state for atomic number ``z_value``.

    The state is a dictionary keyed by ``(n, l)`` tuples.  Values are electron
    counts.  Insertion order follows the Madelung ``(n + l)`` rule with ties
    broken by the lowest principal quantum number, expanding dynamically for
    theoretical elements beyond the known periodic table.
    """
    validate_atomic_number(z_value)

    remaining = z_value
    state = {}
    for orbital in generate_aufbau_order():
        if remaining <= 0:
            break
        capacity = subshell_capacity(orbital[1])
        electrons = min(remaining, capacity)
        state[orbital] = electrons
        remaining -= electrons

    return state


def evaluate_stability(state):
    """Apply a dynamic half/full-filled stability mutation when favorable.

    The algorithm searches occupied d/f and higher-l subshells in the filled
    state.  If one is exactly one electron short of a half-filled or filled
    subshell, it scores the stabilizing mutation against the cost of promoting
    one electron from the highest occupied s subshell.  The cost scales as 1/n
    for the donor s orbital, modelling shrinking energy gaps in heavier atoms.
    """
    adjusted = dict(state)
    candidates = []

    for orbital, electrons in adjusted.items():
        n_value, l_value = orbital
        if l_value < 2:
            continue

        capacity = subshell_capacity(l_value)
        half_filled = capacity // 2
        target = None
        if electrons == half_filled - 1:
            target = half_filled
        elif electrons == capacity - 1:
            target = capacity

        if target is None:
            continue

        donor = highest_occupied_s_orbital(
            adjusted,
            minimum_n=n_value + l_value - 1,
        )
        if donor is None or adjusted[donor] < 1:
            continue

        bonus = stability_bonus(l_value, target, capacity)
        cost = promotion_cost(donor[0], l_value)
        candidates.append((bonus - cost, bonus, cost, orbital, donor, target))

    if not candidates:
        return adjusted

    best = max(candidates, key=lambda item: item[0])
    score, _bonus, _cost, orbital, donor, target = best
    if score > 0:
        adjusted[orbital] = target
        adjusted[donor] -= 1
        if adjusted[donor] == 0:
            del adjusted[donor]

    return adjusted


def highest_occupied_s_orbital(state, minimum_n):
    """Return the highest occupied s orbital at or above ``minimum_n``."""
    donors = (
        orbital
        for orbital, count in state.items()
        if orbital[1] == 0 and count and orbital[0] >= minimum_n
    )
    return max(donors, key=lambda orbital: orbital[0], default=None)


def stability_bonus(l_value, target, capacity):
    """Return a heuristic bonus for half/full high-l subshell stability."""
    half_target = capacity // 2
    base_bonus = 0.95 if target == half_target else 1.25
    angular_exchange_bonus = 0.18 * math.sqrt(l_value * (l_value + 1))
    return base_bonus + angular_exchange_bonus


def promotion_cost(donor_n, l_value):
    """Return the cost of promoting an s electron into a higher-l subshell."""
    angular_penalty = 0.15 + 0.1 * l_value
    return 3.2 / donor_n + angular_penalty


def format_output(state, shorthand=False):
    """Format an electron configuration state for terminal display."""
    start_index = 0
    prefix = ""
    items = list(state.items())

    if shorthand:
        electron_count = sum(items[index][1] for index in range(len(items)))
        core_z, core_symbol = noble_gas_core(electron_count)
        if core_z:
            prefix = "[{}]".format(core_symbol)
            core_state = calculate_aufbau(core_z)
            core_items = set(core_state.items())
            while start_index < len(items) and items[start_index] in core_items:
                start_index += 1

    pieces = []
    for (n_value, l_value), electrons in items[start_index:]:
        if electrons <= 0:
            continue
        pieces.append(
            "{}{}{}".format(
                n_value,
                subshell_label(l_value),
                str(electrons).translate(SUPERSCRIPTS),
            )
        )

    if prefix and pieces:
        return "{} {}".format(prefix, " ".join(pieces))
    if prefix:
        return prefix
    return " ".join(pieces)


def noble_gas_core(z_value):
    """Return the largest noble gas core smaller than ``z_value``."""
    core = (0, "")
    for gas_z, symbol in NOBLE_GASES:
        if gas_z < z_value:
            core = (gas_z, symbol)
        else:
            break
    return core


def valid_quantum_numbers(n_value):
    """Yield all valid (l, m_l) combinations for a principal shell."""
    validate_principal_quantum_number(n_value)
    for l_value in range(n_value):
        for magnetic_l in range(-l_value, l_value + 1):
            yield l_value, magnetic_l


def validate_atomic_number(z_value):
    """Validate an atomic number for known or theoretical elements."""
    if z_value < 1:
        raise ConfigurationError("Atomic number must be a positive integer.")


def validate_principal_quantum_number(n_value):
    """Validate a principal quantum number."""
    if n_value < 1:
        raise ConfigurationError("Principal quantum number n must be positive.")


def build_parser():
    """Build and return the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Calculate atomic electron configurations and shell quantum numbers."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--z", type=int, help="positive atomic number")
    mode.add_argument("--n", type=int, help="principal quantum number")
    return parser


def parse_args(argv):
    """Parse command-line arguments."""
    return build_parser().parse_args(argv)


def systematic_element_identity(z_value):
    """Return IUPAC systematic name and symbol for a theoretical element."""
    roots = [DIGIT_ROOTS[digit] for digit in str(z_value)]
    name_base = "".join(roots)
    if name_base.endswith("i"):
        name_base = name_base[:-1]
    name = "{}ium".format(name_base).capitalize()
    symbol = "".join(root[0] for root in roots).capitalize()
    return name, symbol


def element_identity(z_value):
    """Return display name and symbol for known or theoretical elements."""
    if z_value < len(ELEMENT_SYMBOLS):
        return ELEMENT_SYMBOLS[z_value], ELEMENT_SYMBOLS[z_value]
    return systematic_element_identity(z_value)


def print_configuration(z_value):
    """Calculate and print element configuration information."""
    baseline = calculate_aufbau(z_value)
    state = evaluate_stability(baseline)
    name, symbol = element_identity(z_value)

    if name == symbol:
        print("Element: {} (Z={})".format(symbol, z_value))
    else:
        print("Element: {} ({}, Z={})".format(name, symbol, z_value))
    print("Full configuration: {}".format(format_output(state)))
    print("Noble gas shorthand: {}".format(format_output(state, shorthand=True)))


def magnetic_range(l_value):
    """Return a compact display string for valid magnetic quantum numbers."""
    if l_value == 0:
        return "0"
    return "-{} to +{}".format(l_value, l_value)


def iter_subshell_summaries(n_value):
    """Yield compact subshell summaries for a principal shell.

    This intentionally walks only the azimuthal quantum number ``l`` instead of
    materializing every individual ``(n, l, m_l)`` state, keeping high ``n``
    output proportional to the number of subshells rather than to ``n ** 2``.
    """
    validate_principal_quantum_number(n_value)
    for l_value in range(n_value):
        yield (
            n_value,
            l_value,
            subshell_label(l_value),
            magnetic_range(l_value),
            subshell_capacity(l_value),
        )


def shell_table_widths(n_value):
    """Return dynamic column widths for a shell summary table."""
    max_l = n_value - 1
    name_width = len("Subshell Name")
    for l_value in range(n_value):
        name_width = max(name_width, len(subshell_label(l_value)))

    return (
        max(len("Shell (n)"), len(str(n_value))),
        max(len("Subshell (l)"), len(str(max_l))),
        name_width,
        max(len("Valid m_l Range") + 1, len(magnetic_range(max_l))),
        max(len("Max Electrons"), len(str(subshell_capacity(max_l)))),
    )


def print_shell_quantum_numbers(n_value):
    """Print shell capacity and compact subshell quantum-number ranges."""
    validate_principal_quantum_number(n_value)
    total_capacity = 2 * n_value**2
    widths = shell_table_widths(n_value)
    shell_width, l_width, name_width, range_width, electrons_width = widths
    header = (
        f"{'Shell (n)':<{shell_width}} | "
        f"{'Subshell (l)':<{l_width}} | "
        f"{'Subshell Name':<{name_width}} | "
        f"{'Valid m_l Range':<{range_width}} | "
        f"{'Max Electrons':<{electrons_width}}"
    )
    separator = "=" * len(header)
    rule = "-" * len(header)

    print(separator)
    print("QUANTUM STATE SUMMARY FOR PRINCIPAL SHELL n = {}".format(n_value))
    print(separator)
    print(header)
    print(rule)
    for shell, l_value, name, ml_range, max_electrons in iter_subshell_summaries(
        n_value
    ):
        print(
            f"{shell:<{shell_width}} | "
            f"{l_value:<{l_width}} | "
            f"{name:<{name_width}} | "
            f"{ml_range:<{range_width}} | "
            f"{max_electrons}"
        )
    print(rule)
    print(
        "TOTAL ELECTRON CAPACITY FOR SHELL n={}: {:,} electrons.".format(
            n_value, total_capacity
        )
    )
    print(separator)


def main(argv=None):
    """CLI entry point."""
    effective_argv = sys.argv[1:] if argv is None else argv
    args = parse_args(effective_argv)

    if args.z is None and args.n is None:
        build_parser().print_help()
        return 0

    try:
        if args.z is not None:
            print_configuration(args.z)
        else:
            print_shell_quantum_numbers(args.n)
    except ConfigurationError as error:
        print("error: {}".format(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
