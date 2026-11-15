"""
Polarization vocabulary shared by the xradio schemas.

This module holds the casacore ``Stokes`` types, the canonical order of
polarization labels, the circular and linear correlation products, and
:py:func:`check_polarization_order`, which checks that the polarization labels
of a dataset are in canonical order. The polarization coordinate schema,
:py:class:`~xradio.schema.measures.PolarizationArray`, is defined in
:py:mod:`xradio.schema.measures` and re-exported here.
"""

from __future__ import annotations

import xarray

from xradio.schema.check import SchemaIssue, SchemaIssues
from xradio.schema.measures import PolarizationArray  # noqa: F401

#: casacore ``Stokes`` type code -> polarization label, from the
#: ``casacore::Stokes`` class definition.
CASACORE_STOKES_TYPES = {
    0: "Undefined",
    1: "I",
    2: "Q",
    3: "U",
    4: "V",
    5: "RR",
    6: "RL",
    7: "LR",
    8: "LL",
    9: "XX",
    10: "XY",
    11: "YX",
    12: "YY",
    13: "RX",
    14: "RY",
    15: "LX",
    16: "LY",
    17: "XR",
    18: "XL",
    19: "YR",
    20: "YL",
    21: "PP",
    22: "PQ",
    23: "QP",
    24: "QQ",
    25: "RCircular",
    26: "LCircular",
    27: "Linear",
    28: "Ptotal",
    29: "Plinear",
    30: "PFtotal",
    31: "PFlinear",
    32: "Pangle",
}

#: Canonical in-memory polarization order, the casacore ``Stokes`` enum order,
#: which keeps the correlations of a pair of feeds in Jones (2x2 matrix) order,
#: ``RR, RL, LR, LL`` and ``XX, XY, YX, YY``, after ``I, Q, U, V``.
CANONICAL_POLARIZATION_ORDER = (
    "I",
    "Q",
    "U",
    "V",
    "RR",
    "RL",
    "LR",
    "LL",
    "XX",
    "XY",
    "YX",
    "YY",
    "RX",
    "RY",
    "LX",
    "LY",
    "XR",
    "XL",
    "YR",
    "YL",
    "PP",
    "PQ",
    "QP",
    "QQ",
)

_CANONICAL_POLARIZATION_INDEX = {
    label: index for index, label in enumerate(CANONICAL_POLARIZATION_ORDER)
}


def canonical_polarization_order(labels) -> list[int]:
    """Return the permutation that puts polarization labels in canonical order.

    Images keep their polarization axis in the casacore ``Stokes`` order
    (:data:`CANONICAL_POLARIZATION_ORDER`), so that the correlations of a feed
    pair are in Jones matrix order (``RR, RL, LR, LL``; ``XX, XY, YX, YY``).
    FITS cannot store that order for correlations (see
    ``xradio.image._util.conventions.fits_stokes_axis``), so FITS readers use
    this permutation to restore it, reordering every variable with a
    polarization dimension the same way.

    Parameters
    ----------
    labels : sequence of str
        Polarization labels in their current order.

    Returns
    -------
    list of int
        ``order`` such that ``labels[order]`` is in canonical order (the
        identity when the labels already are). Labels outside
        :data:`CANONICAL_POLARIZATION_ORDER` keep their relative order after
        the known ones.
    """
    labels = [str(label) for label in labels]
    unknown = len(CANONICAL_POLARIZATION_ORDER)
    return sorted(
        range(len(labels)),
        key=lambda i: (_CANONICAL_POLARIZATION_INDEX.get(labels[i], unknown), i),
    )


#: Circular correlation products, in canonical (Jones matrix) order.
CIRCULAR_CORRELATIONS = ("RR", "RL", "LR", "LL")

#: Linear correlation products, in canonical (Jones matrix) order.
LINEAR_CORRELATIONS = ("XX", "XY", "YX", "YY")


def check_polarization_order(dataset: xarray.Dataset) -> SchemaIssues:
    """
    Check that the polarization labels of a dataset are in canonical order

    The canonical order is
    :py:data:`~xradio.schema.polarization.CANONICAL_POLARIZATION_ORDER`, the
    order of the casacore ``Stokes`` enumeration (``I, Q, U, V``,
    ``RR, RL, LR, LL``, ``XX, XY, YX, YY``, ...), in which the correlations of
    a pair of feeds are in Jones (2x2) matrix order. Labels that it does not
    list must come after those it lists, in any order among themselves. A
    dataset without a ``polarization`` coordinate, or with one that is not
    one-dimensional (such as the scalar coordinate that selecting one
    polarization leaves, whose dimensions
    :py:func:`~xradio.schema.check.check_dataset` reports), has no issue.

    :param dataset: Dataset to check
    :returns: :py:class:`~xradio.schema.check.SchemaIssues` found: one issue
       at ``coords['polarization']`` whose ``expected`` gives the labels in
       canonical order
    """

    if "polarization" not in dataset.coords or dataset.coords["polarization"].ndim != 1:
        return SchemaIssues()
    labels = [str(label) for label in dataset.coords["polarization"].values]
    order = canonical_polarization_order(labels)
    if order == list(range(len(labels))):
        return SchemaIssues()
    return SchemaIssues(
        [
            SchemaIssue(
                path=[("coords", "polarization")],
                message="Polarization labels must be in canonical (casacore "
                "Stokes, Jones matrix) order, for example RR, RL, LR, LL!",
                found=labels,
                expected=[[labels[i] for i in order]],
            )
        ]
    )
