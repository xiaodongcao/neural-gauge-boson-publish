"""Static channel metadata and quadrature rules for projected residuals.

The projected-residual basis is deliberately the configured collection of
onsite normal-ordered monomials only.  Physics coefficients and lattice
connectivity belong to the direct ``L^dagger O`` evaluator in
``dynamics_kernel``; they are not encoded in the channel metadata.
"""

from __future__ import annotations

from collections.abc import Sequence


CLOSED_NEWTON_COTES_RULES = {
    3: ((1.0, 4.0, 1.0), 6.0),
    4: ((1.0, 3.0, 3.0, 1.0), 8.0),
    5: ((7.0, 32.0, 12.0, 32.0, 7.0), 90.0),
    6: ((19.0, 75.0, 50.0, 50.0, 75.0, 19.0), 288.0),
}


def normalize_projected_residual_monomials(
    operator_monomials: Sequence[Sequence[int]],
) -> tuple[tuple[int, int], ...]:
    """Validate and freeze the ordered onsite residual basis.

    ``(m,n)`` denotes ``(a^dagger)^m a^n``.  Identity is supplied by the
    trace-conservation channel and therefore must not be repeated here.
    Duplicate channels are rejected because they make the joint covariance
    singular without adding a physical constraint.
    """

    normalized: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    entries = () if operator_monomials is None else operator_monomials
    for index, pair in enumerate(entries):
        try:
            pair_length = len(pair)
        except TypeError:
            pair_length = -1
        if isinstance(pair, (str, bytes)) or pair_length != 2:
            raise ValueError(
                "operator_monomials entries must be (m,n) integer pairs; "
                f"entry {index} is {pair!r}"
            )
        raw_m, raw_n = pair
        if isinstance(raw_m, bool) or isinstance(raw_n, bool):
            raise ValueError(
                "operator_monomials powers must be nonnegative integers; "
                f"entry {index} is {pair!r}"
            )
        try:
            m_power = int(raw_m)
            n_power = int(raw_n)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "operator_monomials powers must be nonnegative integers; "
                f"entry {index} is {pair!r}"
            ) from exc
        if raw_m != m_power or raw_n != n_power or m_power < 0 or n_power < 0:
            raise ValueError(
                "operator_monomials powers must be nonnegative integers; "
                f"entry {index} is {pair!r}"
            )
        monomial = (m_power, n_power)
        if monomial == (0, 0):
            raise ValueError(
                "operator_monomials must not contain (0,0); trace preservation "
                "is included automatically"
            )
        if monomial in seen:
            raise ValueError(
                "operator_monomials must not contain duplicate channels; "
                f"received {monomial!r} more than once"
            )
        seen.add(monomial)
        normalized.append(monomial)
    if not normalized:
        raise ValueError(
            "operator_monomials must contain at least one nonidentity monomial"
        )
    return tuple(normalized)


def normalize_residual_gmm_integrator_nodes(value) -> int:
    """Return a supported equal-spaced closed Newton--Cotes node count."""

    if isinstance(value, bool):
        raise ValueError(
            "residual_gmm_integrator_nodes must be one of "
            f"{tuple(CLOSED_NEWTON_COTES_RULES)}; received {value!r}"
        )
    try:
        nodes = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "residual_gmm_integrator_nodes must be one of "
            f"{tuple(CLOSED_NEWTON_COTES_RULES)}; received {value!r}"
        ) from exc
    if nodes not in CLOSED_NEWTON_COTES_RULES or value != nodes:
        raise ValueError(
            "residual_gmm_integrator_nodes must be one of "
            f"{tuple(CLOSED_NEWTON_COTES_RULES)}; received {value!r}"
        )
    return nodes


def projected_residual_term_names(
    operator_monomials: Sequence[Sequence[int]],
) -> tuple[str, ...]:
    """Return stable history names in exact covariance-channel order."""

    monomials = normalize_projected_residual_monomials(operator_monomials)
    return (
        "loss_residual_gmm_m0_n0",
        *(f"loss_residual_gmm_m{m_power}_n{n_power}" for m_power, n_power in monomials),
    )


def projected_residual_channel_labels(
    operator_monomials: Sequence[Sequence[int]],
) -> tuple[str, ...]:
    """Return concise physical labels in exact covariance-channel order."""

    monomials = normalize_projected_residual_monomials(operator_monomials)
    return (
        "trace (0,0)",
        *(f"({m_power},{n_power})" for m_power, n_power in monomials),
    )


def projected_residual_summary(
    operator_monomials: Sequence[Sequence[int]],
    *,
    num_site: int,
) -> dict:
    """Return serializable metadata for the direct site-local residual basis."""

    monomials = normalize_projected_residual_monomials(operator_monomials)
    site_count = int(num_site)
    if site_count <= 0:
        raise ValueError(f"num_site must be positive; received {num_site!r}")
    channel_count = 1 + len(monomials)
    return {
        "definition": "direct_site_local_adjoint_lindblad_equations",
        "covariance_geometry": (
            "site_average_of_per_site_walker_population_covariances"
        ),
        "trace_in_covariance": True,
        "operator_monomials": [list(pair) for pair in monomials],
        "physical_channel_count": len(monomials),
        "channel_count_including_trace": channel_count,
        "real_channel_dimension": 2 * channel_count,
        "residual_site_count": site_count,
    }


__all__ = [
    "CLOSED_NEWTON_COTES_RULES",
    "normalize_projected_residual_monomials",
    "normalize_residual_gmm_integrator_nodes",
    "projected_residual_channel_labels",
    "projected_residual_summary",
    "projected_residual_term_names",
]
