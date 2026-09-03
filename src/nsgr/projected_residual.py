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

DEFAULT_RESIDUAL_GMM_TRACE_MODE = "diagnostic"
VALID_RESIDUAL_GMM_TRACE_MODES = frozenset({"diagnostic", "joint"})


def normalize_residual_gmm_trace_mode(value) -> str:
    """Return the configured role of the global trace residual.

    ``diagnostic`` retains the physical trace residual in raw channel
    diagnostics while excluding it from covariance estimation, whitening,
    gradient clipping, and the optimized objective.  ``joint`` preserves the
    legacy trace-first joint covariance objective.
    """

    mode = (
        DEFAULT_RESIDUAL_GMM_TRACE_MODE
        if value is None
        else str(value).strip().lower()
    )
    if mode not in VALID_RESIDUAL_GMM_TRACE_MODES:
        choices = ", ".join(
            repr(choice)
            for choice in sorted(VALID_RESIDUAL_GMM_TRACE_MODES)
        )
        raise ValueError(
            "residual_gmm_trace_mode must be one of "
            f"{choices}; received {value!r}"
        )
    return mode


def normalize_projected_residual_monomials(
    operator_monomials: Sequence[Sequence[int]],
) -> tuple[tuple[int, int], ...]:
    """Validate and freeze the ordered onsite residual basis.

    ``(m,n)`` denotes ``(a^dagger)^m a^n``.  Identity is reserved for the
    automatic trace diagnostic and therefore must not be repeated here.
    Duplicate channels are rejected because they make the active covariance
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
    """Return stable trace-first names for raw diagnostic history."""

    monomials = normalize_projected_residual_monomials(operator_monomials)
    return (
        "loss_residual_gmm_m0_n0",
        *(f"loss_residual_gmm_m{m_power}_n{n_power}" for m_power, n_power in monomials),
    )


def projected_residual_objective_channel_count(
    operator_monomials: Sequence[Sequence[int]],
    *,
    trace_mode: str = DEFAULT_RESIDUAL_GMM_TRACE_MODE,
) -> int:
    """Return the complex channel count used by covariance normalization."""

    monomials = normalize_projected_residual_monomials(operator_monomials)
    mode = normalize_residual_gmm_trace_mode(trace_mode)
    return len(monomials) + int(mode == "joint")


def projected_residual_channel_labels(
    operator_monomials: Sequence[Sequence[int]],
) -> tuple[str, ...]:
    """Return concise labels in persistent raw-diagnostic channel order."""

    monomials = normalize_projected_residual_monomials(operator_monomials)
    return (
        "trace (0,0)",
        *(f"({m_power},{n_power})" for m_power, n_power in monomials),
    )


def projected_residual_summary(
    operator_monomials: Sequence[Sequence[int]],
    *,
    num_site: int,
    trace_mode: str = DEFAULT_RESIDUAL_GMM_TRACE_MODE,
) -> dict:
    """Return serializable metadata for the direct site-local residual basis."""

    monomials = normalize_projected_residual_monomials(operator_monomials)
    site_count = int(num_site)
    if site_count <= 0:
        raise ValueError(f"num_site must be positive; received {num_site!r}")
    mode = normalize_residual_gmm_trace_mode(trace_mode)
    diagnostic_channel_count = 1 + len(monomials)
    objective_channel_count = len(monomials) + int(mode == "joint")
    return {
        "definition": "direct_site_local_adjoint_lindblad_equations",
        "covariance_geometry": (
            "site_average_of_per_site_walker_population_covariances"
        ),
        "trace_mode": mode,
        "trace_in_covariance": mode == "joint",
        "trace_diagnostic_retained": True,
        "operator_monomials": [list(pair) for pair in monomials],
        "physical_channel_count": len(monomials),
        "channel_count_including_trace": diagnostic_channel_count,
        "objective_complex_channel_count": objective_channel_count,
        "real_channel_dimension": 2 * objective_channel_count,
        "residual_site_count": site_count,
    }


__all__ = [
    "CLOSED_NEWTON_COTES_RULES",
    "DEFAULT_RESIDUAL_GMM_TRACE_MODE",
    "VALID_RESIDUAL_GMM_TRACE_MODES",
    "normalize_projected_residual_monomials",
    "normalize_residual_gmm_integrator_nodes",
    "normalize_residual_gmm_trace_mode",
    "projected_residual_channel_labels",
    "projected_residual_objective_channel_count",
    "projected_residual_summary",
    "projected_residual_term_names",
]
