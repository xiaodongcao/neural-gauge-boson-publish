from .analytical_gauge import ANALYTICAL_GAUGE_MODES, validate_analytical_gauge_mode
from .lib_preinclude import *
from .projected_residual import (
    DEFAULT_RESIDUAL_GMM_TRACE_MODE,
    normalize_residual_gmm_trace_mode,
)
import math
import re
import shutil
from pathlib import Path


NEURAL_GAUGE_MODES = {"neural_graph", "neural_mlp", "neural_cnn"}
VALID_GAUGE_MODES = {*NEURAL_GAUGE_MODES, *ANALYTICAL_GAUGE_MODES}
VALID_NEURAL_GAUGE_COMPONENTS = {"both", "drift", "diffusion"}
DEFAULT_SDE_SOLVER = "semi_implicit_midpoint"
EXACT_LOCAL_SDE_SOLVER = "interaction_picture_exact_local"
VALID_SDE_SOLVERS = {
    DEFAULT_SDE_SOLVER,
    EXACT_LOCAL_SDE_SOLVER,
    "interaction_picture_implicit_midpoint",
}
SDE_SOLVER_CONTROL_NAMES = (
    "max_iterations",
    "root_rtol",
    "root_atol",
    "affine_expm_order",
    "affine_expm_substeps",
    "newton_damping_steps",
)
SDE_SOLVER_ACTIVE_CONTROLS = {
    DEFAULT_SDE_SOLVER: ("max_iterations",),
    "interaction_picture_implicit_midpoint": SDE_SOLVER_CONTROL_NAMES,
    EXACT_LOCAL_SDE_SOLVER: (
        "affine_expm_order",
        "affine_expm_substeps",
    ),
}
LOSS_TERM_TRACE_THRESHOLD = 1.0e-8
AUTO_MONOMIAL_SELECTOR_VALUES = {"auto"}
PARETO_K_MONOMIAL_MAX_ORDER = 6
OPERATOR_MOMENT_DEFAULT_ORDER = 4
OPERATOR_MOMENT_MAX_ORDER = 6
PARETO_K_MONOMIAL_SPECS = tuple(
    (
        total_order,
        m_power,
        total_order - m_power,
        f"loss_pareto_k_m{m_power}_n{total_order - m_power}",
    )
    for total_order in range(1, PARETO_K_MONOMIAL_MAX_ORDER + 1)
    for m_power in range(total_order + 1)
)
PARETO_K_MONOMIAL_TERMS = tuple(spec[3] for spec in PARETO_K_MONOMIAL_SPECS)
OPERATOR_MOMENT_SPECS = tuple(
    (
        total_order,
        m_power,
        total_order - m_power,
        f"loss_op_m{m_power}_n{total_order - m_power}",
    )
    for total_order in range(1, OPERATOR_MOMENT_MAX_ORDER + 1)
    for m_power in range(total_order + 1)
)
PARETO_K_OBSERVABLE_SPECS = tuple(
    (
        total_order,
        m_power,
        total_order - m_power,
        f"pareto-k_m{m_power}_n{total_order - m_power}",
    )
    for total_order in range(1, PARETO_K_MONOMIAL_MAX_ORDER + 1)
    for m_power in range(total_order + 1)
)
PARETO_K_OBSERVABLE_NAMES = tuple(
    name
    for _total_order, _m_power, _n_power, base_name in PARETO_K_OBSERVABLE_SPECS
    for name in (base_name, f"{base_name}_mean", f"{base_name}_max")
)


def _monomial_terms_at_order(
    specs: Sequence[Tuple[int, int, int, str]],
    total_order: int,
) -> tuple[str, ...]:
    return tuple(term for order, _m_power, _n_power, term in specs if int(order) == int(total_order))


def _expand_loss_ema_term_alias(term: str, auto_terms: Optional[Dict[str, Sequence[str]]] = None) -> tuple[str, ...]:
    key = str(term).strip()
    auto_terms = auto_terms or {}
    auto_aliases = {"loss_pareto_k_auto": "loss_pareto_k"}
    if key in auto_aliases:
        expanded = tuple(str(term) for term in auto_terms.get(auto_aliases[key], ()))
        if not expanded:
            raise ValueError(
                f"training.EMA.terms alias {key!r} requires the corresponding auto monomial selector"
            )
        return expanded
    aliases = {
        "loss_pareto_k_p": (PARETO_K_MONOMIAL_SPECS, PARETO_K_MONOMIAL_MAX_ORDER, "loss_pareto_k"),
    }
    for prefix, (specs, max_order, label) in aliases.items():
        if not key.startswith(prefix):
            continue
        raw_order = key[len(prefix):]
        if not raw_order.isdigit():
            break
        total_order = int(raw_order)
        if total_order < 1 or total_order > int(max_order):
            raise ValueError(
                f"training.EMA.terms alias {key!r} asks for order {total_order}, "
                f"but {label} supports orders 1..{int(max_order)}"
            )
        expanded = _monomial_terms_at_order(specs, total_order)
        if not expanded:
            raise ValueError(f"training.EMA.terms alias {key!r} did not match any monomial terms")
        return expanded
    return (key,)


def expand_loss_ema_terms(terms, auto_terms: Optional[Dict[str, Sequence[str]]] = None) -> tuple[str, ...]:
    """Expand compact Pareto-k EMA aliases to explicit monomial terms."""
    if terms is None:
        return ()
    if isinstance(terms, str):
        terms = [part.strip() for part in terms.split(",") if part.strip()]
    expanded_terms = []
    for term in terms:
        for expanded in _expand_loss_ema_term_alias(str(term), auto_terms=auto_terms):
            if expanded not in expanded_terms:
                expanded_terms.append(expanded)
    return tuple(expanded_terms)


def resolve_pareto_k_tail_count(
    num_walker: int,
    tail_fraction: float,
    min_tail_count: int,
) -> int:
    """Resolve the static PSIS-style upper-tail count used by Pareto-k fits."""

    walker_count = max(2, int(num_walker))
    fraction_count = int(np.floor(float(tail_fraction) * walker_count))
    sqrt_cap_count = int(np.floor(3.0 * np.sqrt(float(walker_count))))
    psis_count = min(fraction_count, sqrt_cap_count)
    requested = max(int(min_tail_count), psis_count)
    return max(1, min(requested, walker_count - 1))


class KeyGenerator:
    """Deterministic PRNG splitter shared by training and simulation code."""

    def __init__(self, seed: int):
        self._key = random.PRNGKey(int(seed))

    def next(self, fold_in_value: Optional[int] = None):
        self._key, subkey = random.split(self._key)
        if fold_in_value is not None:
            subkey = random.fold_in(subkey, int(fold_in_value))
        return subkey


def to_scalar_float(x) -> float:
    return float(np.asarray(x))


def format_end_time(t0: float, dt: float, n_steps: int, n_windows: int) -> str:
    return f"{float(t0) + float(dt) * int(n_steps) * int(n_windows):.8g}"


def archive_extension(archive_format: str) -> str:
    archive_format = str(archive_format).strip().lower()
    if archive_format == "npz":
        return ".npz"
    if archive_format == "zarr":
        return ".zarr"
    raise ValueError(f"Unsupported archive format '{archive_format}'")


def infer_archive_format(path: str, archive_format: Optional[str] = None) -> str:
    if archive_format is not None:
        archive_format = str(archive_format).strip().lower()
        if archive_format not in {"npz", "zarr"}:
            raise ValueError(f"Unsupported archive format '{archive_format}'")
        return archive_format

    suffix = Path(path).suffix.lower()
    if suffix == ".npz":
        return "npz"
    if suffix == ".zarr":
        return "zarr"
    raise ValueError(
        f"Could not infer archive format from path '{path}'. Use a '.npz' or '.zarr' suffix, "
        "or pass archive_format explicitly."
    )


def _import_zarr_runtime():
    try:
        import zarr
    except ImportError as exc:
        raise ImportError(
            "Zarr storage was requested, but the 'zarr' package is not installed in this Python environment."
        ) from exc

    try:
        from numcodecs import Blosc
    except ImportError:
        Blosc = None
    return zarr, Blosc


def require_archive_backend(archive_format: str):
    archive_format = infer_archive_format(f"dummy{archive_extension(archive_format)}", archive_format=archive_format)
    if archive_format == "zarr":
        _import_zarr_runtime()
    return archive_format


def _default_zarr_chunks(array: np.ndarray):
    arr = np.asarray(array)
    if arr.ndim == 0:
        return None
    if arr.ndim == 1:
        return (min(arr.shape[0], 8192),)
    if arr.ndim == 2:
        return (min(arr.shape[0], 1), min(arr.shape[1], 4096))
    if arr.ndim == 3:
        return (min(arr.shape[0], 1), min(arr.shape[1], 2048), min(arr.shape[2], 64))
    chunks = [min(arr.shape[0], 1)]
    chunks.extend(min(dim, 256) for dim in arr.shape[1:])
    return tuple(chunks)


def ensure_parent_dir(path_like) -> Path:
    """Create a file/archive parent directory when the path actually has one."""

    path = Path(path_like).expanduser()
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _path_contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_recursive_delete_target(path_like) -> Path:
    """Reject broad recursive-delete targets used by clean starts/archives."""

    resolved = Path(path_like).expanduser().resolve()
    cwd = Path.cwd().resolve()
    home = Path.home().resolve()
    if resolved.parent == resolved:
        raise ValueError(
            f"Refusing recursive deletion of filesystem root '{resolved}'"
        )
    if resolved == home:
        raise ValueError(
            f"Refusing recursive deletion of home directory '{resolved}'"
        )
    if _path_contains(resolved, cwd):
        raise ValueError(
            "Refusing recursive deletion of the current working directory or "
            f"one of its ancestors: '{resolved}'"
        )
    return resolved


def _save_zarr(path: str, arrays: Dict[str, Any], compressed: bool = False):
    zarr, Blosc = _import_zarr_runtime()
    compressor = None
    if compressed and Blosc is not None:
        compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)

    path_obj = ensure_parent_dir(path)
    if path_obj.exists():
        remove_folders(path_obj)

    group = zarr.open_group(str(path_obj), mode="w")
    group.attrs["archive_format"] = "zarr"
    group.attrs["array_names"] = sorted(arrays.keys())

    for name, value in arrays.items():
        arr = np.asarray(value)
        create_kwargs = {
            "shape": arr.shape,
            "dtype": arr.dtype,
            "chunks": _default_zarr_chunks(arr),
            "overwrite": True,
            "data": arr,
        }
        if compressor is not None:
            create_kwargs["compressor"] = compressor
        try:
            group.create_dataset(name, **create_kwargs)
        except TypeError:
            create_kwargs.pop("compressor", None)
            group.create_dataset(name, **create_kwargs)


def load_array_archive(path: str, archive_format: Optional[str] = None):
    archive_format = infer_archive_format(path, archive_format=archive_format)
    if archive_format == "npz":
        with np.load(path, allow_pickle=False) as raw:
            return {name: np.asarray(raw[name]) for name in raw.files}

    zarr, _ = _import_zarr_runtime()
    group = zarr.open_group(str(path), mode="r")
    if hasattr(group, "array_keys"):
        keys = list(group.array_keys())
    else:
        keys = [name for name in group.keys() if hasattr(group[name], "shape")]
    return {name: np.asarray(group[name]) for name in keys}


def remove_folders(path_like):
    path = Path(path_like)
    if path.exists():
        _validate_recursive_delete_target(path)
        shutil.rmtree(path)


def tree_has_nonfinite(tree):
    return jax.tree_util.tree_reduce(
        lambda acc, x: acc | jnp.any(~jnp.isfinite(x)),
        tree,
        False,
    )


def _require_keys(section_name: str, section: Dict[str, Any], required_keys: Sequence[str]):
    missing = [key for key in required_keys if key not in section]
    if missing:
        raise ValueError(f"{section_name} is missing required keys: {missing}")


def _config_int(
    value,
    option_name: str,
    *,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    """Parse one integer config value without silently truncating fractions."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{option_name} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{option_name} must be an integer") from exc
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value) or float(parsed) != float(value):
            raise ValueError(f"{option_name} must be an integer")
    elif isinstance(value, str) and value.strip() != str(parsed):
        raise ValueError(f"{option_name} must be an integer")
    if minimum is not None and parsed < int(minimum):
        raise ValueError(f"{option_name} must be >= {int(minimum)}")
    if maximum is not None and parsed > int(maximum):
        raise ValueError(f"{option_name} must be <= {int(maximum)}")
    return parsed


def _config_float(
    value,
    option_name: str,
    *,
    minimum: Optional[float] = None,
    strictly_positive: bool = False,
) -> float:
    """Parse one finite real config value and enforce its lower bound."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{option_name} must be a finite real number")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{option_name} must be a finite real number") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{option_name} must be a finite real number")
    if strictly_positive and parsed <= 0.0:
        raise ValueError(f"{option_name} must be finite and > 0")
    if minimum is not None and parsed < float(minimum):
        raise ValueError(f"{option_name} must be finite and >= {float(minimum):g}")
    return parsed


def _config_bool(value, option_name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{option_name} must be a boolean")
    return bool(value)


def _config_path_string(value, option_name: str) -> str:
    """Normalize one nonempty string/path-like config value."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (str, os.PathLike),
    ):
        raise ValueError(f"{option_name} must be a nonempty string or path-like value")
    try:
        normalized = os.fspath(value)
    except TypeError as exc:
        raise ValueError(
            f"{option_name} must be a nonempty string or path-like value"
        ) from exc
    if not isinstance(normalized, str) or not normalized.strip():
        raise ValueError(f"{option_name} must be a nonempty string or path-like value")
    return normalized


def loss_prefactor_is_active(value) -> bool:
    """Return whether a loss prefactor is large enough to instantiate its branch."""

    return abs(float(value)) >= LOSS_TERM_TRACE_THRESHOLD


def _validate_multi_device_config(section: Dict[str, Any], section_name: str):
    raw = section.get("multi_device")
    if raw is None:
        return
    if not isinstance(raw, dict):
        raise ValueError(f"{section_name}.multi_device must be a dict when provided")
    unsupported = sorted(set(raw) - {"enabled", "num_devices", "axis_name"})
    if unsupported:
        raise ValueError(f"Unsupported {section_name}.multi_device options: {unsupported}")
    if "enabled" in raw:
        raw["enabled"] = _config_bool(raw["enabled"], f"{section_name}.multi_device.enabled")
    requested = raw.get("num_devices", "auto")
    if requested is not None and str(requested).strip().lower() != "auto":
        raw["num_devices"] = _config_int(
            requested,
            f"{section_name}.multi_device.num_devices",
            minimum=1,
        )
    if "axis_name" in raw:
        axis_name = str(raw["axis_name"]).strip()
        if not axis_name:
            raise ValueError(f"{section_name}.multi_device.axis_name cannot be empty")
        raw["axis_name"] = axis_name


def _normalize_training_quantity_selector(
    training_cfg: Dict[str, Any],
    *,
    prefix: str,
    default_order: int = 6,
    default_mode: str = "upto",
) -> tuple[int, str]:
    order_value = training_cfg.get(f"{prefix}_applied_quantities", default_order)
    try:
        order = _config_int(
            order_value,
            f"training.{prefix}_applied_quantities",
        )
    except ValueError as exc:
        raise ValueError(f"training.{prefix}_applied_quantities must be an integer from 1 to 6") from exc
    if order < 1 or order > 6:
        raise ValueError(f"training.{prefix}_applied_quantities must be an integer from 1 to 6")
    mode = (
        str(training_cfg.get(f"{prefix}_applied_quantities_mode", default_mode))
        .strip()
        .lower()
        .replace("-", "_")
    )
    if mode not in {"upto", "exact"}:
        raise ValueError(f"training.{prefix}_applied_quantities_mode must be 'upto' or 'exact'")
    training_cfg[f"{prefix}_applied_quantities"] = order
    training_cfg[f"{prefix}_applied_quantities_mode"] = mode
    return order, mode


def _parse_monomial_pair(item, option_name: str) -> tuple[int, int]:
    if isinstance(item, dict):
        if "m" not in item or "n" not in item:
            raise ValueError(f"{option_name} dict entries must contain keys 'm' and 'n'")
        raw_m, raw_n = item["m"], item["n"]
    elif isinstance(item, str):
        numbers = re.findall(r"-?\d+", item)
        if len(numbers) != 2:
            raise ValueError(
                f"{option_name} string entries must contain exactly two non-negative integers; "
                f"got {item!r}"
            )
        raw_m, raw_n = numbers
    elif isinstance(item, (list, tuple)) and len(item) == 2:
        raw_m, raw_n = item
    else:
        raise ValueError(
            f"{option_name} entries must be [m, n], {{'m': m, 'n': n}}, "
            f"or a string like 'm1_n0'; got {item!r}"
        )

    try:
        m_power = _config_int(raw_m, f"{option_name} m power")
        n_power = _config_int(raw_n, f"{option_name} n power")
    except ValueError as exc:
        raise ValueError(f"{option_name} entries must contain integer powers; got {item!r}") from exc
    return m_power, n_power


def is_auto_monomial_selector(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower().replace("-", "_") in AUTO_MONOMIAL_SELECTOR_VALUES
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return is_auto_monomial_selector(value[0])
    return False


def normalize_monomial_pairs(
    value,
    *,
    option_name: str,
    max_order: int,
    min_order: int = 1,
) -> tuple[tuple[int, int], ...]:
    """Normalize an explicit onsite monomial selector.

    JSON configs should use ``[[m, n], ...]``.  For hand-written scripts this
    also accepts ``{"m": m, "n": n}`` and strings such as ``"m1_n0"``.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ()
        if ";" in stripped:
            entries = [part.strip() for part in stripped.split(";") if part.strip()]
        else:
            entries = [stripped]
    elif isinstance(value, (list, tuple)):
        if len(value) == 0:
            return ()
        if len(value) == 2 and all(isinstance(x, (int, np.integer)) for x in value):
            entries = [value]
        else:
            entries = list(value)
    else:
        entries = [value]

    normalized = []
    seen = set()
    for entry in entries:
        pair = _parse_monomial_pair(entry, option_name)
        m_power, n_power = pair
        total_order = m_power + n_power
        if (
            m_power < 0
            or n_power < 0
            or total_order < int(min_order)
            or total_order > int(max_order)
        ):
            raise ValueError(
                f"{option_name} entries must satisfy m>=0, n>=0, and "
                f"{int(min_order)} <= m+n <= {int(max_order)}; "
                f"got {(m_power, n_power)}"
            )
        if pair in seen:
            continue
        seen.add(pair)
        normalized.append(pair)
    return tuple(normalized)


def selected_monomial_specs(
    specs: Sequence[Tuple[int, int, int, str]],
    applied_quantities: int,
    applied_quantities_mode: str,
    monomials=(),
) -> tuple[tuple[int, int, int, str], ...]:
    """Select monomial specs by explicit pairs, or by the old total-order rule."""
    explicit_pairs = tuple((int(m), int(n)) for m, n in (monomials or ()))
    if explicit_pairs:
        spec_by_pair = {
            (int(m_power), int(n_power)): (
                int(total_order),
                int(m_power),
                int(n_power),
                term,
            )
            for total_order, m_power, n_power, term in specs
        }
        missing = [pair for pair in explicit_pairs if pair not in spec_by_pair]
        if missing:
            raise ValueError(f"explicit monomial selector contains unsupported pairs: {missing}")
        return tuple(spec_by_pair[pair] for pair in explicit_pairs)

    order = int(applied_quantities)
    mode = str(applied_quantities_mode).strip().lower().replace("-", "_")
    if mode not in {"upto", "exact"}:
        raise ValueError("applied_quantities_mode must be 'upto' or 'exact'")
    return tuple(
        (int(total_order), int(m_power), int(n_power), term)
        for total_order, m_power, n_power, term in specs
        if (int(total_order) == order if mode == "exact" else int(total_order) <= order)
    )


def selected_monomial_terms(
    specs: Sequence[Tuple[int, int, int, str]],
    applied_quantities: int,
    applied_quantities_mode: str,
    monomials=(),
) -> tuple[str, ...]:
    return tuple(
        term
        for _total_order, _m_power, _n_power, term in selected_monomial_specs(
            specs,
            applied_quantities,
            applied_quantities_mode,
            monomials,
        )
    )


def onsite_monomials_in_operator_equations(
    operator_monomials: Sequence[Tuple[int, int]],
    *,
    max_order: int = PARETO_K_MONOMIAL_MAX_ORDER,
) -> tuple[tuple[int, int], ...]:
    """Return non-identity onsite moments appearing in selected equations."""
    selected = []
    seen = set()

    def add_pair(m_power: int, n_power: int):
        pair = (int(m_power), int(n_power))
        total_order = pair[0] + pair[1]
        if total_order < 1 or total_order > int(max_order):
            return
        if pair not in seen:
            seen.add(pair)
            selected.append(pair)

    for m_power, n_power in operator_monomials:
        m_power = int(m_power)
        n_power = int(n_power)
        add_pair(m_power, n_power)
        if m_power != n_power:
            add_pair(m_power + 1, n_power + 1)
        if n_power > 0:
            add_pair(m_power, n_power - 1)
        if m_power > 0:
            add_pair(m_power - 1, n_power)
    return tuple(selected)


def _normalize_training_monomial_selector(
    training_cfg: Dict[str, Any],
    *,
    prefix: str,
    specs: Sequence[Tuple[int, int, int, str]],
    max_order: int,
    default_order: int = 6,
    default_mode: str = "upto",
    default_monomials=(),
    allow_auto: bool = False,
    auto_monomials=(),
) -> tuple[int, str, tuple[tuple[int, int], ...]]:
    monomial_key = f"{prefix}_monomials"
    quantity_key = f"{prefix}_applied_quantities"
    mode_key = f"{prefix}_applied_quantities_mode"
    raw_quantity = training_cfg.get(quantity_key, default_order)
    if is_auto_monomial_selector(raw_quantity):
        if not allow_auto:
            raise ValueError(f"training.{quantity_key} does not support 'auto'")
        if not auto_monomials:
            raise ValueError(f"training.{quantity_key}='auto' requires selected operator moments")
        training_cfg[monomial_key] = [[m_power, n_power] for m_power, n_power in auto_monomials]
        training_cfg[quantity_key] = max(m_power + n_power for m_power, n_power in auto_monomials)
        training_cfg.setdefault(mode_key, "exact")
        raw_quantity = training_cfg[quantity_key]

    quantity_is_explicit_selector = (
        monomial_key not in training_cfg
        and not isinstance(raw_quantity, (int, np.integer))
        and not (isinstance(raw_quantity, str) and raw_quantity.strip().isdigit())
    )
    if quantity_is_explicit_selector:
        monomials_from_quantity = normalize_monomial_pairs(
            raw_quantity,
            option_name=f"training.{quantity_key}",
            max_order=max_order,
        )
        if not monomials_from_quantity:
            raise ValueError(f"training.{quantity_key} explicit monomial selector cannot be empty")
        training_cfg[monomial_key] = [[m_power, n_power] for m_power, n_power in monomials_from_quantity]
        training_cfg[quantity_key] = max(m_power + n_power for m_power, n_power in monomials_from_quantity)
        training_cfg.setdefault(mode_key, "exact")

    order, mode = _normalize_training_quantity_selector(
        training_cfg,
        prefix=prefix,
        default_order=default_order,
        default_mode=default_mode,
    )
    raw_monomials = training_cfg.get(monomial_key, default_monomials)
    if is_auto_monomial_selector(raw_monomials):
        if not allow_auto:
            raise ValueError(f"training.{monomial_key} does not support 'auto'")
        if not auto_monomials:
            raise ValueError(f"training.{monomial_key}='auto' requires selected operator moments")
        raw_monomials = auto_monomials
    monomials = normalize_monomial_pairs(
        raw_monomials,
        option_name=f"training.{monomial_key}",
        max_order=max_order,
    )
    if monomials:
        selected_monomial_specs(specs, order, mode, monomials)
        training_cfg[monomial_key] = [[m_power, n_power] for m_power, n_power in monomials]
    elif monomial_key in training_cfg:
        training_cfg[monomial_key] = []
    return order, mode, monomials


def normalize_neural_gauge_components(value: str = "both") -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "all": "both",
        "full": "both",
        "drift_only": "drift",
        "diffusion_only": "diffusion",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in VALID_NEURAL_GAUGE_COMPONENTS:
        raise ValueError("neural_gauge_components must be 'both', 'drift', or 'diffusion'")
    return normalized


def normalize_sde_solver(value: str = DEFAULT_SDE_SOLVER) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in VALID_SDE_SOLVERS:
        valid = ", ".join(sorted(VALID_SDE_SOLVERS))
        raise ValueError(f"sde_solver must be one of: {valid}")
    return normalized


def sde_solver_control_metadata(
    solver_name: str,
    *,
    max_iterations: int,
    root_rtol: float,
    root_atol: float,
    affine_expm_order: int,
    affine_expm_substeps: int,
    newton_damping_steps: int,
) -> Dict[str, Any]:
    """Return configured solver controls with explicit applicability metadata.

    All control values remain in the metadata schema for backward-compatible
    provenance.  ``active_controls`` is authoritative about which values can
    affect the selected backend.
    """

    normalized = normalize_sde_solver(solver_name)
    configured = {
        "max_iterations": int(max_iterations),
        "root_rtol": float(root_rtol),
        "root_atol": float(root_atol),
        "affine_expm_order": int(affine_expm_order),
        "affine_expm_substeps": int(affine_expm_substeps),
        "newton_damping_steps": int(newton_damping_steps),
    }
    active = tuple(SDE_SOLVER_ACTIVE_CONTROLS[normalized])
    metadata = {
        "name": normalized,
        **configured,
        "active_controls": list(active),
        "inactive_controls": [
            name for name in SDE_SOLVER_CONTROL_NAMES if name not in active
        ],
    }
    if normalized == EXACT_LOCAL_SDE_SOLVER:
        metadata.update(
            {
                "scheme": "affine_kerr_stochastic_local_split",
                "stochastic_convention": "prepoint_ito_with_frozen_gauge_fields",
                "order_claim": (
                    "no_claim_above_weak_1_or_strong_1/2_without_coupled_dt_validation"
                ),
            }
        )
    return metadata


def format_sde_solver_controls(metadata: Dict[str, Any]) -> str:
    """Format only active solver controls, followed by inactive control names."""

    labels = {
        "max_iterations": lambda value: f"max_iterations={int(value)}",
        "root_rtol": lambda value: f"root_rtol={float(value):g}",
        "root_atol": lambda value: f"root_atol={float(value):g}",
        "affine_expm_order": lambda value: f"affine_expm_order={int(value)}",
        "affine_expm_substeps": lambda value: f"affine_expm_substeps={int(value)}",
        "newton_damping_steps": lambda value: f"newton_damping_steps={int(value)}",
    }
    active = metadata["active_controls"]
    active_text = ", ".join(labels[name](metadata[name]) for name in active)
    inactive = metadata["inactive_controls"]
    if inactive:
        return f"{active_text}; inactive={','.join(inactive)}"
    return active_text


def validate_config(config: Dict[str, Any]):
    if not isinstance(config, dict):
        raise ValueError("config must be a dictionary")
    required_sections = ("io", "lattice", "training", "simulation", "optimizer", "model")
    unsupported_sections = sorted(set(config) - set(required_sections))
    if unsupported_sections:
        raise ValueError(f"Unsupported top-level config sections: {unsupported_sections}")
    for section_name in required_sections:
        if section_name not in config:
            raise ValueError(f"Missing config section '{section_name}'")
        if not isinstance(config[section_name], dict):
            raise ValueError(f"config section '{section_name}' must be a dictionary")

    allowed_section_keys = {
        "io": {"save_dir", "clean_start"},
        "lattice": {
            "Nx",
            "Ny",
            "prim_x",
            "prim_y",
            "boundary_x",
            "boundary_y",
            "hopping_amplitudes",
            "hopping_matrix",
            "hopping_matrix_real",
            "hopping_matrix_imag",
            "U",
            "gamma",
            "F_real",
            "F_imag",
            "Delta",
            "n0",
        },
        "simulation": {
            "seed",
            "num_walker",
            "dt",
            "N_steps",
            "N_windows",
            "t0",
            "gauge_mode",
            "neural_gauge_components",
            "gauge_scale",
            "sde_solver",
            "sde_max_iter",
            "sde_root_rtol",
            "sde_root_atol",
            "sde_affine_expm_order",
            "sde_affine_expm_substeps",
            "sde_newton_damping_steps",
            "apply_neural_gauge_every_steps",
            "apply_neural_gauge_every_stepes",
            "multi_device",
            "walker_batches",
            "params_path",
            "analytic_t_fin",
            "progress_every_window",
            "save_raw_walkers",
            "save_raw_walkers_every_windows",
            "save_format",
            "save_compressed",
            "save_precision",
            "save_observables",
            "save_observables_every_windows",
            "save_observable_errors",
            "remove_unhealth",
            "observables",
            "ed",
            "U",
            "gamma",
            "F_real",
            "F_imag",
            "Delta",
            "n0",
        },
        "optimizer": {
            "type",
            "init_value",
            "peak_value",
            "warmup_steps",
            "end_value",
            "clip_norm",
            "beta1",
            "beta2",
            "decay",
        },
        "model": {
            "embed_dim",
            "time_embed_dim",
            "feature_dims",
            "drift_max",
            "diffusion_max",
            "bound_drift",
            "bound_diffusion",
            "time_feature",
            "num_message_passing_layers",
            "num_cnn_layers",
            "num_tau_encode",
            "num_frequency_encode",
            "graph_time_tau_init",
            "graph_time_w_init",
            "graph_time_tau_min",
            "graph_time_w_min",
            "u1_invariant",
            "apply_output_tanh",
            "bound_drifts",
        },
    }
    for section_name, allowed_keys in allowed_section_keys.items():
        unsupported = sorted(set(config[section_name]) - allowed_keys)
        if unsupported:
            raise ValueError(f"Unsupported {section_name} options: {unsupported}")

    _require_keys("io", config["io"], ("save_dir", "clean_start"))
    config["io"]["save_dir"] = _config_path_string(
        config["io"]["save_dir"],
        "io.save_dir",
    )
    config["io"]["clean_start"] = _config_bool(config["io"]["clean_start"], "io.clean_start")
    for section_name in ("training", "simulation"):
        if config[section_name].get("params_path") is not None:
            config[section_name]["params_path"] = _config_path_string(
                config[section_name]["params_path"],
                f"{section_name}.params_path",
            )
    _require_keys(
        "lattice",
        config["lattice"],
        (
            "Nx",
            "Ny",
            "prim_x",
            "prim_y",
            "boundary_x",
            "boundary_y",
            "U",
            "gamma",
            "F_real",
            "F_imag",
            "Delta",
            "n0",
        ),
    )
    lattice_cfg = config["lattice"]
    lattice_cfg["Nx"] = _config_int(lattice_cfg["Nx"], "lattice.Nx", minimum=1)
    lattice_cfg["Ny"] = _config_int(lattice_cfg["Ny"], "lattice.Ny", minimum=1)
    num_site = lattice_cfg["Nx"] * lattice_cfg["Ny"]
    for key in ("prim_x", "prim_y"):
        primitive = np.asarray(lattice_cfg[key], dtype=float)
        if primitive.shape != (2,) or not np.all(np.isfinite(primitive)):
            raise ValueError(f"lattice.{key} must contain exactly two finite coordinates")
        if np.linalg.norm(primitive) <= 0.0:
            raise ValueError(f"lattice.{key} must be nonzero")
        lattice_cfg[key] = primitive.tolist()
    if lattice_cfg["Nx"] > 1 and lattice_cfg["Ny"] > 1:
        primitive_area = abs(
            float(
                np.linalg.det(
                    np.stack(
                        [
                            np.asarray(lattice_cfg["prim_x"], dtype=float),
                            np.asarray(lattice_cfg["prim_y"], dtype=float),
                        ],
                        axis=0,
                    )
                )
            )
        )
        primitive_scale = (
            np.linalg.norm(np.asarray(lattice_cfg["prim_x"], dtype=float))
            * np.linalg.norm(np.asarray(lattice_cfg["prim_y"], dtype=float))
        )
        if primitive_area <= np.finfo(float).eps * max(primitive_scale, 1.0):
            raise ValueError(
                "lattice.prim_x and lattice.prim_y must be linearly independent "
                "when both lattice dimensions exceed one"
            )
    for key in ("U", "F_real", "F_imag", "Delta"):
        lattice_cfg[key] = _config_float(lattice_cfg[key], f"lattice.{key}")
    lattice_cfg["gamma"] = _config_float(
        lattice_cfg["gamma"],
        "lattice.gamma",
        minimum=0.0,
    )
    lattice_cfg["n0"] = _config_float(
        lattice_cfg["n0"],
        "lattice.n0",
        minimum=0.0,
    )

    hopping_encodings = [
        "hopping_matrix"
        if "hopping_matrix" in lattice_cfg
        else None,
        "hopping_matrix_real/imag"
        if "hopping_matrix_real" in lattice_cfg or "hopping_matrix_imag" in lattice_cfg
        else None,
        "hopping_amplitudes"
        if "hopping_amplitudes" in lattice_cfg
        else None,
    ]
    hopping_encodings = [name for name in hopping_encodings if name is not None]
    if len(hopping_encodings) > 1:
        raise ValueError(
            "lattice hopping must use exactly one encoding; conflicting encodings: "
            f"{hopping_encodings}"
        )
    if hopping_encodings and hopping_encodings[0] == "hopping_matrix_real/imag":
        zero_matrix = np.zeros((num_site, num_site), dtype=float)
        hopping_real = np.asarray(
            lattice_cfg.get("hopping_matrix_real", zero_matrix),
            dtype=float,
        )
        hopping_imag = np.asarray(
            lattice_cfg.get("hopping_matrix_imag", zero_matrix),
            dtype=float,
        )
        if hopping_real.shape != (num_site, num_site):
            raise ValueError(
                "lattice.hopping_matrix_real must have shape "
                f"({num_site}, {num_site})"
            )
        if hopping_imag.shape != (num_site, num_site):
            raise ValueError(
                "lattice.hopping_matrix_imag must have shape "
                f"({num_site}, {num_site})"
            )
        if not np.all(np.isfinite(hopping_real)) or not np.all(np.isfinite(hopping_imag)):
            raise ValueError("lattice hopping-matrix entries must be finite")
        hopping = hopping_real + 1j * hopping_imag
    elif hopping_encodings and hopping_encodings[0] == "hopping_matrix":
        hopping = np.asarray(lattice_cfg["hopping_matrix"], dtype=np.complex128)
        if hopping.shape != (num_site, num_site):
            raise ValueError(
                f"lattice.hopping_matrix must have shape ({num_site}, {num_site})"
            )
        if not np.all(np.isfinite(hopping)):
            raise ValueError("lattice.hopping_matrix entries must be finite")
    elif hopping_encodings:
        hopping_amplitudes = np.asarray(lattice_cfg["hopping_amplitudes"], dtype=float)
        if hopping_amplitudes.ndim != 1 or not np.all(np.isfinite(hopping_amplitudes)):
            raise ValueError("lattice.hopping_amplitudes must be a one-dimensional finite sequence")
        lattice_cfg["hopping_amplitudes"] = hopping_amplitudes.tolist()
        hopping = None
    else:
        hopping = np.zeros((num_site, num_site), dtype=np.complex128)
    if hopping is not None and not np.allclose(
        hopping,
        np.conjugate(hopping.T),
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        raise ValueError("lattice hopping matrix must be Hermitian")

    training_cfg = config["training"]
    staged_schedule = training_cfg.get("staged_schedule")
    segmented_cfg = training_cfg.get("segmented_overlap")
    if segmented_cfg is not None and not isinstance(segmented_cfg, dict):
        raise ValueError("training.segmented_overlap must be a dict when provided")
    if isinstance(segmented_cfg, dict):
        unsupported_segmented = sorted(
            set(segmented_cfg)
            - {
                "enabled",
                "segment_overlap_windows",
                "max_bank_refresh_failures",
                "n_windows_per_segment",
                "n_window_op_steps",
                "stage_schedule",
                "active_stage",
            }
        )
        if unsupported_segmented:
            raise ValueError(
                "Unsupported training.segmented_overlap options: "
                f"{unsupported_segmented}"
            )
    if isinstance(segmented_cfg, dict) and "enabled" in segmented_cfg:
        segmented_cfg["enabled"] = _config_bool(
            segmented_cfg["enabled"],
            "training.segmented_overlap.enabled",
        )
    segmented_enabled = isinstance(segmented_cfg, dict) and bool(segmented_cfg.get("enabled", False))
    segmented_stage_schedule = segmented_cfg.get("stage_schedule") if isinstance(segmented_cfg, dict) else None
    segmented_stage_schedule_present = isinstance(segmented_stage_schedule, list) and len(segmented_stage_schedule) > 0
    if segmented_enabled and not segmented_stage_schedule_present:
        raise ValueError("training.segmented_overlap.enabled requires a non-empty stage_schedule")
    if segmented_stage_schedule_present and not segmented_enabled:
        raise ValueError("training.segmented_overlap.stage_schedule requires segmented_overlap.enabled = true")
    training_required = [
        "seed",
        "dt",
        "gauge_mode",
        "gauge_scale",
        "loss_gauge_prefactor",
        "loss_L2_prefactor",
        "load_parameters",
        "save_every",
    ]
    if staged_schedule is None and not segmented_enabled and not segmented_stage_schedule_present:
        training_required.extend(["n_epoch", "num_walker", "N_steps", "N_windows"])
    _require_keys("training", training_cfg, tuple(training_required))
    training_cfg["seed"] = _config_int(training_cfg["seed"], "training.seed")
    training_cfg["dt"] = _config_float(
        training_cfg["dt"],
        "training.dt",
        strictly_positive=True,
    )
    training_cfg["t0"] = _config_float(
        training_cfg.get("t0", 0.0),
        "training.t0",
    )
    training_cfg["gauge_scale"] = _config_float(
        training_cfg["gauge_scale"],
        "training.gauge_scale",
        minimum=0.0,
    )
    training_cfg["save_every"] = _config_int(
        training_cfg["save_every"],
        "training.save_every",
        minimum=0,
    )
    training_cfg["sde_max_iter"] = _config_int(
        training_cfg.get("sde_max_iter", 4),
        "training.sde_max_iter",
        minimum=1,
    )
    training_cfg["sde_solver"] = normalize_sde_solver(
        training_cfg.get("sde_solver", DEFAULT_SDE_SOLVER)
    )
    training_cfg["sde_root_rtol"] = _config_float(
        training_cfg.get("sde_root_rtol", SDE_ROOT_RTOL_DEFAULT),
        "training.sde_root_rtol",
        strictly_positive=True,
    )
    training_cfg["sde_root_atol"] = _config_float(
        training_cfg.get("sde_root_atol", SDE_ROOT_ATOL_DEFAULT),
        "training.sde_root_atol",
        strictly_positive=True,
    )
    training_cfg["sde_affine_expm_order"] = _config_int(
        training_cfg.get("sde_affine_expm_order", 6),
        "training.sde_affine_expm_order",
        minimum=1,
    )
    training_cfg["sde_affine_expm_substeps"] = _config_int(
        training_cfg.get("sde_affine_expm_substeps", 1),
        "training.sde_affine_expm_substeps",
        minimum=1,
    )
    training_cfg["sde_newton_damping_steps"] = _config_int(
        training_cfg.get("sde_newton_damping_steps", 4),
        "training.sde_newton_damping_steps",
        minimum=1,
    )
    training_cfg["lr_schedule_step_divisor"] = _config_int(
        training_cfg.get("lr_schedule_step_divisor", 1),
        "training.lr_schedule_step_divisor",
        minimum=1,
    )
    for key in ("n_epoch", "num_walker", "N_steps", "N_windows"):
        if key in training_cfg:
            training_cfg[key] = _config_int(
                training_cfg[key],
                f"training.{key}",
                minimum=1,
            )
    for key in ("log_every", "noise_refresh_every"):
        if key in training_cfg:
            training_cfg[key] = _config_int(
                training_cfg[key],
                f"training.{key}",
                minimum=1,
            )
    if "plot_every" in training_cfg:
        training_cfg["plot_every"] = _config_int(
            training_cfg["plot_every"],
            "training.plot_every",
            minimum=0,
        )
    for key in ("load_parameters", "make_plots"):
        if key in training_cfg:
            training_cfg[key] = _config_bool(
                training_cfg[key],
                f"training.{key}",
            )
    _validate_multi_device_config(training_cfg, "training")
    if staged_schedule is not None and segmented_stage_schedule_present:
        raise ValueError(
            "training.staged_schedule cannot be combined with training.segmented_overlap.stage_schedule"
        )
    if "apply_neural_gauge_every_stepes" in training_cfg:
        raise ValueError(
            "training.apply_neural_gauge_every_stepes is misspelled; "
            "use training.apply_neural_gauge_every_steps"
        )
    active_training_keys = {
        "EMA",
        "N_steps",
        "N_windows",
        "apply_neural_gauge_every_steps",
        "dt",
        "gauge_mode",
        "gauge_scale",
        "load_parameters",
        "log_every",
        "loss_L2_prefactor",
        "loss_ess_prefactor",
        "loss_gauge_prefactor",
        "loss_pareto_k_prefactor",
        "loss_residual_gmm_prefactor",
        "lr_schedule_step_divisor",
        "make_plots",
        "multi_device",
        "n_epoch",
        "neural_gauge_components",
        "neural_gauge_state_gradient",
        "noise_refresh_every",
        "num_walker",
        "operator_applied_quantities",
        "operator_applied_quantities_mode",
        "operator_monomials",
        "params_path",
        "pareto_k_applied_quantities",
        "pareto_k_applied_quantities_mode",
        "pareto_k_monomials",
        "pareto_k_envelope_beta",
        "pareto_k_envelope_excess",
        "pareto_k_min_tail_count",
        "pareto_k_tail_fraction",
        "pareto_k_threshold",
        "pareto_k_threshold_tau",
        "plot_every",
        "q_winsor",
        "residual_gmm_cov_floor",
        "residual_gmm_cov_shrinkage",
        "residual_gmm_d_clip",
        "residual_gmm_integrator_nodes",
        "residual_gmm_trace_mode",
        "residual_gmm_time_aggregation",
        "residual_gmm_time_beta",
        "save_every",
        "sde_affine_expm_order",
        "sde_affine_expm_substeps",
        "sde_newton_damping_steps",
        "sde_root_atol",
        "sde_root_rtol",
        "sde_solver",
        "sde_max_iter",
        "seed",
        "segmented_overlap",
        "staged_schedule",
        "t0",
    }
    unsupported_training_keys = sorted(set(training_cfg) - active_training_keys)
    if unsupported_training_keys:
        raise ValueError(f"Unsupported training options: {unsupported_training_keys}")
    if staged_schedule is not None:
        if not isinstance(staged_schedule, list) or len(staged_schedule) == 0:
            raise ValueError("training.staged_schedule must be a non-empty list when provided")
        for idx, stage in enumerate(staged_schedule):
            if not isinstance(stage, dict):
                raise ValueError("Each item in training.staged_schedule must be a dict")
            unsupported_stage = sorted(
                set(stage)
                - {"stage_id", "n_epoch", "N_windows", "num_walker", "N_steps"}
            )
            if unsupported_stage:
                raise ValueError(
                    f"Unsupported training.staged_schedule[{idx}] options: "
                    f"{unsupported_stage}"
                )
            missing = [key for key in ("n_epoch", "N_windows", "num_walker", "N_steps") if key not in stage]
            if missing:
                raise ValueError(f"Stage {idx + 1} is missing keys: {missing}")
            for key in ("n_epoch", "N_windows", "num_walker", "N_steps"):
                stage[key] = _config_int(
                    stage[key],
                    f"training.staged_schedule[{idx}].{key}",
                    minimum=1,
                )
            if "stage_id" in stage:
                stage["stage_id"] = _config_int(
                    stage["stage_id"],
                    f"training.staged_schedule[{idx}].stage_id",
                    minimum=1,
                )
        staged_ids = [
            int(stage.get("stage_id", idx + 1))
            for idx, stage in enumerate(staged_schedule)
        ]
        if len(set(staged_ids)) != len(staged_ids):
            raise ValueError("training.staged_schedule stage_id values must be unique")
    _require_keys(
        "simulation",
        config["simulation"],
        (
            "seed",
            "num_walker",
            "dt",
            "N_steps",
            "N_windows",
            "t0",
            "gauge_mode",
        ),
    )
    simulation_cfg = config["simulation"]
    simulation_cfg["seed"] = _config_int(simulation_cfg["seed"], "simulation.seed")
    simulation_cfg["num_walker"] = _config_int(
        simulation_cfg["num_walker"],
        "simulation.num_walker",
        minimum=1,
    )
    simulation_cfg["N_steps"] = _config_int(
        simulation_cfg["N_steps"],
        "simulation.N_steps",
        minimum=1,
    )
    simulation_cfg["N_windows"] = _config_int(
        simulation_cfg["N_windows"],
        "simulation.N_windows",
        minimum=1,
    )
    simulation_cfg["dt"] = _config_float(
        simulation_cfg["dt"],
        "simulation.dt",
        strictly_positive=True,
    )
    simulation_cfg["t0"] = _config_float(
        simulation_cfg["t0"],
        "simulation.t0",
    )
    simulation_cfg["gauge_scale"] = _config_float(
        simulation_cfg.get("gauge_scale", 1.0),
        "simulation.gauge_scale",
        minimum=0.0,
    )
    simulation_cfg["sde_max_iter"] = _config_int(
        simulation_cfg.get("sde_max_iter", 4),
        "simulation.sde_max_iter",
        minimum=1,
    )
    simulation_cfg["sde_solver"] = normalize_sde_solver(
        simulation_cfg.get("sde_solver", training_cfg["sde_solver"])
    )
    simulation_cfg["sde_root_rtol"] = _config_float(
        simulation_cfg.get("sde_root_rtol", training_cfg["sde_root_rtol"]),
        "simulation.sde_root_rtol",
        strictly_positive=True,
    )
    simulation_cfg["sde_root_atol"] = _config_float(
        simulation_cfg.get("sde_root_atol", training_cfg["sde_root_atol"]),
        "simulation.sde_root_atol",
        strictly_positive=True,
    )
    simulation_cfg["sde_affine_expm_order"] = _config_int(
        simulation_cfg.get(
            "sde_affine_expm_order", training_cfg["sde_affine_expm_order"]
        ),
        "simulation.sde_affine_expm_order",
        minimum=1,
    )
    simulation_cfg["sde_affine_expm_substeps"] = _config_int(
        simulation_cfg.get(
            "sde_affine_expm_substeps",
            training_cfg["sde_affine_expm_substeps"],
        ),
        "simulation.sde_affine_expm_substeps",
        minimum=1,
    )
    simulation_cfg["sde_newton_damping_steps"] = _config_int(
        simulation_cfg.get(
            "sde_newton_damping_steps", training_cfg["sde_newton_damping_steps"]
        ),
        "simulation.sde_newton_damping_steps",
        minimum=1,
    )
    if "progress_every_window" in simulation_cfg:
        simulation_cfg["progress_every_window"] = _config_int(
            simulation_cfg["progress_every_window"],
            "simulation.progress_every_window",
            minimum=0,
        )
    for key in (
        "save_raw_walkers",
        "save_compressed",
        "save_observables",
        "save_observable_errors",
    ):
        if key in simulation_cfg:
            simulation_cfg[key] = _config_bool(
                simulation_cfg[key],
                f"simulation.{key}",
            )
    _validate_multi_device_config(simulation_cfg, "simulation")
    if "apply_neural_gauge_every_stepes" in config["simulation"]:
        raise ValueError(
            "simulation.apply_neural_gauge_every_stepes is misspelled; "
            "use simulation.apply_neural_gauge_every_steps"
        )
    if "apply_neural_gauge_every_steps" in config["simulation"]:
        refresh_steps = _config_int(
            config["simulation"]["apply_neural_gauge_every_steps"],
            "simulation.apply_neural_gauge_every_steps",
            minimum=1,
        )
        if refresh_steps >= int(config["simulation"]["N_steps"]):
            raise ValueError("simulation.apply_neural_gauge_every_steps must be smaller than simulation.N_steps")
        config["simulation"]["apply_neural_gauge_every_steps"] = refresh_steps
    forbidden_simulation_physics_keys = ("U", "gamma", "F_real", "F_imag", "Delta", "n0")
    simulation_physics_overrides = [
        key for key in forbidden_simulation_physics_keys if key in config["simulation"]
    ]
    if simulation_physics_overrides:
        raise ValueError(
            "simulation must not redefine lattice physics. "
            f"Remove these keys from simulation and keep them only in lattice: {simulation_physics_overrides}"
        )
    _require_keys(
        "optimizer",
        config["optimizer"],
        ("type", "init_value", "peak_value", "warmup_steps", "end_value", "clip_norm"),
    )
    optimizer_cfg = config["optimizer"]
    optimizer_cfg["warmup_steps"] = _config_int(
        optimizer_cfg["warmup_steps"],
        "optimizer.warmup_steps",
        minimum=0,
    )
    for key in ("init_value", "peak_value", "end_value"):
        optimizer_cfg[key] = _config_float(
            optimizer_cfg[key],
            f"optimizer.{key}",
            minimum=0.0,
        )
    optimizer_cfg["clip_norm"] = _config_float(
        optimizer_cfg["clip_norm"],
        "optimizer.clip_norm",
        strictly_positive=True,
    )
    for key in ("beta1", "beta2"):
        if key not in optimizer_cfg:
            continue
        value = _config_float(optimizer_cfg[key], f"optimizer.{key}", minimum=0.0)
        if value >= 1.0:
            raise ValueError(f"optimizer.{key} must be < 1")
        optimizer_cfg[key] = value
    if "apply_output_tanh" in config["model"]:
        raise ValueError(
            "model.apply_output_tanh was removed; "
            "use model.bound_drift and model.bound_diffusion as independent switches"
        )
    if "bound_drifts" in config["model"]:
        raise ValueError("model.bound_drifts was removed; use model.bound_drift")
    _require_keys(
        "model",
        config["model"],
        (
            "embed_dim",
            "time_embed_dim",
            "feature_dims",
            "drift_max",
            "diffusion_max",
            "bound_drift",
            "bound_diffusion",
            "time_feature",
            "num_message_passing_layers",
        ),
    )
    model_cfg = config["model"]
    model_cfg["embed_dim"] = _config_int(
        model_cfg["embed_dim"],
        "model.embed_dim",
        minimum=1,
    )
    model_cfg["time_embed_dim"] = _config_int(
        model_cfg["time_embed_dim"],
        "model.time_embed_dim",
        minimum=1,
    )
    feature_dims = model_cfg["feature_dims"]
    if isinstance(feature_dims, (str, bytes)) or not isinstance(feature_dims, Sequence):
        raise ValueError("model.feature_dims must be a sequence of exactly two positive integers")
    feature_dims = [
        _config_int(value, f"model.feature_dims[{idx}]", minimum=1)
        for idx, value in enumerate(feature_dims)
    ]
    if len(feature_dims) != 2:
        raise ValueError("model.feature_dims must contain exactly two positive integers")
    model_cfg["feature_dims"] = feature_dims
    model_cfg["num_message_passing_layers"] = _config_int(
        model_cfg["num_message_passing_layers"],
        "model.num_message_passing_layers",
        minimum=1,
    )
    if "num_cnn_layers" in model_cfg:
        model_cfg["num_cnn_layers"] = _config_int(
            model_cfg["num_cnn_layers"],
            "model.num_cnn_layers",
            minimum=1,
        )
    for key in ("num_tau_encode", "num_frequency_encode"):
        if key in model_cfg:
            model_cfg[key] = _config_int(
                model_cfg[key],
                f"model.{key}",
                minimum=1,
            )
    for key in ("graph_time_tau_min", "graph_time_w_min"):
        if key in model_cfg:
            model_cfg[key] = _config_float(
                model_cfg[key],
                f"model.{key}",
                minimum=0.0,
            )
    for key, count_key in (
        ("graph_time_tau_init", "num_tau_encode"),
        ("graph_time_w_init", "num_frequency_encode"),
    ):
        if key not in model_cfg:
            continue
        values = np.asarray(model_cfg[key], dtype=float)
        if values.ndim != 1 or values.size < 1 or not np.all(np.isfinite(values)):
            raise ValueError(f"model.{key} must be a non-empty one-dimensional finite sequence")
        if np.any(values <= 0.0):
            raise ValueError(f"model.{key} entries must be > 0")
        minimum_key = (
            "graph_time_tau_min"
            if key == "graph_time_tau_init"
            else "graph_time_w_min"
        )
        default_minimum = 1.0e-3 if key == "graph_time_tau_init" else 1.0e-4
        minimum_value = float(model_cfg.get(minimum_key, default_minimum))
        if np.any(values <= minimum_value):
            raise ValueError(
                f"model.{key} entries must be strictly larger than "
                f"model.{minimum_key} ({minimum_value:g})"
            )
        if count_key in model_cfg and int(model_cfg[count_key]) != int(values.size):
            raise ValueError(
                f"model.{count_key} must match the length of model.{key}"
            )
        model_cfg[key] = values.tolist()

    training_mode = str(config["training"]["gauge_mode"]).strip()
    simulation_mode = str(config["simulation"]["gauge_mode"]).strip()
    config["training"]["gauge_mode"] = training_mode
    config["simulation"]["gauge_mode"] = simulation_mode

    config["model"]["u1_invariant"] = _config_bool(
        config["model"].get("u1_invariant", False),
        "model.u1_invariant",
    )
    bound_drift = config["model"]["bound_drift"]
    bound_diffusion = config["model"]["bound_diffusion"]
    if not isinstance(bound_drift, (bool, np.bool_)):
        raise ValueError("model.bound_drift must be a boolean")
    if not isinstance(bound_diffusion, (bool, np.bool_)):
        raise ValueError("model.bound_diffusion must be a boolean")
    drift_max = float(config["model"]["drift_max"])
    diffusion_max = float(config["model"]["diffusion_max"])
    if not np.isfinite(drift_max) or drift_max <= 0.0:
        raise ValueError("model.drift_max must be finite and > 0")
    if not np.isfinite(diffusion_max) or diffusion_max <= 0.0:
        raise ValueError("model.diffusion_max must be finite and > 0")
    config["model"]["bound_drift"] = bool(bound_drift)
    config["model"]["bound_diffusion"] = bool(bound_diffusion)
    config["model"]["drift_max"] = drift_max
    config["model"]["diffusion_max"] = diffusion_max

    if training_mode not in NEURAL_GAUGE_MODES:
        raise ValueError(f"training.gauge_mode must be one of {sorted(NEURAL_GAUGE_MODES)}")
    if config["simulation"]["gauge_mode"] not in VALID_GAUGE_MODES:
        raise ValueError(f"simulation.gauge_mode must be one of {sorted(VALID_GAUGE_MODES)}")
    if config["training"]["gauge_mode"] not in VALID_GAUGE_MODES:
        raise ValueError(f"training.gauge_mode must be one of {sorted(VALID_GAUGE_MODES)}")
    if training_mode == "neural_graph" or simulation_mode == "neural_graph":
        resolved_hopping = (
            np.asarray(hopping, dtype=np.complex128)
            if hopping is not None
            else np.zeros((num_site, num_site), dtype=np.complex128)
        )
        if np.any(np.abs(np.imag(resolved_hopping)) > 1.0e-12):
            raise ValueError(
                "neural_graph requires a real-valued lattice hopping matrix; "
                "use neural_mlp or neural_cnn for complex Hermitian hopping"
            )
    config["training"]["neural_gauge_components"] = normalize_neural_gauge_components(
        config["training"].get("neural_gauge_components", "both")
    )
    config["simulation"]["neural_gauge_components"] = normalize_neural_gauge_components(
        config["simulation"].get("neural_gauge_components", "both")
    )
    if training_mode == "neural_mlp" or simulation_mode == "neural_mlp":
        allowed_time_features = {"Dense", "None", "FAN", "FixedFourier"}
    else:
        allowed_time_features = {
            "Dense",
            "None",
            "Raw",
            "raw",
            "raw_time",
            "DampedTrig",
            "dampedtrig",
            "DampedTrigTrend",
            "dampedtrigtrend",
            "damped_trig_trend",
            "LearnedDampedTrig",
            "learneddampedtrig",
            "LearnedDampedTrigTrend",
            "learneddampedtrigtrend",
            "learned_damped_trig_trend",
            "damped_trig",
            "learned_damped_trig",
        }
    if config["model"]["time_feature"] not in allowed_time_features:
        raise ValueError(f"model.time_feature must be one of {sorted(allowed_time_features)}")
    graph_time_feature = str(config["model"]["time_feature"]).strip().lower().replace("-", "_")
    if (
        training_mode in {"neural_graph", "neural_cnn"}
        or simulation_mode in {"neural_graph", "neural_cnn"}
    ):
        if graph_time_feature in {"dampedtrig", "learneddampedtrig"}:
            graph_time_feature = graph_time_feature.replace("dampedtrig", "damped_trig")
        if graph_time_feature in {"dampedtrigtrend", "learneddampedtrigtrend"}:
            graph_time_feature = graph_time_feature.replace("dampedtrigtrend", "damped_trig_trend")
        if graph_time_feature in {
            "damped_trig",
            "learned_damped_trig",
            "damped_trig_trend",
            "learned_damped_trig_trend",
        }:
            num_tau_encode = int(
                config["model"].get(
                    "num_tau_encode",
                    len(config["model"].get("graph_time_tau_init", (0.1, 0.4, 1.2, 3.6))),
                )
            )
            num_frequency_encode = int(
                config["model"].get(
                    "num_frequency_encode",
                    len(config["model"].get("graph_time_w_init", (0.25, 0.5, 1.0, 2.0))),
                )
            )
            if num_tau_encode < 1:
                raise ValueError("model.num_tau_encode must be >= 1 when graph time_feature uses damped trig")
            if num_frequency_encode < 1:
                raise ValueError(
                    "model.num_frequency_encode must be >= 1 when graph time_feature uses damped trig"
                )
            tau_min = float(config["model"].get("graph_time_tau_min", 1.0e-3))
            w_min = float(config["model"].get("graph_time_w_min", 1.0e-4))
            if tau_min < 0.0 or w_min < 0.0:
                raise ValueError("model.graph_time_tau_min and model.graph_time_w_min must be >= 0")
    if simulation_mode in NEURAL_GAUGE_MODES and simulation_mode != training_mode:
        raise ValueError(
            "training.gauge_mode and simulation.gauge_mode must match when simulation uses a neural checkpoint"
        )
    loss_prefactor_defaults = {
        "loss_L2_prefactor": 0.0,
        "loss_ess_prefactor": 0.0,
        "loss_gauge_prefactor": 0.0,
        "loss_pareto_k_prefactor": 0.0,
        "loss_residual_gmm_prefactor": 0.0,
    }
    for key, default in loss_prefactor_defaults.items():
        value = float(config["training"].get(key, default))
        if (not np.isfinite(value)) or value < 0.0:
            raise ValueError(f"training.{key} must be finite and >= 0")
        config["training"][key] = value
    pareto_k_threshold = float(config["training"].get("pareto_k_threshold", 0.7))
    if (not np.isfinite(pareto_k_threshold)) or pareto_k_threshold < 0.0:
        raise ValueError("training.pareto_k_threshold must be finite and >= 0")
    config["training"]["pareto_k_threshold"] = pareto_k_threshold
    pareto_k_threshold_tau = float(config["training"].get("pareto_k_threshold_tau", 0.1))
    if (not np.isfinite(pareto_k_threshold_tau)) or pareto_k_threshold_tau < 0.0:
        raise ValueError("training.pareto_k_threshold_tau must be finite and >= 0")
    config["training"]["pareto_k_threshold_tau"] = pareto_k_threshold_tau
    pareto_k_envelope_beta = float(config["training"].get("pareto_k_envelope_beta", 0.5))
    if (not np.isfinite(pareto_k_envelope_beta)) or pareto_k_envelope_beta < 0.0:
        raise ValueError("training.pareto_k_envelope_beta must be finite and >= 0")
    config["training"]["pareto_k_envelope_beta"] = pareto_k_envelope_beta
    pareto_k_envelope_excess = str(config["training"].get("pareto_k_envelope_excess", "log")).strip().lower()
    if pareto_k_envelope_excess not in {"log", "ratio"}:
        raise ValueError('training.pareto_k_envelope_excess must be "log" or "ratio"')
    config["training"]["pareto_k_envelope_excess"] = pareto_k_envelope_excess
    pareto_k_tail_fraction = float(config["training"].get("pareto_k_tail_fraction", 0.01))
    if (not np.isfinite(pareto_k_tail_fraction)) or pareto_k_tail_fraction <= 0.0 or pareto_k_tail_fraction >= 1.0:
        raise ValueError("training.pareto_k_tail_fraction must be finite and in (0, 1)")
    config["training"]["pareto_k_tail_fraction"] = pareto_k_tail_fraction
    pareto_k_min_tail_count = _config_int(
        config["training"].get("pareto_k_min_tail_count", 32),
        "training.pareto_k_min_tail_count",
        minimum=1,
    )
    config["training"]["pareto_k_min_tail_count"] = pareto_k_min_tail_count
    config["training"]["residual_gmm_trace_mode"] = (
        normalize_residual_gmm_trace_mode(
            config["training"].get(
                "residual_gmm_trace_mode",
                DEFAULT_RESIDUAL_GMM_TRACE_MODE,
            )
        )
    )
    residual_gmm_integrator_nodes = config["training"].get(
        "residual_gmm_integrator_nodes",
        6,
    )
    if (
        isinstance(residual_gmm_integrator_nodes, (bool, np.bool_))
        or not isinstance(
            residual_gmm_integrator_nodes,
            (int, np.integer),
        )
        or int(residual_gmm_integrator_nodes) not in {3, 4, 5, 6}
    ):
        raise ValueError(
            "training.residual_gmm_integrator_nodes must be "
            "one of the integers 3, 4, 5, or 6"
        )
    config["training"]["residual_gmm_integrator_nodes"] = int(
        residual_gmm_integrator_nodes
    )
    config["training"]["residual_gmm_d_clip"] = _config_float(
        config["training"].get("residual_gmm_d_clip", 10.0),
        "training.residual_gmm_d_clip",
        strictly_positive=True,
    )
    config["training"]["residual_gmm_cov_floor"] = _config_float(
        config["training"].get("residual_gmm_cov_floor", 1.0e-8),
        "training.residual_gmm_cov_floor",
        strictly_positive=True,
    )
    residual_gmm_cov_shrinkage = _config_float(
        config["training"].get("residual_gmm_cov_shrinkage", 0.05),
        "training.residual_gmm_cov_shrinkage",
        minimum=0.0,
    )
    if residual_gmm_cov_shrinkage > 1.0:
        raise ValueError(
            "training.residual_gmm_cov_shrinkage must be finite and in [0, 1]"
        )
    config["training"]["residual_gmm_cov_shrinkage"] = (
        residual_gmm_cov_shrinkage
    )
    residual_gmm_time_aggregation = str(
        config["training"].get("residual_gmm_time_aggregation", "mean")
    ).strip().lower()
    if residual_gmm_time_aggregation not in {
        "mean",
        "log1p",
        "entropic",
        "entropic_log1p",
    }:
        raise ValueError(
            "training.residual_gmm_time_aggregation must be one of "
            "'mean', 'log1p', 'entropic', or 'entropic_log1p'"
        )
    config["training"]["residual_gmm_time_aggregation"] = (
        residual_gmm_time_aggregation
    )
    config["training"]["residual_gmm_time_beta"] = _config_float(
        config["training"].get("residual_gmm_time_beta", 2.0),
        "training.residual_gmm_time_beta",
        minimum=0.0,
    )
    if (
        loss_prefactor_is_active(
            config["training"]["loss_residual_gmm_prefactor"]
        )
        and "operator_monomials" not in config["training"]
    ):
        config["training"]["operator_monomials"] = [
            [0, 1],
            [1, 0],
            [1, 1],
            [2, 2],
        ]
    operator_applied_quantities, operator_applied_quantities_mode, operator_monomials = _normalize_training_monomial_selector(
        config["training"],
        prefix="operator",
        specs=OPERATOR_MOMENT_SPECS,
        max_order=OPERATOR_MOMENT_MAX_ORDER,
        default_order=OPERATOR_MOMENT_DEFAULT_ORDER,
    )
    if (
        loss_prefactor_is_active(
            config["training"]["loss_residual_gmm_prefactor"]
        )
        and not operator_monomials
    ):
        raise ValueError(
            "training.operator_monomials must contain at least one nonidentity "
            "equation when loss_residual_gmm_prefactor is nonzero"
        )
    pareto_k_auto_requested = is_auto_monomial_selector(
        config["training"].get("pareto_k_monomials", config["training"].get("pareto_k_applied_quantities"))
    )
    selected_operator_specs_for_auto = selected_monomial_specs(
        OPERATOR_MOMENT_SPECS,
        operator_applied_quantities,
        operator_applied_quantities_mode,
        operator_monomials,
    )
    auto_health_monomials = onsite_monomials_in_operator_equations(
        ((m_power, n_power) for _total_order, m_power, n_power, _term in selected_operator_specs_for_auto),
        max_order=PARETO_K_MONOMIAL_MAX_ORDER,
    )
    pareto_k_applied_quantities, pareto_k_applied_quantities_mode, pareto_k_monomials = _normalize_training_monomial_selector(
        config["training"],
        prefix="pareto_k",
        specs=PARETO_K_MONOMIAL_SPECS,
        max_order=PARETO_K_MONOMIAL_MAX_ORDER,
        default_order=6,
        allow_auto=True,
        auto_monomials=auto_health_monomials,
    )
    loss_residual_gmm_prefactor = float(
        config["training"].get("loss_residual_gmm_prefactor", 0.0)
    )
    residual_gmm_active = loss_prefactor_is_active(
        loss_residual_gmm_prefactor
    )
    if loss_prefactor_is_active(
        config["training"].get("loss_pareto_k_prefactor", 0.0)
    ):
        if staged_schedule is not None:
            pareto_walker_entries = [
                (f"Stage {idx + 1}", int(stage["num_walker"]))
                for idx, stage in enumerate(staged_schedule)
            ]
        elif segmented_stage_schedule_present:
            pareto_walker_entries = [
                (f"Segmented stage {idx + 1}", int(stage["num_walker"]))
                for idx, stage in enumerate(segmented_stage_schedule)
            ]
        elif "num_walker" in training_cfg:
            pareto_walker_entries = [("training", int(training_cfg["num_walker"]))]
        else:
            pareto_walker_entries = []
        for label, walker_count in pareto_walker_entries:
            if walker_count < 2:
                raise ValueError(
                    "Pareto-k training requires at least two walkers; "
                    f"{label} has num_walker={walker_count}"
                )
    if (
        not operator_monomials
        and operator_applied_quantities > OPERATOR_MOMENT_MAX_ORDER
        and residual_gmm_active
    ):
        raise ValueError(
            "training.operator_applied_quantities must be <= "
            f"{OPERATOR_MOMENT_MAX_ORDER} when residual GMM is active"
        )
    selected_pareto_terms = set(
        selected_monomial_terms(
            PARETO_K_MONOMIAL_SPECS,
            pareto_k_applied_quantities,
            pareto_k_applied_quantities_mode,
            pareto_k_monomials,
        )
    )
    q_winsor = float(config["training"].get("q_winsor", 0.95))
    if (not np.isfinite(q_winsor)) or (q_winsor > 0.0 and (q_winsor <= 0.5 or q_winsor > 1.0)):
        raise ValueError(
            "training.q_winsor must be finite and either <= 0 to disable quantile winsorization "
            "or in (0.5, 1] to enable it"
        )
    config["training"]["q_winsor"] = q_winsor
    config["training"]["noise_refresh_every"] = _config_int(
        config["training"].get("noise_refresh_every", 1),
        "training.noise_refresh_every",
        minimum=1,
    )
    neural_gauge_state_gradient = str(training_cfg.get("neural_gauge_state_gradient", "full"))
    if neural_gauge_state_gradient not in ("full", "each_apply"):
        raise ValueError(
            "training.neural_gauge_state_gradient must be 'full' or 'each_apply'"
        )
    training_cfg["neural_gauge_state_gradient"] = neural_gauge_state_gradient
    if segmented_stage_schedule_present:
        if segmented_enabled:
            segmented_cfg["segment_overlap_windows"] = _config_int(
                segmented_cfg.get("segment_overlap_windows", 1),
                "training.segmented_overlap.segment_overlap_windows",
                minimum=1,
            )
            if "n_window_op_steps" in segmented_cfg:
                raise ValueError(
                    "training.segmented_overlap.n_window_op_steps was removed; "
                    "residual-GMM training uses each full segment window."
                )
            segmented_cfg["max_bank_refresh_failures"] = _config_int(
                segmented_cfg.get("max_bank_refresh_failures", 8),
                "training.segmented_overlap.max_bank_refresh_failures",
                minimum=1,
            )

        default_windows_per_segment = segmented_cfg.get("n_windows_per_segment")
        if default_windows_per_segment is not None:
            default_windows_per_segment = _config_int(
                default_windows_per_segment,
                "training.segmented_overlap.n_windows_per_segment",
                minimum=1,
            )
            segmented_cfg["n_windows_per_segment"] = default_windows_per_segment
        overlap_windows = _config_int(
            segmented_cfg.get("segment_overlap_windows", 1),
            "training.segmented_overlap.segment_overlap_windows",
            minimum=1,
        )
        for idx, stage in enumerate(segmented_stage_schedule):
            if not isinstance(stage, dict):
                raise ValueError("Each item in training.segmented_overlap.stage_schedule must be a dict")
            unsupported_stage = sorted(
                set(stage)
                - {
                    "stage_id",
                    "n_epoch",
                    "n_segments",
                    "n_windows_per_segment",
                    "num_walker",
                    "N_steps",
                }
            )
            if unsupported_stage:
                raise ValueError(
                    "Unsupported training.segmented_overlap.stage_schedule"
                    f"[{idx}] options: {unsupported_stage}"
                )

            missing = [key for key in ("n_epoch", "n_segments", "num_walker", "N_steps") if key not in stage]
            stage_windows_per_segment = stage.get("n_windows_per_segment", default_windows_per_segment)
            if stage_windows_per_segment is None:
                missing.append("n_windows_per_segment")
            if missing:
                raise ValueError(f"Segmented stage {idx + 1} is missing keys: {missing}")

            for key in ("n_epoch", "n_segments", "num_walker", "N_steps"):
                stage[key] = _config_int(
                    stage[key],
                    f"training.segmented_overlap.stage_schedule[{idx}].{key}",
                    minimum=1,
                )
            if "stage_id" in stage:
                stage["stage_id"] = _config_int(
                    stage["stage_id"],
                    f"training.segmented_overlap.stage_schedule[{idx}].stage_id",
                    minimum=1,
                )
            stage_windows_per_segment = _config_int(
                stage_windows_per_segment,
                f"training.segmented_overlap.stage_schedule[{idx}].n_windows_per_segment",
                minimum=1,
            )
            if "n_windows_per_segment" in stage:
                stage["n_windows_per_segment"] = stage_windows_per_segment
            if segmented_enabled and overlap_windows >= int(stage_windows_per_segment):
                raise ValueError(
                    "training.segmented_overlap.segment_overlap_windows must be smaller than "
                    "each stage's n_windows_per_segment"
                )
        segmented_ids = [
            int(stage.get("stage_id", idx + 1))
            for idx, stage in enumerate(segmented_stage_schedule)
        ]
        if len(set(segmented_ids)) != len(segmented_ids):
            raise ValueError(
                "training.segmented_overlap.stage_schedule stage_id values must be unique"
            )
    quadrature_denominators = []
    quadrature_components = []
    if residual_gmm_active:
        residual_nodes = int(
            config["training"]["residual_gmm_integrator_nodes"]
        )
        quadrature_denominators.append(residual_nodes - 1)
        quadrature_components.append(f"{residual_nodes}-node residual GMM")
    required_step_divisor = (
        math.lcm(*quadrature_denominators) if quadrature_denominators else 1
    )
    if required_step_divisor > 1:
        quadrature_name = " combined with ".join(quadrature_components)
        if staged_schedule is not None:
            step_entries = [
                (f"Stage {idx + 1}", int(stage["N_steps"]))
                for idx, stage in enumerate(staged_schedule)
            ]
        elif segmented_stage_schedule_present:
            step_entries = [
                (f"Segmented stage {idx + 1}", int(stage["N_steps"]))
                for idx, stage in enumerate(segmented_stage_schedule)
            ]
        elif "N_steps" in training_cfg:
            step_entries = [("training", int(training_cfg["N_steps"]))]
        else:
            step_entries = []
        for label, n_steps in step_entries:
            if n_steps % required_step_divisor != 0:
                raise ValueError(
                    f"active quadrature grid ({quadrature_name}) requires "
                    "N_steps divisible by "
                    f"{required_step_divisor}; {label} has N_steps={n_steps}"
                )
    if "apply_neural_gauge_every_steps" in training_cfg:
        refresh_steps = _config_int(
            training_cfg["apply_neural_gauge_every_steps"],
            "training.apply_neural_gauge_every_steps",
            minimum=1,
        )
        if staged_schedule is not None:
            for idx, stage in enumerate(staged_schedule):
                if refresh_steps >= int(stage["N_steps"]):
                    raise ValueError(
                        "training.apply_neural_gauge_every_steps must be smaller than "
                        f"Stage {idx + 1} N_steps"
                    )
        elif segmented_stage_schedule_present:
            for idx, stage in enumerate(segmented_stage_schedule):
                if refresh_steps >= int(stage["N_steps"]):
                    raise ValueError(
                        "training.apply_neural_gauge_every_steps must be smaller than "
                        f"Segmented stage {idx + 1} N_steps"
                    )
        elif "N_steps" in training_cfg and refresh_steps >= int(training_cfg["N_steps"]):
            raise ValueError("training.apply_neural_gauge_every_steps must be smaller than training.N_steps")
        training_cfg["apply_neural_gauge_every_steps"] = refresh_steps
    ema_cfg = config["training"].get("EMA", {})
    if ema_cfg is not None:
        if not isinstance(ema_cfg, dict):
            raise ValueError("training.EMA must be a dict when provided")
        unsupported_ema = sorted(
            set(ema_cfg)
            - {
                "enabled",
                "terms",
                "decay",
                "warmup_epochs",
                "eps",
                "floor",
                "ceiling",
                "r_max",
            }
        )
        if unsupported_ema:
            raise ValueError(f"Unsupported training.EMA options: {unsupported_ema}")
        ema_enabled = _config_bool(
            ema_cfg.get("enabled", False),
            "training.EMA.enabled",
        )
        ema_cfg["enabled"] = ema_enabled
        if ema_enabled:
            default_ema_terms_list = []
            if residual_gmm_active:
                default_ema_terms_list.append("loss_residual_gmm")
            if loss_prefactor_is_active(
                config["training"].get("loss_pareto_k_prefactor", 0.0)
            ):
                default_ema_terms_list.append("loss_pareto_k")
            default_ema_terms = tuple(default_ema_terms_list)
            ema_terms = ema_cfg.get("terms", default_ema_terms)
            if ema_terms is None:
                ema_terms = default_ema_terms
            auto_ema_terms = {
                "loss_pareto_k": selected_monomial_terms(
                    PARETO_K_MONOMIAL_SPECS,
                    pareto_k_applied_quantities,
                    pareto_k_applied_quantities_mode,
                    pareto_k_monomials,
                )
                if pareto_k_auto_requested
                else (),
            }
            ema_terms = expand_loss_ema_terms(ema_terms, auto_terms=auto_ema_terms)
            allowed_ema_terms = {
                "loss_residual_gmm",
                "loss_pareto_k",
            } | set(PARETO_K_MONOMIAL_TERMS)
            bad_terms = [term for term in ema_terms if str(term) not in allowed_ema_terms]
            if bad_terms:
                raise ValueError(
                    "training.EMA.terms may only contain loss_residual_gmm, "
                    "loss_pareto_k, loss_pareto_k_m*_n*, "
                    "loss_pareto_k_p*, or loss_pareto_k_auto; "
                    f"got {bad_terms}"
                )
            inactive_monomial_terms = [
                str(term)
                for term in ema_terms
                if str(term) in PARETO_K_MONOMIAL_TERMS and str(term) not in selected_pareto_terms
            ]
            if inactive_monomial_terms:
                raise ValueError(
                    "training.EMA.terms contains Pareto-k monomial terms not selected by "
                    "training.pareto_k_monomials or training.pareto_k_applied_quantities/mode: "
                    f"{inactive_monomial_terms}"
                )
            normalized_terms = []
            for term in ema_terms:
                key = str(term)
                if key not in normalized_terms:
                    normalized_terms.append(key)
            ema_cfg["terms"] = list(normalized_terms)
            if "loss_pareto_k" in normalized_terms and any(
                term in normalized_terms for term in PARETO_K_MONOMIAL_TERMS
            ):
                raise ValueError(
                    "training.EMA.terms cannot mix aggregate loss_pareto_k with "
                    "component monomial terms loss_pareto_k_m*_n*"
                )
            ema_decay = _config_float(
                ema_cfg.get("decay", 0.995),
                "training.EMA.decay",
                minimum=0.0,
            )
            if ema_decay >= 1.0:
                raise ValueError("training.EMA.decay must satisfy 0 <= decay < 1")
            ema_warmup_epochs = _config_int(
                ema_cfg.get("warmup_epochs", 10),
                "training.EMA.warmup_epochs",
                minimum=0,
            )
            ema_eps = _config_float(
                ema_cfg.get("eps", 1.0e-12),
                "training.EMA.eps",
                minimum=0.0,
            )
            ema_floor = _config_float(
                ema_cfg.get("floor", 1.0e-8),
                "training.EMA.floor",
                strictly_positive=True,
            )
            ema_ceiling = _config_float(
                ema_cfg.get("ceiling", 1.0e8),
                "training.EMA.ceiling",
                strictly_positive=True,
            )
            ema_r_max = ema_cfg.get("r_max", 5.0)
            if ema_ceiling <= ema_floor:
                raise ValueError("training.EMA requires 0 < floor < ceiling")
            if ema_r_max is not None:
                ema_r_max = _config_float(
                    ema_r_max,
                    "training.EMA.r_max",
                    minimum=1.0,
                )
            ema_cfg.update(
                {
                    "decay": ema_decay,
                    "warmup_epochs": ema_warmup_epochs,
                    "eps": ema_eps,
                    "floor": ema_floor,
                    "ceiling": ema_ceiling,
                    "r_max": ema_r_max,
                }
            )
    if config["lattice"]["boundary_x"] not in {"open", "periodic"}:
        raise ValueError("lattice.boundary_x must be 'open' or 'periodic'")
    if config["lattice"]["boundary_y"] not in {"open", "periodic"}:
        raise ValueError("lattice.boundary_y must be 'open' or 'periodic'")
    config["optimizer"]["type"] = str(config["optimizer"]["type"]).strip().lower()
    if config["optimizer"]["type"] not in {"adam", "rmsprop"}:
        raise ValueError("optimizer.type must be 'adam' or 'rmsprop'")
    if "decay" in config["optimizer"]:
        decay = _config_float(
            config["optimizer"]["decay"],
            "optimizer.decay",
            minimum=0.0,
        )
        if decay >= 1.0:
            raise ValueError("optimizer.decay must satisfy 0 <= decay < 1")
        config["optimizer"]["decay"] = decay
    walker_batches = config["simulation"].get("walker_batches", {})
    if walker_batches is not None:
        if not isinstance(walker_batches, dict):
            raise ValueError("simulation.walker_batches must be a dict when provided")
        unsupported_batches = sorted(
            set(walker_batches)
            - {
                "enabled",
                "num_walker_per_batch",
                "batch_num_walker",
                "num_batches",
            }
        )
        if unsupported_batches:
            raise ValueError(
                f"Unsupported simulation.walker_batches options: {unsupported_batches}"
            )
        if "enabled" in walker_batches:
            walker_batches["enabled"] = _config_bool(
                walker_batches["enabled"],
                "simulation.walker_batches.enabled",
            )
        if (
            "num_walker_per_batch" in walker_batches
            and "batch_num_walker" in walker_batches
            and _config_int(
                walker_batches["num_walker_per_batch"],
                "simulation.walker_batches.num_walker_per_batch",
                minimum=1,
            )
            != _config_int(
                walker_batches["batch_num_walker"],
                "simulation.walker_batches.batch_num_walker",
                minimum=1,
            )
        ):
            raise ValueError(
                "simulation.walker_batches.num_walker_per_batch conflicts with "
                "the batch_num_walker alias"
            )
        batch_size = walker_batches.get(
            "num_walker_per_batch",
            walker_batches.get("batch_num_walker"),
        )
        num_batches = walker_batches.get("num_batches")
        if batch_size is not None:
            batch_size = _config_int(
                batch_size,
                "simulation.walker_batches.num_walker_per_batch",
                minimum=1,
            )
            walker_batches["num_walker_per_batch"] = batch_size
            walker_batches.pop("batch_num_walker", None)
        if num_batches is not None:
            walker_batches["num_batches"] = _config_int(
                num_batches,
                "simulation.walker_batches.num_batches",
                minimum=1,
            )

    if simulation_mode not in NEURAL_GAUGE_MODES:
        validate_analytical_gauge_mode(mode=simulation_mode)
        if "analytic_t_fin" in config["simulation"]:
            analytic_t_fin = float(config["simulation"]["analytic_t_fin"])
            if not np.isfinite(analytic_t_fin):
                raise ValueError("simulation.analytic_t_fin must be finite when provided")
            if analytic_t_fin < float(config["simulation"]["t0"]):
                raise ValueError("simulation.analytic_t_fin must be greater than or equal to simulation.t0")

    ed_cfg = config["simulation"].get("ed")
    if ed_cfg is not None:
        if not isinstance(ed_cfg, dict):
            raise ValueError("simulation.ed must be a dict when provided")
        unsupported_ed = sorted(set(ed_cfg) - {"enabled", "n_cut"})
        if unsupported_ed:
            raise ValueError(f"Unsupported simulation.ed options: {unsupported_ed}")
        enabled = _config_bool(
            ed_cfg.get("enabled", False),
            "simulation.ed.enabled",
        )
        ed_cfg["enabled"] = enabled
        if enabled and "n_cut" not in ed_cfg:
            raise ValueError("simulation.ed.n_cut is required when simulation.ed.enabled is true")
        if "n_cut" in ed_cfg:
            ed_cfg["n_cut"] = _config_int(
                ed_cfg["n_cut"],
                "simulation.ed.n_cut",
                minimum=0,
            )
        if enabled and int(config["simulation"]["N_steps"]) % 10 != 0:
            raise ValueError(
                "simulation.ed.enabled requires simulation.N_steps divisible by 10 "
                "because the ED reference grid uses N_steps/10 and N_windows*10"
            )

    for key in ("save_precision",):
        if key in config["simulation"]:
            precision = str(config["simulation"][key]).strip().lower()
            if precision not in {"runtime", "float32", "float64"}:
                raise ValueError(
                    f"simulation.{key} must be one of ['runtime', 'float32', 'float64']"
                )
            config["simulation"][key] = precision
    for key in ("save_format",):
        if key in config["simulation"]:
            archive_format = str(config["simulation"][key]).strip().lower()
            if archive_format not in {"npz", "zarr"}:
                raise ValueError(
                    f"simulation.{key} must be one of ['npz', 'zarr']"
                )
            config["simulation"][key] = archive_format
    config["simulation"]["save_raw_walkers_every_windows"] = _config_int(
        config["simulation"].get("save_raw_walkers_every_windows", 1),
        "simulation.save_raw_walkers_every_windows",
        minimum=1,
    )
    config["simulation"]["save_observables_every_windows"] = _config_int(
        config["simulation"].get("save_observables_every_windows", 1),
        "simulation.save_observables_every_windows",
        minimum=1,
    )
    remove_unhealth_cfg = config["simulation"].get("remove_unhealth")
    if remove_unhealth_cfg is not None:
        if isinstance(remove_unhealth_cfg, bool):
            remove_unhealth_cfg = {"enabled": remove_unhealth_cfg}
        if not isinstance(remove_unhealth_cfg, dict):
            raise ValueError("simulation.remove_unhealth must be a dict when provided")
        unsupported_remove = sorted(set(remove_unhealth_cfg) - {"enabled", "logstd"})
        if unsupported_remove:
            raise ValueError(
                f"Unsupported simulation.remove_unhealth options: {unsupported_remove}"
            )
        logstd = _config_float(
            remove_unhealth_cfg.get("logstd", 15.0),
            "simulation.remove_unhealth.logstd",
            strictly_positive=True,
        )
        config["simulation"]["remove_unhealth"] = {
            "enabled": _config_bool(
                remove_unhealth_cfg.get("enabled", False),
                "simulation.remove_unhealth.enabled",
            ),
            "logstd": logstd,
        }
    if "observables" in config["simulation"]:
        observable_names = config["simulation"]["observables"]
        if isinstance(observable_names, str):
            observable_names = [observable_names]
        if not isinstance(observable_names, list):
            raise ValueError("simulation.observables must be a string or a list of strings")
        if not all(isinstance(name, str) for name in observable_names):
            raise ValueError("simulation.observables entries must be strings")
        observable_aliases = {
            "g2_iinital": "g2_initial",
            "G2_iinital": "G2_initial",
            "pareto_k_mean": "pareto-k_mean",
            "pareto_k_max": "pareto-k_max",
            **{name.replace("pareto-k", "pareto_k", 1): name for name in PARETO_K_OBSERVABLE_NAMES},
        }
        supported_observables = {
            "A",
            "B",
            "density",
            "N",
            "G1",
            "G1_local",
            "G1_nn",
            "G1_nnn",
            "G1_initial",
            "G1_initial_local",
            "G1_initial_nn",
            "G1_initial_nnn",
            "G2",
            "G2_initial",
            "g2",
            "g2_initial",
            "G2_local",
            "g2_local",
            "g2_nn",
            "g2_nnn",
            "coherence_fraction",
            "pareto-k_mean",
            "pareto-k_max",
            *PARETO_K_OBSERVABLE_NAMES,
        }
        unsupported = [
            name for name in (observable_aliases.get(str(item), str(item)) for item in observable_names)
            if name not in supported_observables
        ]
        if unsupported:
            raise ValueError(
                f"Unsupported simulation observables {unsupported}. "
                f"Supported: {sorted(supported_observables)}"
            )
        normalized_observables = [
            observable_aliases.get(name, name)
            for name in observable_names
        ]
        if (
            any(
                name in {"pareto-k_mean", "pareto-k_max", *PARETO_K_OBSERVABLE_NAMES}
                for name in normalized_observables
            )
            and int(config["simulation"]["num_walker"]) < 2
        ):
            raise ValueError(
                "simulation Pareto-k observables require simulation.num_walker >= 2"
            )
        config["simulation"]["observables"] = normalized_observables


def resolve_config_paths(config: Dict[str, Any], config_path: str):
    """Resolve config-relative output and checkpoint paths consistently."""

    resolved_config_path = Path(config_path).expanduser().resolve()
    config_dir = resolved_config_path.parent
    save_dir = Path(
        _config_path_string(config["io"]["save_dir"], "io.save_dir")
    ).expanduser()
    if not save_dir.is_absolute():
        save_dir = config_dir / save_dir
    save_dir = save_dir.resolve()
    config["io"]["save_dir"] = str(save_dir)
    if bool(config["io"].get("clean_start", False)):
        _validate_recursive_delete_target(save_dir)
        if _path_contains(save_dir, resolved_config_path):
            raise ValueError(
                "io.clean_start cannot be used when io.save_dir contains the "
                f"active config file: '{resolved_config_path}'"
            )
    for section_name in ("training", "simulation"):
        params_path = config[section_name].get("params_path")
        if params_path is None:
            continue
        params_path = Path(
            _config_path_string(
                params_path,
                f"{section_name}.params_path",
            )
        ).expanduser()
        if not params_path.is_absolute():
            params_path = (config_dir / params_path).resolve()
        else:
            params_path = params_path.resolve()
        config[section_name]["params_path"] = str(params_path)
    return config


def load_config(config_path: str):
    config_path = str(Path(config_path).expanduser().resolve())
    with open(config_path, "r") as f:
        config = json.load(f)
    validate_config(config)
    return resolve_config_paths(config, config_path)


def compute_l2_penalty(model_params):
    leaves = jax.tree_util.tree_leaves(model_params)
    return sum(jnp.sum(jnp.square(x)) for x in leaves)


def count_params(model_params):
    return sum(p.size for p in jax.tree_util.tree_leaves(model_params))


def resolve_training_epoch_count(config: Dict[str, Any]) -> int:
    training_cfg = config["training"]
    if "n_epoch" in training_cfg:
        return max(int(training_cfg["n_epoch"]), 1)

    staged_schedule = training_cfg.get("staged_schedule")
    if isinstance(staged_schedule, list) and staged_schedule:
        staged_epochs = [int(stage["n_epoch"]) for stage in staged_schedule if "n_epoch" in stage]
        if staged_epochs:
            return max(max(staged_epochs), 1)

    segmented_cfg = training_cfg.get("segmented_overlap")
    if isinstance(segmented_cfg, dict) and bool(segmented_cfg.get("enabled", False)):
        segmented_schedule = segmented_cfg.get("stage_schedule")
        if isinstance(segmented_schedule, list) and segmented_schedule:
            segmented_epochs = [int(stage["n_epoch"]) for stage in segmented_schedule if "n_epoch" in stage]
            if segmented_epochs:
                return max(sum(segmented_epochs), 1)

    warmup_steps = int(config["optimizer"].get("warmup_steps", 0))
    return max(warmup_steps + 1, 1)


def make_lr_schedule(config: Dict[str, Any]):
    warmup_steps = int(config["optimizer"]["warmup_steps"])
    decay_steps = max(resolve_training_epoch_count(config), warmup_steps + 1)
    base_schedule = opx.warmup_cosine_decay_schedule(
        init_value=config["optimizer"]["init_value"],
        peak_value=config["optimizer"]["peak_value"],
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=config["optimizer"]["end_value"],
    )
    step_divisor = max(1, int(config["training"].get("lr_schedule_step_divisor", 1)))
    if step_divisor == 1:
        return base_schedule

    def divided_schedule(count):
        return base_schedule(jnp.floor_divide(jnp.asarray(count), step_divisor))

    return divided_schedule


def create_optimizer(config: Dict[str, Any]):
    lr_schedule = make_lr_schedule(config)
    optimizer_type = config["optimizer"]["type"]
    if optimizer_type == "adam":
        beta1 = config["optimizer"].get("beta1", 0.9)
        beta2 = config["optimizer"].get("beta2", 0.999)
        return opx.chain(
            opx.clip_by_global_norm(config["optimizer"]["clip_norm"]),
            opx.adam(learning_rate=lr_schedule, b1=beta1, b2=beta2),
        )

    decay = float(config["optimizer"].get("decay", 0.99))
    return opx.chain(
        opx.clip_by_global_norm(config["optimizer"]["clip_norm"]),
        opx.rmsprop(learning_rate=lr_schedule, decay=decay, eps=1e-8),
    )


def create_train_state(
    config: Dict[str, Any],
    key: jax.Array,
    model: nn.Module,
    sample_lnOmega_real: jnp.ndarray,
    sample_alpha_real: jnp.ndarray,
    sample_beta_real: jnp.ndarray,
    sample_t: float,
    physical_params: jnp.ndarray,
    params_path: Optional[str] = None,
):
    params = model.init(
        key,
        sample_lnOmega_real,
        sample_alpha_real,
        sample_beta_real,
        sample_t,
        physical_params,
    )
    if params_path is not None:
        with open(params_path, "rb") as f:
            params = flax.serialization.from_bytes(params, f.read())

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=create_optimizer(config),
    )


def prepare_output_dirs(save_dir: str, clean_start: bool):
    root = Path(save_dir).expanduser()
    if clean_start:
        _validate_recursive_delete_target(root)
        remove_folders(root)
    for subdir in ("", "train", "simulation"):
        os.makedirs(root / subdir, exist_ok=True)


def save_parameters(params, params_path: str):
    path = ensure_parent_dir(params_path)
    with path.open("wb") as f:
        f.write(flax.serialization.to_bytes(params))


def resolve_checkpoint_params_path(
    config: Dict[str, Any],
    save_dir: str,
    purpose: str = "simulation",
) -> str:
    section_cfg = config.get(purpose, {})
    explicit_path = section_cfg.get("params_path")
    if explicit_path:
        return explicit_path

    del config
    return os.path.join(save_dir, "train", "model_params.msgpack")


def _to_python(value):
    if isinstance(value, dict):
        return {k: _to_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_python(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return value.tolist()
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def save_json(data: Dict[str, Any], path: str):
    path_obj = ensure_parent_dir(path)
    with path_obj.open("w") as f:
        json.dump(_to_python(data), f, indent=2)


def cast_array_for_storage(array: Any, precision: str = "runtime"):
    arr = np.asarray(array)
    precision = str(precision).strip().lower()
    if precision == "runtime":
        return arr
    if precision == "float32":
        if np.issubdtype(arr.dtype, np.complexfloating):
            return arr.astype(np.complex64, copy=False)
        if np.issubdtype(arr.dtype, np.floating):
            return arr.astype(np.float32, copy=False)
        return arr
    if precision == "float64":
        if np.issubdtype(arr.dtype, np.complexfloating):
            return arr.astype(np.complex128, copy=False)
        if np.issubdtype(arr.dtype, np.floating):
            return arr.astype(np.float64, copy=False)
        return arr
    raise ValueError(f"Unsupported storage precision '{precision}'")


def prepare_arrays_for_storage(arrays: Dict[str, Any], precision: str = "runtime"):
    return {k: cast_array_for_storage(v, precision=precision) for k, v in arrays.items()}


def save_npz(path: str, arrays: Dict[str, Any], *, compressed: bool = False, precision: str = "runtime"):
    path_obj = ensure_parent_dir(path)
    prepared = prepare_arrays_for_storage(arrays, precision=precision)
    if compressed:
        np.savez_compressed(path_obj, **prepared)
    else:
        np.savez(path_obj, **prepared)


def save_array_archive(
    path: str,
    arrays: Dict[str, Any],
    *,
    archive_format: Optional[str] = None,
    compressed: bool = False,
    precision: str = "runtime",
):
    prepared = prepare_arrays_for_storage(arrays, precision=precision)
    archive_format = infer_archive_format(path, archive_format=archive_format)
    if archive_format == "npz":
        save_npz(path, prepared, compressed=compressed, precision="runtime")
        return
    if archive_format == "zarr":
        _save_zarr(path, prepared, compressed=compressed)
        return
    raise ValueError(f"Unsupported archive format '{archive_format}'")


__all__ = [
    "archive_extension",
    "KeyGenerator",
    "LOSS_TERM_TRACE_THRESHOLD",
    "OPERATOR_MOMENT_DEFAULT_ORDER",
    "OPERATOR_MOMENT_MAX_ORDER",
    "OPERATOR_MOMENT_SPECS",
    "VALID_GAUGE_MODES",
    "compute_l2_penalty",
    "count_params",
    "create_optimizer",
    "create_train_state",
    "ensure_parent_dir",
    "format_end_time",
    "infer_archive_format",
    "load_array_archive",
    "load_config",
    "loss_prefactor_is_active",
    "make_lr_schedule",
    "normalize_neural_gauge_components",
    "prepare_arrays_for_storage",
    "prepare_output_dirs",
    "remove_folders",
    "require_archive_backend",
    "resolve_checkpoint_params_path",
    "resolve_config_paths",
    "resolve_pareto_k_tail_count",
    "cast_array_for_storage",
    "save_array_archive",
    "save_json",
    "save_npz",
    "save_parameters",
    "to_scalar_float",
    "tree_has_nonfinite",
    "validate_config",
]
