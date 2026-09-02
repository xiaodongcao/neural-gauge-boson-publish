from .lib_preinclude import *
from dataclasses import dataclass, field


DISTANCE_SHELL_RTOL = 1.0e-10
DISTANCE_SHELL_ATOL = 1.0e-12


def _same_distance_shell(left, right):
    return np.isclose(
        left,
        right,
        rtol=DISTANCE_SHELL_RTOL,
        atol=DISTANCE_SHELL_ATOL,
    )


def broadcast_site_param(param, num_site: int, dtype):
    return jnp.broadcast_to(jnp.asarray(param, dtype=dtype), (num_site,))


def apply_hopping_matrix(J, x):
    if isinstance(J, dict):
        edge_dst = jnp.asarray(J["edge_dst"], dtype=jnp.int32)
        edge_src = jnp.asarray(J["edge_src"], dtype=jnp.int32)
        edge_weight = jnp.asarray(J["edge_weight"], dtype=x.dtype)
        if edge_src.shape[0] == 0:
            return jnp.zeros_like(x)
        messages = x[..., edge_src] * edge_weight
        return jnp.zeros_like(x).at[..., edge_dst].add(messages)
    J = jnp.asarray(J, dtype=x.dtype)
    if J.ndim == 0:
        return jnp.zeros_like(x)
    return x @ jnp.swapaxes(J, -1, -2) if x.ndim > 1 else J @ x


def conjugate_hopping_operator(J):
    if isinstance(J, dict):
        return {
            "edge_dst": J["edge_dst"],
            "edge_src": J["edge_src"],
            "edge_weight": jnp.conj(jnp.asarray(J["edge_weight"])),
        }
    return jnp.conj(J)


@dataclass
class Lattice:
    """Lattice geometry, physical parameters, hopping graph, and ED utilities."""

    Nx: int
    Ny: int
    prim_x: Tuple[float, ...]
    prim_y: Tuple[float, ...]
    U: float
    gamma: float
    F: complex
    Delta: float
    n0: float
    hopping_matrix: jnp.ndarray
    boundary_x: str = "open"
    boundary_y: str = "open"
    _ed_operator_cache: Dict[int, Dict[str, Any]] = field(default_factory=dict, init=False, repr=False, compare=False)
    _hopping_operator_cache: Dict[float, Any] = field(default_factory=dict, init=False, repr=False, compare=False)
    _shell_pair_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    @staticmethod
    def _validate_boundary_condition(boundary: str, axis_name: str):
        boundary = str(boundary)
        if boundary not in {"open", "periodic"}:
            raise ValueError(f"{axis_name} must be 'open' or 'periodic', got {boundary!r}")
        return boundary

    @staticmethod
    def _boundary_shifts(size: int, boundary: str):
        if boundary == "periodic":
            return (-size, 0, size)
        return (0,)

    @staticmethod
    def _wrapped_displacement_vector(
        ix: int,
        iy: int,
        jx: int,
        jy: int,
        Nx: int,
        Ny: int,
        prim_vec_x,
        prim_vec_y,
        boundary_x: str,
        boundary_y: str,
    ):
        """Return the minimum-image displacement vector between two sites."""
        prim_vec_x = np.asarray(prim_vec_x, dtype=float)
        prim_vec_y = np.asarray(prim_vec_y, dtype=float)
        boundary_x = Lattice._validate_boundary_condition(boundary_x, "boundary_x")
        boundary_y = Lattice._validate_boundary_condition(boundary_y, "boundary_y")

        base_dx = jx - ix
        base_dy = jy - iy
        best_vec = None
        best_norm_sq = None

        for shift_x in Lattice._boundary_shifts(Nx, boundary_x):
            for shift_y in Lattice._boundary_shifts(Ny, boundary_y):
                dx = base_dx + shift_x
                dy = base_dy + shift_y
                vec = dx * prim_vec_x + dy * prim_vec_y
                norm_sq = float(np.dot(vec, vec))
                if best_norm_sq is None or norm_sq < best_norm_sq:
                    best_norm_sq = norm_sq
                    best_vec = vec

        return best_vec

    @staticmethod
    def _distance_shells(
        prim_vec_x,
        prim_vec_y,
        Nx: int,
        Ny: int,
        num_shells: int,
        boundary_x: str = "open",
        boundary_y: str = "open",
    ):
        """Compute unique real-space distance shells for shell-based hopping input."""
        distances = []
        for ix in range(Nx):
            for iy in range(Ny):
                row = ix * Ny + iy
                for jx in range(Nx):
                    for jy in range(Ny):
                        col = jx * Ny + jy
                        if col <= row:
                            continue
                        disp = Lattice._wrapped_displacement_vector(
                            ix,
                            iy,
                            jx,
                            jy,
                            Nx,
                            Ny,
                            prim_vec_x,
                            prim_vec_y,
                            boundary_x,
                            boundary_y,
                        )
                        distances.append(float(np.linalg.norm(disp)))

        unique = []
        for distance in sorted(distances):
            if not unique or not _same_distance_shell(distance, unique[-1]):
                unique.append(distance)
        unique = np.asarray(unique, dtype=float)
        return unique[:num_shells]

    @staticmethod
    def build_boson_hopping_matrix(
        Nx: int,
        Ny: int,
        prim_vec_x,
        prim_vec_y,
        hopping_amplitudes,
        boundary_x: str = "open",
        boundary_y: str = "open",
    ):
        """Build a dense hopping matrix from shell amplitudes."""
        hopping_amplitudes = np.asarray(hopping_amplitudes, dtype=float)
        num_site = Nx * Ny
        hopping_matrix = np.zeros((num_site, num_site), dtype=np.complex128)

        if hopping_amplitudes.size == 0:
            return jnp.asarray(hopping_matrix, dtype=CDTYPE)

        prim_vec_x = np.asarray(prim_vec_x, dtype=float)
        prim_vec_y = np.asarray(prim_vec_y, dtype=float)
        boundary_x = Lattice._validate_boundary_condition(boundary_x, "boundary_x")
        boundary_y = Lattice._validate_boundary_condition(boundary_y, "boundary_y")
        # A one-site lattice has no intersite hopping edges. Keep accepting
        # shell amplitudes because benchmark configurations commonly retain
        # this field when reducing a lattice to 1x1.
        if num_site == 1:
            return jnp.asarray(hopping_matrix, dtype=CDTYPE)

        shells = Lattice._distance_shells(
            prim_vec_x,
            prim_vec_y,
            Nx,
            Ny,
            hopping_amplitudes.size,
            boundary_x=boundary_x,
            boundary_y=boundary_y,
        )
        if shells.size != hopping_amplitudes.size:
            raise ValueError(
                "hopping_amplitudes requests "
                f"{hopping_amplitudes.size} distance shell(s), but this "
                f"lattice has only {shells.size} distinct nonzero pair-distance "
                "shell(s)"
            )

        for ix in range(Nx):
            for iy in range(Ny):
                row = ix * Ny + iy
                for jx in range(Nx):
                    for jy in range(Ny):
                        col = jx * Ny + jy
                        if row == col:
                            continue
                        disp = Lattice._wrapped_displacement_vector(
                            ix,
                            iy,
                            jx,
                            jy,
                            Nx,
                            Ny,
                            prim_vec_x,
                            prim_vec_y,
                            boundary_x,
                            boundary_y,
                        )
                        distance = float(np.linalg.norm(disp))
                        matches = np.where(_same_distance_shell(shells, distance))[0]
                        if matches.size:
                            hopping_matrix[row, col] = hopping_amplitudes[matches[0]]

        return jnp.asarray(hopping_matrix, dtype=CDTYPE)

    @staticmethod
    def resolve_boson_hopping_matrix(lat_cfg: Dict[str, Any]):
        """Resolve the hopping matrix from one validated lattice-config section."""
        Nx = int(lat_cfg["Nx"])
        Ny = int(lat_cfg["Ny"])
        num_site = Nx * Ny
        boundary_x = lat_cfg.get("boundary_x", "open")
        boundary_y = lat_cfg.get("boundary_y", "open")

        if "hopping_matrix_real" in lat_cfg or "hopping_matrix_imag" in lat_cfg:
            real = np.asarray(lat_cfg.get("hopping_matrix_real", np.zeros((num_site, num_site))), dtype=float)
            imag = np.asarray(lat_cfg.get("hopping_matrix_imag", np.zeros((num_site, num_site))), dtype=float)
            return jnp.asarray(real + 1j * imag, dtype=CDTYPE)

        if "hopping_matrix" in lat_cfg:
            return jnp.asarray(lat_cfg["hopping_matrix"], dtype=CDTYPE)

        if "hopping_amplitudes" in lat_cfg:
            return Lattice.build_boson_hopping_matrix(
                Nx,
                Ny,
                lat_cfg["prim_x"],
                lat_cfg["prim_y"],
                lat_cfg["hopping_amplitudes"],
                boundary_x=boundary_x,
                boundary_y=boundary_y,
            )

        return jnp.zeros((num_site, num_site), dtype=CDTYPE)

    @classmethod
    def from_config(cls, lat_cfg: Dict[str, Any]):
        return cls(
            Nx=int(lat_cfg["Nx"]),
            Ny=int(lat_cfg["Ny"]),
            prim_x=tuple(lat_cfg["prim_x"]),
            prim_y=tuple(lat_cfg["prim_y"]),
            U=float(lat_cfg["U"]),
            gamma=float(lat_cfg["gamma"]),
            F=complex(lat_cfg["F_real"] + 1j * lat_cfg["F_imag"]),
            Delta=float(lat_cfg["Delta"]),
            n0=float(lat_cfg["n0"]),
            hopping_matrix=cls.resolve_boson_hopping_matrix(lat_cfg),
            boundary_x=cls._validate_boundary_condition(lat_cfg.get("boundary_x", "open"), "boundary_x"),
            boundary_y=cls._validate_boundary_condition(lat_cfg.get("boundary_y", "open"), "boundary_y"),
        )

    def hopping_operator(self, sparsity_threshold: float = 0.5):
        """Return a dense matrix or sparse edge-list operator for phase-space hopping."""
        threshold = float(sparsity_threshold)
        if threshold in self._hopping_operator_cache:
            return self._hopping_operator_cache[threshold]

        hopping = np.asarray(self.hopping_matrix, dtype=np.complex128)
        if hopping.ndim == 0 or hopping.size == 1:
            operator = jnp.asarray(hopping, dtype=CDTYPE)
        else:
            edge_dst, edge_src = np.nonzero(np.abs(hopping) > 0.0)
            dense_size = int(hopping.size)
            use_dense = edge_dst.size > max(1, int(threshold * dense_size))
            if use_dense:
                operator = jnp.asarray(hopping, dtype=CDTYPE)
            else:
                operator = {
                    "edge_dst": jnp.asarray(edge_dst, dtype=jnp.int32),
                    "edge_src": jnp.asarray(edge_src, dtype=jnp.int32),
                    "edge_weight": jnp.asarray(hopping[edge_dst, edge_src], dtype=CDTYPE),
                }
        self._hopping_operator_cache[threshold] = operator
        return operator

    @property
    def num_site(self) -> int:
        return self.Nx * self.Ny

    @property
    def positions(self) -> np.ndarray:
        prim_x = np.asarray(self.prim_x, dtype=float)
        prim_y = np.asarray(self.prim_y, dtype=float)
        coords = []
        for ix in range(self.Nx):
            for iy in range(self.Ny):
                coords.append(ix * prim_x + iy * prim_y)
        return np.asarray(coords, dtype=float)

    def shell_pair_indices(self, shell: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return directed site-pair indices for local, nearest, or next-nearest shells."""

        shell_key = str(shell).strip().lower()
        aliases = {
            "onsite": "local",
            "diag": "local",
            "nearest": "nn",
            "nearest_neighbor": "nn",
            "nearest-neighbor": "nn",
            "next_nearest": "nnn",
            "next-nearest": "nnn",
            "next_nearest_neighbor": "nnn",
            "next-nearest-neighbor": "nnn",
        }
        shell_key = aliases.get(shell_key, shell_key)
        if shell_key not in {"local", "nn", "nnn"}:
            raise ValueError(f"Unsupported shell {shell!r}; expected local, nn, or nnn")
        if shell_key in self._shell_pair_cache:
            return self._shell_pair_cache[shell_key]

        num_site = self.num_site
        if shell_key == "local":
            indices = np.arange(num_site, dtype=np.int32)
            pairs = (indices, indices)
            self._shell_pair_cache[shell_key] = pairs
            return pairs

        shell_number = {"nn": 1, "nnn": 2}[shell_key]
        prim_x = np.asarray(self.prim_x, dtype=float)
        prim_y = np.asarray(self.prim_y, dtype=float)
        shells = self._distance_shells(
            prim_x,
            prim_y,
            self.Nx,
            self.Ny,
            shell_number,
            boundary_x=self.boundary_x,
            boundary_y=self.boundary_y,
        )
        if shells.size < shell_number:
            empty = np.zeros((0,), dtype=np.int32)
            pairs = (empty, empty)
            self._shell_pair_cache[shell_key] = pairs
            return pairs

        target_distance = float(shells[shell_number - 1])
        left = []
        right = []
        for ix in range(self.Nx):
            for iy in range(self.Ny):
                row = ix * self.Ny + iy
                for jx in range(self.Nx):
                    for jy in range(self.Ny):
                        col = jx * self.Ny + jy
                        if row == col:
                            continue
                        disp = self._wrapped_displacement_vector(
                            ix,
                            iy,
                            jx,
                            jy,
                            self.Nx,
                            self.Ny,
                            prim_x,
                            prim_y,
                            self.boundary_x,
                            self.boundary_y,
                        )
                        if _same_distance_shell(
                            float(np.linalg.norm(disp)),
                            target_distance,
                        ):
                            left.append(row)
                            right.append(col)

        pairs = (np.asarray(left, dtype=np.int32), np.asarray(right, dtype=np.int32))
        self._shell_pair_cache[shell_key] = pairs
        return pairs

    def physical_params(self, dtype=NNDTYPE, **overrides):
        U = overrides.get("U", self.U)
        gamma = overrides.get("gamma", self.gamma)
        n0 = overrides.get("n0", self.n0)
        F = overrides.get("F", self.F)
        Delta = overrides.get("Delta", self.Delta)
        return jnp.asarray([U, gamma, n0, np.real(F), np.imag(F), Delta], dtype=dtype)

    def initialize_phase_space(self, num_walker: int, n0: Optional[float] = None):
        from .dynamics_kernel import initialize_phase_space_variables

        n0_value = self.n0 if n0 is None else n0
        return initialize_phase_space_variables(num_walker, self.num_site, DTYPE(n0_value))

    def coherent_product_amplitudes(self, n0: Optional[float] = None):
        n0_value = self.n0 if n0 is None else n0
        n0_value = float(n0_value)
        if n0_value < 0.0:
            raise ValueError("coherent-product initialization requires n0 >= 0")
        amplitude = np.sqrt(n0_value).astype(np.complex128)
        return np.full((self.num_site,), amplitude, dtype=np.complex128)

    @staticmethod
    def _single_site_coherent_state(amplitude: complex, n_cut: int):
        dim = int(n_cut) + 1
        n = np.arange(dim, dtype=float)
        coeff = np.exp(-0.5 * abs(amplitude) ** 2) * np.power(amplitude, n)
        coeff = coeff / np.sqrt(scipy.special.factorial(n, exact=False))
        coeff = np.asarray(coeff, dtype=np.complex128)
        norm = np.linalg.norm(coeff)
        if norm <= 0.0:
            raise RuntimeError("Truncated coherent-state vector has zero norm.")
        return coeff / norm

    def coherent_product_state_vector(
        self,
        n_cut: int,
        amplitudes: Optional[Sequence[complex]] = None,
        n0: Optional[float] = None,
    ):
        if amplitudes is None:
            amplitudes = self.coherent_product_amplitudes(n0=n0)
        amplitudes = np.broadcast_to(np.asarray(amplitudes, dtype=np.complex128), (self.num_site,))
        state = self._single_site_coherent_state(amplitudes[0], n_cut)
        for amplitude in amplitudes[1:]:
            state = np.kron(state, self._single_site_coherent_state(amplitude, n_cut))
        return np.asarray(state, dtype=np.complex128)

    def coherent_product_density_matrix(
        self,
        n_cut: int,
        amplitudes: Optional[Sequence[complex]] = None,
        n0: Optional[float] = None,
    ):
        state = self.coherent_product_state_vector(n_cut=n_cut, amplitudes=amplitudes, n0=n0)
        return np.outer(state, np.conjugate(state)).astype(np.complex128)

    def reciprocal_k_vectors(self) -> np.ndarray:
        a1 = np.asarray(self.prim_x, dtype=float)
        a2 = np.asarray(self.prim_y, dtype=float)
        lattice_matrix = np.stack([a1, a2], axis=1)
        reciprocal = 2.0 * np.pi * np.linalg.inv(lattice_matrix).T
        k_vectors = []
        for mx in range(self.Nx):
            for my in range(self.Ny):
                k_vectors.append((mx / self.Nx) * reciprocal[:, 0] + (my / self.Ny) * reciprocal[:, 1])
        return np.asarray(k_vectors, dtype=float)

    def _single_site_annihilation(self, n_cut: int):
        dim = n_cut + 1
        data = np.sqrt(np.arange(1, dim, dtype=float))
        return scipy.sparse.diags(data, 1, shape=(dim, dim), dtype=np.complex128).tocsr()

    def _operator_bundle(self, n_cut: int):
        n_cut = int(n_cut)
        bundle = self._ed_operator_cache.get(n_cut)
        if bundle is not None:
            return bundle

        a_local = self._single_site_annihilation(n_cut)
        ident_local = scipy.sparse.identity(n_cut + 1, format="csr", dtype=np.complex128)

        annihilation_ops = []
        for site in range(self.num_site):
            factors = [ident_local] * self.num_site
            factors[site] = a_local
            op = factors[0]
            for factor in factors[1:]:
                op = scipy.sparse.kron(op, factor, format="csr")
            annihilation_ops.append(op.tocsr())

        creation_ops = tuple(op.getH().tocsr() for op in annihilation_ops)
        annihilation_ops = tuple(annihilation_ops)
        number_ops = tuple((adag @ a).tocsr() for adag, a in zip(creation_ops, annihilation_ops))
        interaction_ops = tuple((adag @ adag @ a @ a).tocsr() for adag, a in zip(creation_ops, annihilation_ops))
        g1_ops = tuple(
            tuple((creation_ops[i] @ annihilation_ops[j]).tocsr() for j in range(self.num_site))
            for i in range(self.num_site)
        )
        g2_ops = tuple(
            tuple((creation_ops[i] @ creation_ops[j] @ annihilation_ops[j] @ annihilation_ops[i]).tocsr() for j in range(self.num_site))
            for i in range(self.num_site)
        )
        dim = (n_cut + 1) ** self.num_site
        bundle = {
            "dim": dim,
            "identity": scipy.sparse.identity(dim, format="csr", dtype=np.complex128),
            "annihilation_ops": annihilation_ops,
            "creation_ops": creation_ops,
            "number_ops": number_ops,
            "interaction_ops": interaction_ops,
            "g1_ops": g1_ops,
            "g2_ops": g2_ops,
        }
        self._ed_operator_cache[n_cut] = bundle
        return bundle

    def site_operators(self, n_cut: int):
        bundle = self._operator_bundle(n_cut)
        return bundle["annihilation_ops"], bundle["creation_ops"], bundle["number_ops"]

    def hamiltonian(
        self,
        n_cut: int,
        U: Optional[float] = None,
        Delta: Optional[float] = None,
        F: Optional[complex] = None,
        hopping_matrix: Optional[np.ndarray] = None,
    ):
        U = self.U if U is None else U
        Delta = self.Delta if Delta is None else Delta
        F = self.F if F is None else F
        J = np.asarray(self.hopping_matrix if hopping_matrix is None else hopping_matrix, dtype=np.complex128)

        bundle = self._operator_bundle(n_cut)
        annihilation_ops = bundle["annihilation_ops"]
        creation_ops = bundle["creation_ops"]
        number_ops = bundle["number_ops"]
        interaction_ops = bundle["interaction_ops"]
        g1_ops = bundle["g1_ops"]
        H = scipy.sparse.csr_matrix((bundle["dim"], bundle["dim"]), dtype=np.complex128)

        for site in range(self.num_site):
            a = annihilation_ops[site]
            adag = creation_ops[site]
            n_op = number_ops[site]
            H = H - Delta * n_op
            H = H + 0.5 * U * interaction_ops[site]
            H = H + F * adag + np.conjugate(F) * a

        nonzero_hopping = np.argwhere(np.abs(J) > 0.0)
        for i, j in nonzero_hopping:
            H = H - J[i, j] * g1_ops[i][j]

        return H.tocsr(), annihilation_ops, creation_ops, number_ops

    def collapse_operators(self, n_cut: int, gamma: Optional[float] = None):
        gamma = self.gamma if gamma is None else gamma
        annihilation_ops = self._operator_bundle(n_cut)["annihilation_ops"]
        return [np.sqrt(gamma) * a for a in annihilation_ops]

    def liouvillian(
        self,
        n_cut: int,
        U: Optional[float] = None,
        gamma: Optional[float] = None,
        Delta: Optional[float] = None,
        F: Optional[complex] = None,
        hopping_matrix: Optional[np.ndarray] = None,
    ):
        gamma = self.gamma if gamma is None else gamma
        bundle = self._operator_bundle(n_cut)
        H, _, _, _ = self.hamiltonian(n_cut=n_cut, U=U, Delta=Delta, F=F, hopping_matrix=hopping_matrix)
        ident = bundle["identity"]
        collapses = [np.sqrt(gamma) * a for a in bundle["annihilation_ops"]]
        L = -1j * (scipy.sparse.kron(ident, H, format="csr") - scipy.sparse.kron(H.T, ident, format="csr"))

        for c_op in collapses:
            cdag_c = (c_op.getH() @ c_op).tocsr()
            L = L + scipy.sparse.kron(c_op.conjugate(), c_op, format="csr")
            L = L - 0.5 * scipy.sparse.kron(ident, cdag_c, format="csr")
            L = L - 0.5 * scipy.sparse.kron(cdag_c.T, ident, format="csr")

        return L.tocsr()

    def steady_state(
        self,
        n_cut: int,
        U: Optional[float] = None,
        gamma: Optional[float] = None,
        Delta: Optional[float] = None,
        F: Optional[complex] = None,
        hopping_matrix: Optional[np.ndarray] = None,
        dense_dim_limit: int = 4096,
    ):
        L = self.liouvillian(
            n_cut=n_cut,
            U=U,
            gamma=gamma,
            Delta=Delta,
            F=F,
            hopping_matrix=hopping_matrix,
        )
        dim = int(round(np.sqrt(L.shape[0])))

        if L.shape[0] <= dense_dim_limit:
            evals, evecs = scipy.linalg.eig(L.toarray())
            idx = int(np.argmin(np.abs(evals)))
            rho_vec = evecs[:, idx]
        else:
            evals, evecs = scipy.sparse.linalg.eigs(L, k=1, sigma=0.0, which="LM")
            rho_vec = evecs[:, 0]

        rho = rho_vec.reshape((dim, dim), order="F")
        rho = 0.5 * (rho + rho.conjugate().T)
        trace = np.trace(rho)
        if abs(trace) < 1e-12:
            raise RuntimeError("Steady-state trace is numerically zero.")
        return rho / trace

    def evolve_density_matrix(
        self,
        times: Sequence[float],
        n_cut: int,
        rho0: Optional[np.ndarray] = None,
        U: Optional[float] = None,
        gamma: Optional[float] = None,
        Delta: Optional[float] = None,
        F: Optional[complex] = None,
        hopping_matrix: Optional[np.ndarray] = None,
    ):
        times = np.asarray(times, dtype=float)
        if times.ndim != 1:
            raise ValueError("times must be a one-dimensional sequence")

        bundle = self._operator_bundle(n_cut)
        dim = bundle["dim"]
        if rho0 is None:
            rho0 = np.zeros((dim, dim), dtype=np.complex128)
            rho0[0, 0] = 1.0
        if times.size == 0:
            return []

        L = self.liouvillian(
            n_cut=n_cut,
            U=U,
            gamma=gamma,
            Delta=Delta,
            F=F,
            hopping_matrix=hopping_matrix,
        )
        rho0_vec = np.asarray(rho0, dtype=np.complex128).reshape(-1, order="F")

        order = np.argsort(times)
        sorted_times = times[order]
        sorted_density_matrices = [None] * len(sorted_times)

        if len(sorted_times) == 1:
            propagated = [scipy.sparse.linalg.expm_multiply(L * float(sorted_times[0]), rho0_vec)]
        else:
            dt_sorted = np.diff(sorted_times)
            evenly_spaced = np.allclose(dt_sorted, dt_sorted[0], rtol=1e-10, atol=1e-12)
            if evenly_spaced:
                propagated = scipy.sparse.linalg.expm_multiply(
                    L,
                    rho0_vec,
                    start=float(sorted_times[0]),
                    stop=float(sorted_times[-1]),
                    num=len(sorted_times),
                    endpoint=True,
                )
            else:
                propagated = []
                current_vec = None
                previous_t = None
                for t in sorted_times:
                    if previous_t is None:
                        current_vec = scipy.sparse.linalg.expm_multiply(L * float(t), rho0_vec)
                    else:
                        current_vec = scipy.sparse.linalg.expm_multiply(L * float(t - previous_t), current_vec)
                    propagated.append(current_vec)
                    previous_t = t

        propagated = np.atleast_2d(np.asarray(propagated))
        for idx, rho_t_vec in enumerate(propagated):
            rho_t = np.asarray(rho_t_vec).reshape((dim, dim), order="F")
            rho_t = 0.5 * (rho_t + rho_t.conjugate().T)
            trace = np.trace(rho_t)
            sorted_density_matrices[idx] = rho_t / trace

        densities = [None] * len(times)
        for sorted_idx, original_idx in enumerate(order):
            densities[original_idx] = sorted_density_matrices[sorted_idx]
        return densities

    def equal_time_observables_from_rho(
        self,
        rho: np.ndarray,
        n_cut: int,
        initial_beta: Optional[Sequence[complex]] = None,
    ):
        """Benchmark observables from an ED density matrix with the same conventions as Measurements."""
        bundle = self._operator_bundle(n_cut)
        annihilation_ops = bundle["annihilation_ops"]
        number_ops = bundle["number_ops"]
        g1_ops = bundle["g1_ops"]
        g2_ops = bundle["g2_ops"]

        A = np.asarray([self.expectation(rho, a_op) for a_op in annihilation_ops], dtype=np.complex128)
        density = np.asarray([self.expectation(rho, n_op) for n_op in number_ops], dtype=np.complex128)

        G1 = np.zeros((self.num_site, self.num_site), dtype=np.complex128)
        G2 = np.zeros((self.num_site, self.num_site), dtype=np.complex128)
        for i in range(self.num_site):
            for j in range(self.num_site):
                G1[i, j] = self.expectation(rho, g1_ops[i][j])
                G2[i, j] = self.expectation(rho, g2_ops[i][j])

        N = np.real(density)
        G2_local = np.real(np.diag(G2))
        N_safe = np.maximum(N, 1e-12)
        g2_local = np.full_like(G2_local, np.nan, dtype=float)
        coherence_fraction = np.full_like(N, np.nan, dtype=float)
        valid_g2 = N >= float(OBSERVABLE_OCCUPATION_FLOOR)
        valid_cf = N >= float(OBSERVABLE_OCCUPATION_FLOOR)
        g2_local[valid_g2] = G2_local[valid_g2] / (N_safe[valid_g2] ** 2)
        coherence_fraction[valid_cf] = np.abs(A[valid_cf]) ** 2 / N_safe[valid_cf]

        observables = {
            "rho": rho,
            "A": A,
            "density": density,
            "N": N,
            "G1": G1,
            "G2": G2,
            "G2_local": G2_local,
            "g2_local": g2_local,
            "coherence_fraction": coherence_fraction,
        }
        if initial_beta is not None:
            initial_beta = np.broadcast_to(np.asarray(initial_beta, dtype=np.complex128), (self.num_site,))
            observables["G1_initial"] = initial_beta[:, None] * A[None, :]
        return observables

    def steady_state_observables(
        self,
        n_cut: int,
        U: Optional[float] = None,
        gamma: Optional[float] = None,
        Delta: Optional[float] = None,
        F: Optional[complex] = None,
        hopping_matrix: Optional[np.ndarray] = None,
        dense_dim_limit: int = 4096,
    ):
        rho = self.steady_state(
            n_cut=n_cut,
            U=U,
            gamma=gamma,
            Delta=Delta,
            F=F,
            hopping_matrix=hopping_matrix,
            dense_dim_limit=dense_dim_limit,
        )
        return self.equal_time_observables_from_rho(rho, n_cut=n_cut)

    def time_evolution_observables(
        self,
        times: Sequence[float],
        n_cut: int,
        rho0: Optional[np.ndarray] = None,
        initial_beta: Optional[Sequence[complex]] = None,
        U: Optional[float] = None,
        gamma: Optional[float] = None,
        Delta: Optional[float] = None,
        F: Optional[complex] = None,
        hopping_matrix: Optional[np.ndarray] = None,
    ):
        rho_traj = self.evolve_density_matrix(
            times=times,
            n_cut=n_cut,
            rho0=rho0,
            U=U,
            gamma=gamma,
            Delta=Delta,
            F=F,
            hopping_matrix=hopping_matrix,
        )
        return [
            self.equal_time_observables_from_rho(
                rho_t,
                n_cut=n_cut,
                initial_beta=initial_beta,
            )
            for rho_t in rho_traj
        ]

    @staticmethod
    def expectation(rho: np.ndarray, operator):
        if scipy.sparse.issparse(operator):
            return operator.multiply(np.asarray(rho).T).sum()
        return np.trace(rho @ operator)
