# jax
import jax
from jax import lax
import jax.numpy as jnp
from jax import random
import jax.scipy as jsp

# flax
import flax
import flax.linen as nn
from flax.training import train_state

# optax
import optax as opx

# general
import json
import os
from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import scipy



# ---- master dtype control ----
USE_FLOAT64 = os.environ.get("NSGR_USE_FLOAT64", "1").strip().lower() in {"1", "true", "yes", "on"}
NN_USE_FLOAT64 = os.environ.get("NSGR_NN_USE_FLOAT64", "1").strip().lower() in {"1", "true", "yes", "on"}

DTYPE  = jnp.float64 if USE_FLOAT64 else jnp.float32
CDTYPE = jnp.complex128 if USE_FLOAT64 else jnp.complex64
NNDTYPE = jnp.float64 if NN_USE_FLOAT64 else jnp.float32
OBSERVABLE_OCCUPATION_FLOOR = 1e-8 if USE_FLOAT64 else 1e-5
SDE_ROOT_RTOL_DEFAULT = 1.0e-9 if USE_FLOAT64 else 1.0e-5
SDE_ROOT_ATOL_DEFAULT = 1.0e-11 if USE_FLOAT64 else 1.0e-7

# Enable 64-bit whenever either the solver or neural network requests it.
# Otherwise NSGR_NN_USE_FLOAT64=1 with a float32 solver would declare NNDTYPE as
# float64 while JAX silently truncated every neural array to float32.

if USE_FLOAT64 or NN_USE_FLOAT64:
    from jax import config
    config.update("jax_enable_x64", True)  # enable float64

def to_r(x): return jnp.asarray(x, dtype=DTYPE)
def to_c(re, im): return lax.complex(to_r(re), to_r(im))    # complex from two reals (no 1j!)
def cplx_i(): return to_c(0.0, 1.0)                         # returns 1j with CDTYPE
