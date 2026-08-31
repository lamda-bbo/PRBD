import math
import time
from functools import partial

import numpy as np
import numpyro
import numpyro.distributions as dist

from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS
from .util import chunk_vmap

root_five = math.sqrt(5.0)
five_thirds = 5.0 / 3.0

