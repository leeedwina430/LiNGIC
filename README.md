# LINGIC

This folder contains the LINGIC independence test and a minimal reproducible
demo against the standard HSIC baseline.

## Files

| File | Purpose |
| --- | --- |
| `LINGIC.py` | Core file |
| `demo.py` | Mminimal demo with comparison between HSIC and LINGIC |
| `README.md` | Installation, API, DGP, and reproduction instructions. |

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy scipy tqdm causal-learn
```

## Use LINGIC

Inputs must be two-dimensional NumPy arrays with the same number of rows:

```python
import numpy as np

from LINGIC import gauss_poly_HSIC_twice_pvalue

rng = np.random.default_rng(7)
x = rng.standard_normal((200, 1))
y = x**2 + 0.2 * rng.standard_normal((200, 1))

# The sequential permutation routine uses NumPy's global random state.
np.random.seed(42)
p_value = gauss_poly_HSIC_twice_pvalue(x, y, N_PERM=500)
print(f"p-value: {p_value:.4f}")
```


## Run the demo

```bash
python demo.py
```

Default settings:

- 120 observations;
- 3 Student-t sources with 3 degrees of freedom;
- 200 Monte Carlo repetitions under H0 and H1;
- 199 LINGIC permutations per data set; and
- significance level 0.05.


## DGP


Under H0, two independent Student-t source matrices are used:

```text
X = Z_X w_X,  Y = Z_Y w_Y,  with Z_X independent of Z_Y.
```

Under H1, both mixtures use the same source matrix:

```text
X = Z w_X,  Y = Z w_Y.
```
