# py-BQE

Bayesian Quantal Estimation (JAX / NumPyro)

Python implementation of the Bayesian method for jointly estimating quantal and short-term plasticity parameters from trains of synaptic responses, based on:

> Bird AD, Wall MJ & Richardson MJE (2016)  
> Bayesian Inference of Synaptic Quantal Parameters from Correlated Vesicle Release  
> *Frontiers in Computational Neuroscience* 10:116  
> https://doi.org/10.3389/fncom.2016.00116

This port replaces the original discrete-grid Metropolis-Hastings sampler with continuous Hamiltonian Monte-Carlo (NUTS from NumPyro) and uses JAX for vectorised likelihood calculation.

Like the original version, it is intended for properly correlated quantal analysis of trains rather than classical mean-variance or simple least-squares STP fits.

---

## Functionality

- Exact likelihood that accounts for serial correlations between successive responses in a train.
- Joint estimation of quantal parameters (`a`, `σ_a`, `σ_b`) and short-term plasticity parameters.
- Two of the original models are implemented:
    - **daf** - depression + facilitation
    - **dep** - pure depression
- Number of release sites `n` is marginalised (posterior `P(n | data)` is returned).
- Built-in simulator for generating synthetic data with known ground truth.
- Support for both voltage (mV) and current (pA) recordings via different default prior ranges.
- Diagnostics (R-hat, ESS) via NumPyro.

---

## Installation

GPU users: install JAX according to the official [JAX installation instructions](https://docs.jax.dev/en/latest/installation.html) *before* installing `py-BQE`, since the following `pip install` on its own will get the CPU build of JAX.


```bash
git clone https://github.com/AMikroulis/py-BQE
cd py-BQE
pip install -e .
```

---

## Data format

The input file is a plain text matrix:

```
time_ms   sweep1   sweep2   sweep3   ...
0.0       0.416    0.392    0.451    ...
50.0      0.299    0.308    0.281    ...
100.0     0.179    0.204    0.163    ...
...
```

- First column = pulse times in ms
- Remaining columns = one amplitude per sweep (EPSP or EPSC)
- Missing values are not supported

---

## Quick start
### 1. Generate synthetic data (optional)

```
bqe --simulate synthetic_daf.txt \
    --sim-n 5 --sim-p 0.25 --sim-f 0.4 --sim-tauf 80 \
    --sim-tD 250 --sim-a 0.18 --sim-siga 0.04 --sim-sigb 0.015 \
    --sim-pulses 8 --sim-isi 50 --sim-sweeps 40 --sim-seed 64
```

(`python -m bqe --simulate ...` also works.)

### 2. Fit a model

Depression-only model:

```
bqe synthetic_daf.txt \
    --model dep --n-max 12 --units mV \
    --warmup 800 --samples 1500 --chains 4
```

Depression + facilitation model:

```
bqe synthetic_daf.txt \
    --model daf --n-max 12 --units mV \
    --warmup 1000 --samples 2000 --chains 4
```

The script will:

1. Plot the raw data and amplitude histogram
2. Run NUTS and print a posterior summary
3. Compute and display the marginal posterior over n
4. Show 1-D and 2-D marginal posterior plots

---

## Command-line options


|Option|Default|Description|
|------|------|-----------|
|filename|-|Input data file|
|--model|dep|dep or daf|
|--n-max|10|Marginalise n over 1 … n-max|
|--units|mV|mV or pA (sets default prior ranges)|
|--warmup|500|NUTS warmup steps|
|--samples|1000|Posterior samples per chain|
|--chains|4|Number of chains|
|--a-range LO HI|-|Override prior for quantal amplitude|
|--siga-range LO HI|-|Override prior for quantal SD|
|--sigb-range LO HI|-|Override prior for baseline noise|
|--tauf-range LO HI|-|Override prior for facilitation time constant (daf only)|
|--target-accept|0.8|NUTS target acceptance probability|


---

## Interpreting the output

- `Posterior summary` - means, standard deviations, and 90 % highest-density intervals for all continuous parameters.
- `P(n | data)` - discrete posterior over the number of release sites.
- `u1` (daf model only) - effective release probability on the second pulse (p + f·(1-p)).
- `2-D marginals` - useful for spotting strong correlations (e.g. a <-> σ_a, p <-> tD).

Always check the NumPyro diagnostics (R-hat ≈ 1.0, reasonable ESS). If many divergences appear, try raising --target-accept to 0.9-0.95 or tightening the priors.

---

## Limitations
The method assumes:

- A fixed number of independent release sites
- Binomial release with a single release probability per pulse
- Gamma-distributed quantal size + independent Gaussian baseline noise
- Deterministic recovery / facilitation dynamics (Tsodyks-Markram)
- Stationarity across sweeps

It will perform poorly (or give misleading posteriors) if:

- The synapse has strong multivesicular release or strong postsynaptic receptor saturation
- There is significant rundown or non-stationarity across sweeps
- The number of sweeps is very small and release probability is low
- The true number of sites is far outside the chosen n-max

The original paper and the supplementary Julia codes also implement two additional models (release-independent depression and frequency-dependent recovery). These are not ported here.

---

## Citation

If you use this code, please cite both the original method paper and this implementation:

```
Bird AD, Wall MJ, Richardson MJE (2016)
Bayesian Inference of Synaptic Quantal Parameters from Correlated Vesicle Release.
Front. Comput. Neurosci. 10:116. doi:10.3389/fncom.2016.00116

py-BQE (2026)
Bayesian Quantal Estimation - JAX/NumPyro port.
https://github.com/AMikroulis/py-BQE
```

## License
This code is released under the GNU General Public License v3 (same as the original Julia implementations by Magnus Richardson).

## Disclaimer
This software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.

---

© Apostolos Mikroulis, 2026