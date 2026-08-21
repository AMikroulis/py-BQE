import math
import time
from functools import partial
import matplotlib.pyplot as plt
import numpy as npy
import jax
import jax.numpy as jnpy
from jax import lax
import numpyro
import numpyro.distributions as dists
from numpyro.infer import MCMC, NUTS

# set float64 for JAX
jax.config.update('jax_enable_x64', True)

# Gauss-Legendre quadrature nodes/weights. 64-point, fixed at import
_GL_NODES, _GL_WEIGHTS = npy.polynomial.legendre.leggauss(64)
_GL_NODES_J   = jnpy.array(_GL_NODES)
_GL_WEIGHTS_J = jnpy.array(_GL_WEIGHTS)

MODELS = ('dep', 'daf')


def _stirling_ratio_vec(n_j):
    log_r = (0.5 * jnpy.log(2.0 * math.pi * n_j) + n_j * (jnpy.log(n_j) - 1.0) - jax.scipy.special.gammaln(n_j + 1.0))

    return jnpy.exp(log_r)


def _conv_batch(A_t, a, siga, n_int: int, sigb):
    '''
    Convolution likelihoods for all (pulse, sweep, k).
    A_t: (N, S)
    out: (N, S, n+1)
    '''
    N, S = A_t.shape
    n = n_int

    nvar = sigb * sigb
    k0 = jnpy.exp(-A_t * A_t / (2.0 * nvar)) / jnpy.sqrt(2.0 * math.pi * nvar)

    if n == 0:
        return k0[:, :, None]
        
    ks = jnpy.arange(1, n + 1, dtype=jnpy.float64)
    betk = ks * a * a / (siga * siga)
    betk_safe = jnpy.clip(betk, 1.001, None)
    lamk = a / (siga * siga)
    ys = (betk_safe - 1.0) / lamk
    std_bd = 7.0 * jnpy.sqrt(ys / (lamk * lamk) + sigb * sigb)
    yL = jnpy.clip(ys - std_bd, 1e-12, None)
    yR = ys + std_bd
    half = (yR - yL) * 0.5
    mid = (yR + yL) * 0.5

    y = mid[:, None] + half[:, None] * _GL_NODES_J[None, :]     # y[k, j] — (K, 64)

    log_base = ((betk_safe - 1.0)[:, None] * jnpy.log(y / ys[:, None]) - lamk * (y - ys[:, None]))

    #  (K,1,1,64) into (1,N,S,1) -> (K,N,S,64)
    A_4d = A_t[None, :, :, None]
    y_4d = y[:, None, None, :]

    log_integ = (log_base[:, None, None, :] - (A_4d - y_4d) ** 2 / (2.0 * sigb * sigb))

    kappas = _stirling_ratio_vec(betk_safe - 1.0)
    norms = kappas / (2.0 * math.pi * sigb * jnpy.sqrt(ys / lamk))

    integrals = (jnpy.exp(log_integ) * _GL_WEIGHTS_J[None, None, None, :]).sum(axis=3)
    raw = (norms * half)[:, None, None] * integrals   # (K, N, S)

    valid = (betk > 1.0)[:, None, None]
    kgt0 = jnpy.where(valid, raw, 1e-300 * jnpy.ones_like(raw))
    kgt0 = kgt0.transpose(1, 2, 0)          # (N, S, K)

    return jnpy.concatenate([k0[:, :, None], kgt0], axis=2)  # (N, S, n+1)


_conv_batch_j = partial(jax.jit, static_argnums=(3,))(_conv_batch)


def _release_prob_train(p, f, tauf, T_gaps, N: int):
    '''
    Release probability before each pulse.

    u_1 = p
    u = u + u_m + f * (1 - u_m)        ## jump toward 1 at each pulse
    u_[m+1] = p + (u+ - p) * exp(-T_m / tauf)   ## relax back to p, not to 0

    f is the facilitation increment.
    paired-pulse release probability is u1 = p + f * (1 - p)
    f = (u1 - p) / (1 - p)
    '''
    tauf_safe = jnpy.clip(tauf, 1e-6, None)

    def step(u_m, T_m):
        u_plus = u_m + f * (1.0 - u_m)
        u_next = p + (u_plus - p) * jnpy.exp(-T_m / tauf_safe)
        return u_next, u_next
    
    u_first = jnpy.asarray(p, dtype=jnpy.float64)
    _, u_rest = lax.scan(step, u_first, T_gaps)

    return jnpy.concatenate([u_first[None], u_rest])


def _P_k_given_x(n_int: int, p):
    # P(k|x) — shape (n+1, n+1), index [k, x]
    n = n_int

    k2 = jnpy.arange(n + 1, dtype=jnpy.float64)[:, None]
    x2 = jnpy.arange(n + 1, dtype=jnpy.float64)[None, :]

    lp = jnpy.log(jnpy.clip(p, 1e-12, 1.0 - 1e-12))
    lq = jnpy.log(jnpy.clip(1.0 - p, 1e-12, 1.0 - 1e-12))

    log_binom = (jax.scipy.special.gammaln(x2 + 1.) - jax.scipy.special.gammaln(jnpy.clip(k2, 0, None) + 1.) - jax.scipy.special.gammaln(jnpy.clip(x2 - k2, 0, None) + 1.))

    valid = (k2 <= x2).astype(jnpy.float64)

    return valid * jnpy.exp(log_binom + k2 * lp + (x2 - k2) * lq)


def _joint_transition(n_int: int, g, Pkx):
    # T_all[k, xnext, x] = P(xnext | k, x) * P(k|x)
    n = n_int
    ks = jnpy.arange(n + 1, dtype=jnpy.float64).reshape(n+1, 1, 1)
    xns = jnpy.arange(n + 1, dtype=jnpy.float64).reshape(1, n+1, 1)
    xs = jnpy.arange(n + 1, dtype=jnpy.float64).reshape(1, 1, n+1)

    rec   = xns - xs + ks
    total = float(n) - xs + ks

    lg = jnpy.log(jnpy.clip(g, 1e-12, 1.0 - 1e-12))
    l1g = jnpy.log(jnpy.clip(1.0 - g, 1e-12, 1.0 - 1e-12))

    s_total = jnpy.clip(total, 0, None)
    s_rec = jnpy.clip(rec, 0, None)
    s_tmr = jnpy.clip(total - rec, 0, None)
    log_binom = (jax.scipy.special.gammaln(s_total + 1.) - jax.scipy.special.gammaln(s_rec + 1.) - jax.scipy.special.gammaln(s_tmr + 1.))

    valid = ((rec >= 0) & (rec <= total) & (xns >= xs - ks)).astype(jnpy.float64)
    P_xnext_k_x = valid * jnpy.exp(log_binom + rec * lg + (total - rec) * l1g)

    return P_xnext_k_x * Pkx[:, None, :]   # (k, xnext, x)


@partial(jax.jit, static_argnums=(6, 7))
def _log_likelihood_fixed_n(p, tD, f, tauf, T_gaps, conv_liks, n_int: int, model: str):
    '''
    sum log P(sweep) over all sweeps for fixed n

    conv_liks takes (N, S, n+1)

    f, tauf ignored if model == 'dep'
    '''

    N, S = conv_liks.shape[0], conv_liks.shape[1]
    n = n_int

    if model == 'daf':
        u = _release_prob_train(p, f, tauf, T_gaps, N)
        Pkx_all = jax.vmap(partial(_P_k_given_x, n))(u)
        Pkx_last = Pkx_all[N - 1]

    else:
        Pkx_all = None
        Pkx_last = _P_k_given_x(n, p)

    QR_init = jnpy.zeros((n + 1, S), dtype=jnpy.float64).at[n, :].set(1.0)
    log_scale_init = jnpy.zeros(S, dtype=jnpy.float64)

    def step(carry, m):
        QR, log_scale = carry
        g_t = 1.0 - jnpy.exp(-T_gaps[m] / tD)
        Pkx_m = Pkx_all[m] if model == 'daf' else Pkx_last
        T_all = _joint_transition(n, g_t, Pkx_m)
        Gm = conv_liks[m]

        # Qm[s, xnext, x] = sum_k Gm[s,k] * T_all[k, xnext, x]
        Qm = jnpy.einsum('sk,kno->sno', Gm, T_all)

        # QR_new[xnext, s] = sum_x Qm[s, xnext, x] * QR[x, s]
        QR_new = jnpy.einsum('sno,os->ns', Qm, QR)

        scales = QR_new.sum(axis=0).clip(1e-300)
        QR_new = QR_new / scales[None, :]
        log_scale = log_scale + jnpy.log(scales)
        return (QR_new, log_scale), jnpy.zeros(())
    
    (QR, log_scale), _ = lax.scan(step, (QR_init, log_scale_init), jnpy.arange(N - 1, dtype=jnpy.int32))

    G_last = conv_liks[N - 1]
    Lm_last = Pkx_last.T @ G_last.T
    p_total = (Lm_last * QR).sum(axis=0)

    return jnpy.sum(jnpy.log(p_total.clip(1e-300)) + log_scale)


def _log_liks_all_n(p, tD, a, siga, sigb, f, tauf, T_gaps, A_t, n_max: int, model: str):
    # LL at every n in 1...n_max

    conv_full = _conv_batch_j(A_t, a, siga, n_max, sigb)     # (N, S, n_max+1)

    return jnpy.stack([_log_likelihood_fixed_n(p, tD, f, tauf, T_gaps, conv_full[:, :, :n + 1], n, model) for n in range(1, n_max + 1)])


def _log_lik_marginal_n(p, tD, a, siga, sigb, f, tauf, T_gaps, A_t, n_max: int, model: str):
    # marginalise over n in 1...n_max

    log_prior_n = -math.log(float(n_max))
    lls = _log_liks_all_n(p, tD, a, siga, sigb, f, tauf, T_gaps, A_t, n_max, model)
    return jax.scipy.special.logsumexp(lls + log_prior_n)


# default bounds, adjust as needed
UNIT_PRIORS = {
    'mV': dict(a=(0.05, 0.50), siga=(0.02, 0.20), sigb=(0.01, 0.10)),
    'pA': dict(a=(0.1, 500.0), siga=(0.05, 100.0), sigb=(0.05, 50.0)),
}

# facilitation decay constant (ms) -> tauf_range
TAUF_RANGE = (10.0, 500.0)


def _make_synaptic_model(priors, model, n_max):

    def _synaptic_model(T_gaps, A_t):
        p = numpyro.sample('p', dists.Uniform(0.05, 0.95))
        tD = numpyro.sample('tD', dists.Uniform(50.0, 500.0))
        a = numpyro.sample('a', dists.Uniform(*priors['a']))
        siga = numpyro.sample('siga', dists.Uniform(*priors['siga']))
        sigb = numpyro.sample('sigb', dists.Uniform(*priors['sigb']))

        if model == 'daf':
            f = numpyro.sample('f', dists.Uniform(0.0, 1.0))
            tauf = numpyro.sample('tauf', dists.Uniform(*priors['tauf']))
            numpyro.deterministic('u1', p + f * (1.0 - p))

        else:
            f, tauf = 0.0, 1.0

        numpyro.factor('a_gt_siga', -1e6 * jnpy.maximum(siga - a, 0.0))
        numpyro.factor('obs', _log_lik_marginal_n(p, tD, a, siga, sigb, f, tauf, T_gaps, A_t, n_max, model))

    return _synaptic_model
    

def _param_names(model):
    # parameter names in plot order
    names = ['p', 'tD', 'a', 'siga', 'sigb']

    if model == 'daf':
        names += ['f', 'tauf', 'u1']

    return names


def _marginal_pairs(model):
    pairs = [('p', 'tD'), ('p', 'a'), ('a', 'siga'), ('tD', 'a'), ('siga', 'sigb')]

    if model == 'daf':
        pairs += [('p', 'f'), ('f', 'tauf'), ('p', 'u1')]

    return pairs


@partial(jax.jit, static_argnums=(7, 8))
def _ll_all_n_batch(p, tD, a, siga, sigb, f, tauf, n_max, model, T_gaps, A_t):
    # LL at every n

    def _ll_single(p_i, tD_i, a_i, siga_i, sigb_i, f_i, tauf_i):
        return _log_liks_all_n(p_i, tD_i, a_i, siga_i, sigb_i, f_i, tauf_i, T_gaps, A_t, n_max, model)

    return jax.vmap(_ll_single, in_axes=(0, 0, 0, 0, 0, 0, 0))(p, tD, a, siga, sigb, f, tauf)


def _posterior_n(samples_flat, T_gaps, A_t, n_max, model, chunk=200):
    # P(n | data)

    log_prior_n = -math.log(float(n_max))

    n_samp = samples_flat['p'].shape[0]
    p_arr = jnpy.array(samples_flat['p'])
    tD_arr = jnpy.array(samples_flat['tD'])
    a_arr = jnpy.array(samples_flat['a'])
    siga_arr = jnpy.array(samples_flat['siga'])
    sigb_arr = jnpy.array(samples_flat['sigb'])

    if model == 'daf':
        f_arr = jnpy.array(samples_flat['f'])
        tauf_arr = jnpy.array(samples_flat['tauf'])
    else:
        f_arr = jnpy.zeros(n_samp)
        tauf_arr = jnpy.ones(n_samp)

    log_w = npy.zeros((n_samp, n_max))

    for lo in range(0, n_samp, chunk):
        hi = min(lo + chunk, n_samp)

        log_w[lo:hi, :] = npy.array(_ll_all_n_batch(p_arr[lo:hi], tD_arr[lo:hi], a_arr[lo:hi], siga_arr[lo:hi], sigb_arr[lo:hi], f_arr[lo:hi], tauf_arr[lo:hi], n_max, model, T_gaps, A_t)) + log_prior_n

    log_Z   = npy.array(jax.scipy.special.logsumexp(jnpy.array(log_w), axis=1, keepdims=True))
    weights = npy.exp(log_w - log_Z)

    return weights.mean(axis=0)


def simulate(filename, n=5, p=0.2, f=0.5, tauf=100.0, tD=300.0, a=0.2, siga=0.05, sigb=0.02, n_pulses=8, isi=50.0, n_sweeps=30, ts=None, seed=0):
    '''
    facilitating train from the forward model as (col 0 = times [in ms], cols 1,... = one column per sweep)

    outputs (ts, A) with A of shape (n_pulses, n_sweeps)

    f = 0 gives depression-only
    ts for custom ISIs
    '''

    rng = npy.random.default_rng(seed)

    if ts is None:
        ts = npy.arange(n_pulses, dtype=float) * isi
    else:
        ts = npy.asarray(ts, dtype=float)
        n_pulses = len(ts)

    T_gaps = npy.diff(ts)

    # release probability before each pulse
    u = npy.empty(n_pulses)
    u[0] = p

    for m in range(n_pulses - 1):
        u_plus = u[m] + f * (1.0 - u[m])
        u[m + 1] = p + (u_plus - p) * npy.exp(-T_gaps[m] / tauf)

    shape_1q = a * a / (siga * siga)    # gamma shape of a single quantum
    rate = a / (siga * siga)

    A = npy.empty((n_pulses, n_sweeps))
    x = npy.full(n_sweeps, n, dtype=int)    # all sites available at pulse 1

    for m in range(n_pulses):
        k = rng.binomial(x, u[m])
        quanta = rng.gamma(npy.maximum(k, 1) * shape_1q, 1.0 / rate)

        A[m] = npy.where(k > 0, quanta, 0.0) + rng.normal(0.0, sigb, n_sweeps)

        if m < n_pulses - 1:
            g = 1.0 - npy.exp(-T_gaps[m] / tD)
            x = x - k + rng.binomial(n - x + k, g)
        
    npy.savetxt(filename, npy.column_stack([ts, A]), fmt='%.4f')

    print(f'Wrote {filename}: {n_pulses} pulses, {n_sweeps} sweeps, seed {seed}')
    print('Ground truth:')
    print(f'  n={n}  p={p}  f={f}  tauf={tauf}  tD={tD}')
    print(f'  a={a}  siga={siga}  sigb={sigb}   u1 = p+f(1-p) = {p + f*(1-p):.4f}')
    print('  release prob u:', npy.round(u, 4))
    print('  mean amplitude:', npy.round(A.mean(axis=1), 4))
    return ts, A

    

TARGET_ACCEPT = {'dep': 0.8, 'daf': 0.8} # NUTS


def main(filename, num_warmup=500, num_samples=1000, num_chains=4, units='mV', a_range=None, siga_range=None, sigb_range=None, model='dep', n_max=10, tauf_range=None, target_accept=None):
    t0 = time.time()

    if units not in UNIT_PRIORS:
        raise ValueError(f'units must be one of {list(UNIT_PRIORS)}; got {units!r}')

    if model not in MODELS:
        raise ValueError(f'model must be one of {list(MODELS)}; got {model!r}')

    priors = dict(UNIT_PRIORS[units])
    priors['tauf'] = TAUF_RANGE

    if a_range is not None:
        priors['a'] = a_range

    if siga_range is not None:
        priors['siga'] = siga_range

    if sigb_range is not None:
        priors['sigb'] = sigb_range

    if tauf_range is not None:
        priors['tauf'] = tauf_range

    if target_accept is None:
        target_accept = TARGET_ACCEPT[model]

    M = npy.loadtxt(filename)

    ts_np = M[:, 0]
    A_np = M[:, 1:]
    N_pulses, N_sweeps = A_np.shape

    print(f'Loaded {filename}: {N_pulses} pulses, {N_sweeps} sweeps.')
    print(f'Model: {model}  |  n marginalised over 1..{n_max}')
    print(f'Units: {units}  |  priors: a={priors['a']}, siga={priors['siga']}, sigb={priors['sigb']}')

    if model == 'daf':
        print(f'Facilitation: f=(0, 1), tauf={priors['tauf']}')
    
    # plot raw data
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))

    axes[0].plot(ts_np, A_np, '-o', lw=0.8, ms=3)
    axes[0].set_xlabel('Time [ms]')
    axes[0].set_ylabel(f'Amplitude [{units}]')
    axes[0].set_title('Raw data')

    counts, edges = npy.histogram(A_np.ravel(), bins=30)

    axes[1].plot((edges[:-1] + edges[1:]) / 2, counts, '-')
    axes[1].set_xlabel(f'Amplitude [{units}]')
    axes[1].set_ylabel('Count')

    fig.tight_layout()

    T_gaps = jnpy.array(npy.diff(ts_np))
    A_t = jnpy.array(A_np)

    # precompile before MCMC
    # midpoints of selected priors as dummy values

    a_mid = (priors['a'][0] + priors['a'][1]) / 2
    siga_mid = (priors['siga'][0] + priors['siga'][1]) / 2
    sigb_mid = (priors['sigb'][0] + priors['sigb'][1]) / 2
    tauf_mid = (priors['tauf'][0] + priors['tauf'][1]) / 2

    print('JAX compilation ...')
    t_compile = time.time()

    ll0 = _log_lik_marginal_n(jnpy.float64(0.3), jnpy.float64(200.0), jnpy.float64(a_mid), jnpy.float64(siga_mid), jnpy.float64(sigb_mid), jnpy.float64(0.5), jnpy.float64(tauf_mid), T_gaps, A_t, n_max, model)
    print(f'  Compiled in {time.time() - t_compile:.1f} s '
        f'(LL at prior midpoints: {float(ll0):.2f})')
    
    # MCMC
    print(f'\nRunning NUTS ({num_chains} chains x {num_samples} samples + '
          f'{num_warmup} warmup, target_accept={target_accept}) ...')
    
    nuts = NUTS(_make_synaptic_model(priors, model, n_max), dense_mass=True, target_accept_prob=target_accept)

    mcmc = MCMC(nuts, num_samples=num_samples, num_warmup=num_warmup, num_chains=num_chains, chain_method='vectorized', progress_bar=True)

    mcmc.run(jax.random.PRNGKey(0), T_gaps, A_t)

    print('\n~~~~ Posterior summary (NUTS) ~~~~')

    mcmc.print_summary(exclude_deterministic=False)

    samples_raw = mcmc.get_samples(group_by_chain=False)

    samples = {k: npy.array(v).ravel() for k, v in samples_raw.items()}

    # posterior-mean release probability for the full the train
    if model == 'daf':
        def _u_train(p_i, f_i, tauf_i):
            return _release_prob_train(p_i, f_i, tauf_i, T_gaps, N_pulses)

        u_mean = npy.array(jax.vmap(_u_train)(jnpy.array(samples['p']), jnpy.array(samples['f']), jnpy.array(samples['tauf']))).mean(axis=0)

        print('\nPosterior-mean release probability u per pulse:', npy.round(u_mean, 4))

    # marginal posterior over n
    print('\n marginal posterior over n ...')
    p_n = _posterior_n(samples, T_gaps, A_t, n_max, model)
    ns = npy.arange(1, n_max + 1)
    print('P(n | data):', {int(ns[i]): round(float(p_n[i]), 3) for i in range(n_max)})

    elapsed = time.time() - t0
    print(f'\nTotal runtime: {elapsed:.1f} s ({elapsed/60:.1f} min)')

    # plot posteriors
    names = _param_names(model)
    ncols = 3
    nrows = math.ceil((len(names) + 1) / ncols)
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes2 = npy.atleast_1d(axes2).ravel()

    for j, name in enumerate(names):
        ax = axes2[j]
        ax.hist(samples[name], bins=40, density=True)
        ax.set_xlabel(name)
        ax.set_ylabel('density')
        ax.set_title(name)

    ax_n = axes2[len(names)]
    ax_n.bar(ns, p_n)
    ax_n.set_xlabel('n')
    ax_n.set_ylabel('P(n | data)')
    ax_n.set_title('n (marginalised)')

    for ax in axes2[len(names) + 1:]:
        ax.axis('off')

    fig2.suptitle(f'Marginal posteriors (NUTS, model={model})')
    fig2.tight_layout()

    # 2d marginal plots
    pairs = _marginal_pairs(model)
    pcols = min(5, len(pairs))
    prows = math.ceil(len(pairs) / pcols)
    fig3, axes3 = plt.subplots(prows, pcols, figsize=(4 * pcols, 4 * prows))
    axes3 = npy.atleast_1d(axes3).ravel()

    for ax, (na, nb) in zip(axes3, pairs):
        ax.scatter(samples[nb], samples[na], s=2, alpha=0.33)
        ax.set_xlabel(nb)
        ax.set_ylabel(na)

    for ax in axes3[len(pairs):]:
        ax.axis('off')
    
    fig3.suptitle('2-D marginal posteriors')
    fig3.tight_layout()

    plt.show()
    return samples, p_n
    