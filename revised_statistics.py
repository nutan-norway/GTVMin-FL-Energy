# Revised code for GTVMIN-FL Statistics
# Author: Nutan Gupta, University of Inland Norway
# Date: 10 November 2026

"""
Forecast-comparison statistics.

Tests that account for (i) serial dependence of annual forecast errors
and (ii) contemporaneous dependence across countries (common shocks such as
2009 and 2020), and that distinguish "no significant difference" from
"equivalence":

  * per-country Diebold-Mariano test, squared-error loss, Bartlett HAC
    variance and Harvey-Leybourne-Newbold small-sample correction, with
    Holm adjustment across countries;
  * pooled test of the mean RELATIVE loss differential
        d_ct = (SE_A,ct - SE_B,ct) / MSE_B,c
    (scale-free, so Poland does not dominate), with standard errors
    clustered by forecast year;
  * TOST equivalence test on the same quantity with margin +/- delta
    (default 5% of the benchmark MSE);
  * moving-block bootstrap (blocks of years, all countries jointly) CI for
    the geometric-mean relative MSE across countries.

Reads results_revised/predictions.csv and writes results_revised/stats_*.csv.
"""

import itertools

import numpy as np
import pandas as pd
from scipy import stats

IN = 'results_revised/predictions.csv'
OUT = 'results_revised'
HAC_LAGS = 1
DELTA = 0.05
N_BOOT = 2000
BLOCK = 3
SEED = 20261102

PAIRS = [('FL_similarity', 'Local'), ('FL_similarity', 'Persistence'),
         ('Local', 'Persistence'), ('Hierarchical', 'Local'),
         ('Pooled_LocalScale', 'Local'), ('Pooled_GlobalScale', 'Local'),
         ('FLorig_FixedGraph', 'Localorig'),
         ('FLorig_OriginGraph', 'FLorig_FixedGraph')]
GRAPH_METHODS = ['FL_similarity', 'FL_similarity_fullhistory',
                 'FL_complete_similarity', 'FL_complete_equal',
                 'FL_geographic', 'FL_param_knn']


def dm_test(e_a, e_b, h=1, hac_lags=HAC_LAGS):
    """Diebold-Mariano (1995) with HLN (1997) correction; squared loss.
    Negative statistic => model A more accurate."""
    d = e_a ** 2 - e_b ** 2
    n = len(d)
    dbar = d.mean()
    dc = d - dbar
    gamma = [np.dot(dc[k:], dc[:n - k]) / n for k in range(hac_lags + 1)]
    var = gamma[0] + 2 * sum((1 - k / (hac_lags + 1)) * gamma[k]
                             for k in range(1, hac_lags + 1))
    if var <= 0:
        var = gamma[0]
    if var == 0:
        return 0.0, 1.0
    dm = dbar / np.sqrt(var / n)
    dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return dm, 2 * stats.t.sf(abs(dm), n - 1)


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def paired_frame(g, a, b):
    pa = g[g['Method'] == a].set_index(['Country', 'Year'])
    pb = g[g['Method'] == b].set_index(['Country', 'Year'])
    j = pa[['Actual', 'Predicted']].join(pb[['Predicted']], rsuffix='_b', how='inner')
    j['e_a'] = j['Actual'] - j['Predicted']
    j['e_b'] = j['Actual'] - j['Predicted_b']
    return j.reset_index()


def cluster_mean_test(j, delta=DELTA):
    """Mean relative loss differential, SE clustered by year, plus TOST."""
    mse_b = j.groupby('Country')['e_b'].apply(lambda e: np.mean(e ** 2))
    d = (j['e_a'] ** 2 - j['e_b'] ** 2) / j['Country'].map(mse_b)
    dbar = d.mean()
    resid_sum = (d - dbar).groupby(j['Year']).sum()
    G, N = len(resid_sum), len(d)
    se = np.sqrt(G / (G - 1) * np.sum(resid_sum ** 2)) / N
    df = G - 1
    if se == 0:
        return dict(MeanRelDiff=dbar, SE=0.0, p_diff=1.0, CI90_low=dbar,
                    CI90_high=dbar, p_TOST=0.0 if abs(dbar) < delta else 1.0)
    tcrit = stats.t.ppf(0.95, df)
    p_lower = stats.t.sf((dbar + delta) / se, df)    # H0: mu <= -delta
    p_upper = stats.t.cdf((dbar - delta) / se, df)   # H0: mu >= +delta
    return dict(MeanRelDiff=dbar, SE=se,
                p_diff=2 * stats.t.sf(abs(dbar / se), df),
                CI90_low=dbar - tcrit * se, CI90_high=dbar + tcrit * se,
                p_TOST=max(p_lower, p_upper))


def block_bootstrap_gm(j, rng, n_boot=N_BOOT, block=BLOCK):
    """GM over countries of MSE_A/MSE_B, with a moving-block bootstrap over
    years (all countries resampled jointly to keep common shocks)."""
    years = np.sort(j['Year'].unique())
    se_a = j.pivot(index='Year', columns='Country', values='e_a').loc[years] ** 2
    se_b = j.pivot(index='Year', columns='Country', values='e_b').loc[years] ** 2

    def gm(idx):
        r = se_a.values[idx].mean(0) / se_b.values[idx].mean(0)
        return np.exp(np.mean(np.log(r)))

    T = len(years)
    starts = np.arange(T - block + 1)
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([np.arange(s, s + block)
                              for s in rng.choice(starts, int(np.ceil(T / block)))])[:T]
        boots.append(gm(idx))
    return gm(np.arange(T)), *np.percentile(boots, [2.5, 97.5])


def main():
    df = pd.read_csv(IN)
    rng = np.random.default_rng(SEED)
    per_country, pooled = [], []
    for (cset, spec), g in df.groupby(['CountrySet', 'Spec']):
        methods = set(g['Method'])
        pairs = [p for p in PAIRS if p[0] in methods and p[1] in methods]
        pairs += [(a, b) for a, b in itertools.combinations(GRAPH_METHODS, 2)
                  if a in methods and b in methods]
        for a, b in pairs:
            j = paired_frame(g, a, b)
            rows = []
            for c, jc in j.sort_values('Year').groupby('Country'):
                dm, p = dm_test(jc['e_a'].values, jc['e_b'].values)
                rows.append({'CountrySet': cset, 'Spec': spec, 'A': a, 'B': b,
                             'Country': c, 'N': len(jc),
                             'MSE_A': np.mean(jc['e_a'] ** 2),
                             'MSE_B': np.mean(jc['e_b'] ** 2),
                             'DM': dm, 'p_DM': p})
            p_holm = holm([r['p_DM'] for r in rows])
            for r, ph in zip(rows, p_holm):
                r['p_DM_Holm'] = ph
            per_country += rows
            gm, lo, hi = block_bootstrap_gm(j, rng)
            pooled.append({'CountrySet': cset, 'Spec': spec, 'A': a, 'B': b,
                           'N': len(j), 'Countries': j['Country'].nunique(),
                           'N_A_better_DM05': sum(r['DM'] < 0 and r['p_DM'] < 0.05 for r in rows),
                           'N_B_better_DM05': sum(r['DM'] > 0 and r['p_DM'] < 0.05 for r in rows),
                           'N_sig_Holm': sum(r['p_DM_Holm'] < 0.05 for r in rows),
                           **cluster_mean_test(j),
                           'GM_RelMSE': gm, 'GM_CI95_low': lo, 'GM_CI95_high': hi})
    pc = pd.DataFrame(per_country)
    po = pd.DataFrame(pooled)
    pc.to_csv(f'{OUT}/stats_dm_by_country.csv', index=False)
    po.to_csv(f'{OUT}/stats_pooled.csv', index=False)
    pd.set_option('display.width', 250)
    print(po.to_string(index=False, float_format=lambda v: f'{v:.4f}'))


if __name__ == '__main__':
    main()
