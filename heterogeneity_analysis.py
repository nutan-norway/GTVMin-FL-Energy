# Revised code for GTVMIN-FL Heterogeneity Analysis
# Author: Nutan Gupta, University of Inland Norway
# Date: 10 November 2026

"""
Quantitative characterization of national energy-system heterogeneity

  1. Country profiles from OWID: shares of hydro, wind, solar, nuclear,
     coal, gas, oil and other renewables in primary energy, energy per
     capita, energy intensity of GDP, and growth volatility (1995-2022
     means; descriptive only, never used by any forecasting model).
  2. Formal slope-homogeneity F-test: country-specific slopes vs common
     slopes with country intercepts, for both specifications.
  3. Spearman correlations of the profiles with per-country forecast
     skill, the cost of pooling (MSE_pooled / MSE_local), similarity-graph
     degree and the distance of each country's local parameters from the
     cross-country mean.

Run after run_revised_experiments.py.
"""

import numpy as np
import pandas as pd
from scipy import stats

from gtvmin_fl_revised import (COUNTRY_SETS, FEATURES, EnergyPanel,
                               LocalTransform)
from revised_statistics import holm

DATA_PATH = 'owid-energy-data.csv'
OUT = 'results_revised'
SHARES = ['hydro_share_energy', 'wind_share_energy', 'solar_share_energy',
          'nuclear_share_energy', 'coal_share_energy', 'gas_share_energy',
          'oil_share_energy', 'other_renewables_share_energy']
INTENSITY = ['energy_per_capita', 'energy_per_gdp']


def profiles(panel):
    df = pd.read_csv(DATA_PATH, usecols=['country', 'year'] + SHARES + INTENSITY)
    df = df[df['country'].isin(panel.countries) & df['year'].between(1995, 2022)]
    prof = df.groupby('country')[SHARES + INTENSITY].mean()
    for c in panel.countries:
        g = panel.designs['growth'][c]
        prof.loc[c, 'energy_growth_sd'] = g['y'].std()
        prof.loc[c, 'gdp_growth_mean'] = g['x_gdp'].mean()
        prof.loc[c, 'log_energy_level'] = np.log(g['E'].mean())
    return prof


def slope_homogeneity_test(panel, spec):
    """F-test of H0: all countries share the slope vector (country
    intercepts allowed) vs country-specific slopes, OLS on the full window,
    data in each country's model units."""
    X, y, ids = [], [], []
    for k, c in enumerate(panel.countries):
        d = panel.designs[spec][c]
        tf = LocalTransform(spec).fit(d[FEATURES].values, d['y'].values)
        X.append(tf.tx(d[FEATURES].values)); y.append(tf.ty(d['y'].values))
        ids += [k] * len(d)
    X, y, ids = np.vstack(X), np.concatenate(y), np.array(ids)
    m, p, N = len(panel.countries), X.shape[1], len(y)
    D = np.eye(m)[ids]
    Zr = np.hstack([X, D])                                   # common slopes
    Zu = np.hstack([D] + [X[:, [j]] * D for j in range(p)])  # country slopes
    rss = [np.sum((y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]) ** 2) for Z in (Zr, Zu)]
    q, dfu = Zu.shape[1] - Zr.shape[1], N - Zu.shape[1]
    F = ((rss[0] - rss[1]) / q) / (rss[1] / dfu)
    return {'Spec': spec, 'Countries': m, 'N': N, 'RSS_common': rss[0],
            'RSS_country': rss[1], 'df1': q, 'df2': dfu, 'F': F,
            'p': stats.f.sf(F, q, dfu)}


def main():
    summary = pd.read_csv(f'{OUT}/summary_by_country.csv')
    diag = pd.read_csv(f'{OUT}/fl_diagnostics.csv')
    prof_rows, tests, cors = [], [], []
    for cset in summary['CountrySet'].unique():
        panel = EnergyPanel(DATA_PATH, COUNTRY_SETS[cset])
        prof = profiles(panel)
        prof_rows.append(prof.assign(CountrySet=cset).reset_index())
        for spec in summary.query('CountrySet == @cset')['Spec'].unique():
            tests.append({'CountrySet': cset, **slope_homogeneity_test(panel, spec)})
            s = summary.query('CountrySet == @cset and Spec == @spec').set_index(['Method', 'Country'])
            mse = s['MSE'].unstack('Method')
            outcomes = pd.DataFrame({
                'Skill_Local': s.loc['Local', 'Skill_vs_Persistence'],
                'Skill_FL': s.loc['FL_similarity', 'Skill_vs_Persistence'],
                'PoolingCost_Global': mse['Pooled_GlobalScale'] / mse['Local'],
                'FLGain_vs_Local': 1 - mse['FL_similarity'] / mse['Local'],
            })
            if 'Hierarchical' in mse:
                outcomes['PoolingCost_Hier'] = mse['Hierarchical'] / mse['Local']
            dg = diag.query('CountrySet == @cset and Spec == @spec and Method == "FL_similarity"')
            outcomes['MeanDegree'] = dg.groupby('Country')['Degree'].mean()
            outcomes['MedianParamDistRel'] = dg.groupby('Country')['ParamDistRel'].median()
            for o in outcomes:
                family = []
                for ch in prof.columns:
                    jj = pd.concat([outcomes[o], prof[ch]], axis=1).dropna()
                    rho, p = stats.spearmanr(jj.iloc[:, 0], jj.iloc[:, 1])
                    family.append({'CountrySet': cset, 'Spec': spec, 'Outcome': o,
                                   'Characteristic': ch, 'n': len(jj),
                                   'Spearman_rho': rho, 'p': p})
                for f, a in zip(family, holm([f['p'] for f in family])):
                    f['p_Holm'] = a
                cors += family
    pd.concat(prof_rows).to_csv(f'{OUT}/heterogeneity_profiles.csv', index=False)
    pd.DataFrame(tests).to_csv(f'{OUT}/heterogeneity_slope_tests.csv', index=False)
    cor = pd.DataFrame(cors)
    cor.to_csv(f'{OUT}/heterogeneity_correlations.csv', index=False)
    pd.set_option('display.width', 200)
    print(pd.DataFrame(tests).to_string(index=False))
    print("\nCorrelations with p < 0.05 (unadjusted):")
    print(cor[cor['p'] < 0.05].sort_values('p').to_string(index=False,
          float_format=lambda v: f'{v:.3f}'))


if __name__ == '__main__':
    main()
