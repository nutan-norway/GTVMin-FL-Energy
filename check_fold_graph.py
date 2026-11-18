# Revised code for GTVMIN-FL Statistics
# Author: Nutan Gupta, University of Inland Norway
# Date: 18 November 2026
"""
Sensitivity check for the inner validation (Section III-D).

At each forecast origin t0, lambda is selected on inner one-step-ahead folds
using the collaboration graph built at t0, i.e. from mean profiles that also
cover the validation years. This never touches the evaluation year t0 + 1,
but to show that it does not matter, this script rebuilds the similarity
graph inside every fold from that fold's fitting years only and compares the
selected lambda with the one used in the paper.

Writes results_revised/fold_graph_check.csv and prints a summary.
"""

import numpy as np
import pandas as pd

import gtvmin_fl_revised as g

DATA = 'owid-energy-data.csv'
OUT = 'results_revised/fold_graph_check.csv'
N_VAL, MIN_FIT = 5, 6


def select_lambda_fold_graphs(panel, train, spec, betas, nodes):
    """Network-wide lambda by inner validation, with the similarity graph of
    each fold built from that fold's fitting years only."""
    years = train[nodes[0]][2]
    errs = np.zeros(len(g.LAMBDA_GRID))
    for mask in g.inner_splits(years, N_VAL, MIN_FIT):
        last_fit = years[mask[0]].max()
        W = g.weight_matrix(g.build_graph('similarity', nodes,
                                          means=panel.profile_means(last_fit)), nodes)
        A, c, const, val = g.node_systems(train, spec, betas, nodes, mask)
        for k, lam in enumerate(g.LAMBDA_GRID):
            Theta = g.GTVMin(A, c, const, W, lam).exact()
            errs[k] += sum(((g.predict_linear(Theta[i], Xv) - yv) ** 2).sum()
                           for i, (Xv, yv) in enumerate(val))
    return g.LAMBDA_GRID[int(np.argmin(errs))]


def main():
    rows = []
    for cset in ('europe', 'paper18'):
        panel = g.EnergyPanel(DATA, g.COUNTRY_SETS[cset])
        nodes = panel.countries
        for spec in ('levels', 'growth'):
            for t0 in range(panel.first_target_year + 9, panel.last_year):
                train = {}
                for n in nodes:
                    d = panel.designs[spec][n]
                    tr = d[d['year'] <= t0]
                    train[n] = (tr[g.FEATURES].values, tr['y'].values, tr['year'].values)
                b = g.tune_beta_network(train, nodes, spec, g.BETA_GRID, N_VAL, MIN_FIT)
                betas = {n: b for n in nodes}
                W = g.weight_matrix(g.build_graph('similarity', nodes,
                                                  means=panel.profile_means(t0)), nodes)
                lam_origin, _ = g.tune_lambda(train, spec, betas, nodes, W,
                                              g.LAMBDA_GRID, N_VAL, MIN_FIT)
                lam_fold = select_lambda_fold_graphs(panel, train, spec, betas, nodes)
                rows.append({'CountrySet': cset, 'Spec': spec, 'Origin': t0,
                             'Lambda_origin_graph': lam_origin,
                             'Lambda_fold_graphs': lam_fold,
                             'Same': lam_origin == lam_fold})
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    summary = df.groupby(['CountrySet', 'Spec'])['Same'].agg(['sum', 'count'])
    print('Origins with the same selected lambda:')
    print(summary.to_string())
    print(df[~df['Same']].to_string(index=False))


if __name__ == '__main__':
    main()