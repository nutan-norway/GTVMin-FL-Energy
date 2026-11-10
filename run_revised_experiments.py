# Revised code for GTVMIN-FL Experiments
# Author: Nutan Gupta, University of Inland Norway
# Date: 10 November 2026

"""
Runs every experiment and writes CSVs/figures to
results_revised/. Usage:

    python run_revised_experiments.py
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gtvmin_fl_revised import (COUNTRY_SETS, FEATURES, GTVMin, EnergyPanel,
                               RollingOriginEvaluator, aggregate, build_graph,
                               mase_scales, node_systems, summarize,
                               tune_beta_local, tune_lambda, weight_matrix)

DATA_PATH = 'owid-energy-data.csv'
OUT = 'results_revised'
GRAPH_KINDS = ['similarity', 'similarity_fullhistory', 'complete_similarity',
               'complete_equal', 'geographic', 'param_knn']
LAMBDA_PATH = [0.0, 1e-3, 1e-2, 1e-1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 1000.0, 1e4]


def convergence_trace(panel, spec, origin, lambdas, n_val=3, min_fit=6):
    """Objective value and distance to the exact GTVMin solution per
    federated round, at one origin, for the similarity graph (R4.1)."""
    nodes = panel.countries
    train = {}
    for n in nodes:
        d = panel.designs[spec][n]
        d = d[d['year'] <= origin]
        train[n] = (d[FEATURES].values, d['y'].values, d['year'].values)
    from gtvmin_fl_revised import BETA_GRID, LAMBDA_GRID
    betas = {n: tune_beta_local(*train[n], spec, BETA_GRID, n_val, min_fit)
             for n in nodes}
    G = build_graph('similarity', nodes, means=panel.profile_means(origin))
    W = weight_matrix(G, nodes)
    tuned, _ = tune_lambda(train, spec, betas, nodes, W, LAMBDA_GRID, n_val, min_fit)
    A, c, const, _ = node_systems(train, spec, betas, nodes)
    rows = []
    for lam in sorted(set(lambdas) | {tuned}):
        if lam == 0:
            continue
        solver = GTVMin(A, c, const, W, lam)
        star = solver.exact()
        f_star = solver.objective(star)
        _, info = solver.federated(tol=1e-12, max_iter=5000, trace=True)
        for k, (f, th) in enumerate(zip(info['objective'], info['thetas'])):
            rows.append({'Spec': spec, 'Origin': origin, 'Lambda': lam,
                         'Tuned': lam == tuned, 'Round': k,
                         'Objective': f, 'ObjectiveGap': f - f_star,
                         'DistToExact': np.linalg.norm(th - star)})
    return pd.DataFrame(rows)


def plot_lambda_path(path_summary, fname):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for (cset, spec), g in path_summary.groupby(['CountrySet', 'Spec']):
        g = g.sort_values('Lambda')
        x = g['Lambda'].replace(0, g['Lambda'][g['Lambda'] > 0].min() / 10)
        axes[0].semilogx(x, g['GM_RelMSE'], marker='o', label=f'{cset}/{spec}')
        axes[1].loglog(x, g['Median_ParamDistRel'], marker='o', label=f'{cset}/{spec}')
    axes[0].axhline(1.0, color='grey', ls='--', lw=1)
    axes[0].set_xlabel('lambda (0 plotted at left edge)')
    axes[0].set_ylabel('GM of MSE / MSE_persistence')
    axes[1].set_xlabel('lambda')
    axes[1].set_ylabel('median ||theta_FL - theta_local|| / ||theta_local||')
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    plt.close(fig)


def plot_convergence(trace, fname):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for (spec, lam), g in trace.groupby(['Spec', 'Lambda']):
        lbl = f'{spec}, lambda={lam:g}' + (' (tuned)' if g['Tuned'].iloc[0] else '')
        axes[0].semilogy(g['Round'], g['ObjectiveGap'].clip(lower=1e-16), label=lbl)
        axes[1].semilogy(g['Round'], g['DistToExact'].clip(lower=1e-16), label=lbl)
    axes[0].set_ylabel('f(Theta_k) - f(Theta*)')
    axes[1].set_ylabel('||Theta_k - Theta*||')
    for ax in axes:
        ax.set_xlabel('communication round')
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sets', nargs='+', default=['paper18', 'europe'])
    ap.add_argument('--specs', nargs='+', default=['growth', 'levels'])
    ap.add_argument('--quick', action='store_true',
                    help='fewer graphs, no lambda path, no hierarchical model')
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    preds, paths, diags, gstats, summaries, traces = [], [], [], [], [], []
    path_summaries = []
    for cset in args.sets:
        panel = EnergyPanel(DATA_PATH, COUNTRY_SETS[cset])
        print(f"\n=== Country set '{cset}': {len(panel.countries)} countries ===")
        for c, why in panel.excluded.items():
            print(f"  excluded {c}: {why}")
        for spec in args.specs:
            ev = RollingOriginEvaluator(
                panel, spec,
                graph_kinds=['similarity'] if args.quick else GRAPH_KINDS,
                lambda_path=None if args.quick else LAMBDA_PATH,
                include_original=True,
                include_hierarchical=not args.quick,
                rolling_window=None if args.quick else 10)
            res = ev.run()
            scales = mase_scales(panel, ev.origins()[0])
            for key, store in [('predictions', preds), ('lambda_path', paths),
                               ('fl_diagnostics', diags), ('graph_stats', gstats)]:
                if len(res[key]):
                    store.append(res[key].assign(CountrySet=cset))
            summaries.append(summarize(res['predictions'], scales).assign(CountrySet=cset))
            if len(res['lambda_path']):
                ps = summarize(res['lambda_path'], scales,
                               group_cols=('Spec', 'Method', 'Lambda'))
                dist = (res['lambda_path'].groupby(['Spec', 'Method', 'Lambda'])
                        ['ParamDistRel'].median().rename('Median_ParamDistRel'))
                agg = (aggregate(ps, group_cols=('Spec', 'Method', 'Lambda'))
                       .merge(dist.reset_index(), on=['Spec', 'Method', 'Lambda'])
                       .assign(CountrySet=cset))
                path_summaries.append(agg)
            if not args.quick:
                traces.append(convergence_trace(panel, spec, ev.origins()[-1],
                                                [1.0, 10.0, 100.0]).assign(CountrySet=cset))

    summary = pd.concat(summaries, ignore_index=True)
    overall = aggregate(summary, group_cols=('CountrySet', 'Spec', 'Method'))
    pd.concat(preds).to_csv(f'{OUT}/predictions.csv', index=False)
    summary.to_csv(f'{OUT}/summary_by_country.csv', index=False)
    overall.to_csv(f'{OUT}/summary_overall.csv', index=False)
    if diags:
        pd.concat(diags).to_csv(f'{OUT}/fl_diagnostics.csv', index=False)
    if gstats:
        pd.concat(gstats).to_csv(f'{OUT}/graph_stats.csv', index=False)
    if paths:
        pd.concat(paths).to_csv(f'{OUT}/lambda_path_predictions.csv', index=False)
        ps = pd.concat(path_summaries, ignore_index=True)
        ps.to_csv(f'{OUT}/lambda_path_summary.csv', index=False)
        plot_lambda_path(ps[ps['Method'] == 'FLpath_similarity'],
                         f'{OUT}/fig_lambda_path.png')
    if traces:
        tr = pd.concat(traces)
        tr.to_csv(f'{OUT}/convergence_trace.csv', index=False)
        plot_convergence(tr[tr['CountrySet'] == args.sets[0]],
                         f'{OUT}/fig_convergence.png')

    pd.set_option('display.width', 220)
    cols = ['CountrySet', 'Spec', 'Method', 'Countries', 'Mean_MSE', 'GM_RelMSE',
            'Mean_Skill', 'N_beat_persistence', 'Mean_MASE', 'Mean_R2_test_mean',
            'Mean_R2_OOS']
    print("\n=== Overall (sorted by GM_RelMSE within set/spec) ===")
    print(overall.sort_values(['CountrySet', 'Spec', 'GM_RelMSE'])[cols]
          .to_string(index=False, float_format=lambda v: f'{v:.4f}'))


if __name__ == '__main__':
    main()
