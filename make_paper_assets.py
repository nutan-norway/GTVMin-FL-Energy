"""
Tables (LaTeX + CSV) and figures from results_revised/.

Main setting:     30-country Europe set, levels specification.
Robustness:       growth specification; the submitted 18-country set.
Run after run_revised_experiments.py, revised_statistics.py and
heterogeneity_analysis.py. Writes to results_revised/paper/.
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from gtvmin_fl_revised import (COUNTRY_SETS, REGIONS, EnergyPanel, build_graph)
from revised_statistics import block_bootstrap_gm, paired_frame

RES = 'results_revised'
OUT = f'{RES}/paper'
MAIN_SET, MAIN_SPEC = 'europe', 'levels'
SEED = 20261102

# Categorical slots, fixed order (reference palette, light mode).
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = (
    '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300')
INK, MUTED, GRID = '#1f1f1e', '#6b6a63', '#d9d8d2'

METHOD_LABELS = {
    'Persistence': 'Persistence ($\\hat y_t = y_{t-1}$)',
    'Drift': 'Random walk with drift',
    'Local': 'Local-only (Ridge)',
    'Local_RW10': 'Local-only, 10-year window',
    'FL_similarity': 'GTVMin FL (similarity graph)',
    'Pooled_GlobalScale': 'Pooled, global scaling',
    'Pooled_LocalScale': 'Pooled, per-country scaling',
    'Pooled_FE': 'Pooled, country fixed effects',
    'Hierarchical': 'Hierarchical (mixed effects)',
    'Localorig': 'Local-only, submitted ($\\alpha=5$)',
    'FLorig_FixedGraph': 'FL, submitted (full-history graph)',
    'FLorig_OriginGraph': 'FL, submitted (per-origin graph)',
    'Pooledorig_GlobalScale': 'Pooled global scaling, submitted',
    'Pooledorig_FE': 'Pooled fixed effects, submitted',
}
ISO3 = {
    'Austria': 'AUT', 'Belgium': 'BEL', 'Bulgaria': 'BGR', 'Croatia': 'HRV',
    'Cyprus': 'CYP', 'Czechia': 'CZE', 'Denmark': 'DNK', 'Estonia': 'EST',
    'Finland': 'FIN', 'France': 'FRA', 'Germany': 'DEU', 'Greece': 'GRC',
    'Hungary': 'HUN', 'Iceland': 'ISL', 'Ireland': 'IRL', 'Italy': 'ITA',
    'Latvia': 'LVA', 'Lithuania': 'LTU', 'Luxembourg': 'LUX', 'Malta': 'MLT',
    'Netherlands': 'NLD', 'Norway': 'NOR', 'Poland': 'POL', 'Portugal': 'PRT',
    'Romania': 'ROU', 'Slovakia': 'SVK', 'Slovenia': 'SVN', 'Spain': 'ESP',
    'Sweden': 'SWE', 'Switzerland': 'CHE', 'United Kingdom': 'GBR',
}
MAIN_METHODS = ['Persistence', 'Drift', 'Local', 'Local_RW10', 'FL_similarity',
                'Pooled_GlobalScale', 'Pooled_LocalScale', 'Pooled_FE',
                'Hierarchical']
GRAPH_LABELS = {
    'FL_similarity': 'Sparse similarity (per origin)',
    'FL_similarity_fullhistory': 'Sparse similarity (full history)',
    'FL_complete_similarity': 'Complete, similarity weights',
    'FL_complete_equal': 'Complete, equal weights',
    'FL_geographic': 'Geographic regions',
    'FL_param_knn': 'Parameter $k$-NN',
}


def setup_matplotlib():
    plt.rcParams.update({
        'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
        'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'text.color': INK,
        'xtick.color': MUTED, 'ytick.color': MUTED,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.5,
        'lines.linewidth': 1.5, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
    })


def to_latex(df, path, caption, label, col_format=None):
    tex = df.to_latex(index=False, escape=False, column_format=col_format,
                      caption=caption, label=label, position='htbp')
    tex = tex.replace('\\begin{table}[htbp]', '\\begin{table*}[htbp]\n\\centering')
    tex = tex.replace('\\end{table}', '\\end{table*}')
    with open(path, 'w') as f:
        f.write(tex)


def fmt(v, d=3):
    return '--' if pd.isna(v) else f'{v:.{d}f}'


# ======================================================================
# TABLES
# ======================================================================

def gm_ci_vs(preds, cset, spec, method, bench, rng):
    g = preds[(preds['CountrySet'] == cset) & (preds['Spec'] == spec)]
    return block_bootstrap_gm(paired_frame(g, method, bench), rng)


def table_main(preds, overall, rng, cset, spec, methods, name, caption):
    o = overall[(overall['CountrySet'] == cset) & (overall['Spec'] == spec)].set_index('Method')
    rows = []
    for m in methods:
        r = o.loc[m]
        if m == 'Persistence':
            gm = '1.000 (reference)'
        else:
            est, lo, hi = gm_ci_vs(preds, cset, spec, m, 'Persistence', rng)
            gm = f'{est:.3f} [{lo:.3f}, {hi:.3f}]'
        rows.append({'Method': METHOD_LABELS[m],
                     'MSE (TWh$^2$)': fmt(r['Mean_MSE'], 1),
                     'MAE (TWh)': fmt(r['Mean_MAE'], 2),
                     'MAPE (\\%)': fmt(r['Mean_MAPE'], 2),
                     'MASE': fmt(r['Mean_MASE'], 3),
                     'GM rel. MSE [95\\% CI]': gm,
                     'Beat pers.': f"{int(r['N_beat_persistence'])}/{int(r['Countries'])}",
                     '$R^2_{OOS}$': fmt(r['Mean_R2_OOS'], 3)})
    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT}/{name}.csv', index=False)
    to_latex(df, f'{OUT}/{name}.tex', caption, f'tab:{name}', 'lccccccc')
    return df


def table_per_country(summary, dm, profiles):
    s = summary[(summary['CountrySet'] == MAIN_SET) & (summary['Spec'] == MAIN_SPEC)]
    mse = s.pivot(index='Country', columns='Method', values='MSE')
    skill = s.pivot(index='Country', columns='Method', values='Skill_vs_Persistence')
    d = dm[(dm['CountrySet'] == MAIN_SET) & (dm['Spec'] == MAIN_SPEC)
           & (dm['A'] == 'FL_similarity') & (dm['B'] == 'Local')].set_index('Country')
    prof = profiles[profiles['CountrySet'] == MAIN_SET].set_index('country')
    df = pd.DataFrame({
        'Country': mse.index,
        'Mean $E$ (TWh)': [fmt(np.exp(prof.loc[c, 'log_energy_level']), 0) for c in mse.index],
        'MSE pers.': mse['Persistence'].map(lambda v: fmt(v, 2)).values,
        'MSE local': mse['Local'].map(lambda v: fmt(v, 2)).values,
        'MSE FL': mse['FL_similarity'].map(lambda v: fmt(v, 2)).values,
        'MSE hier.': mse['Hierarchical'].map(lambda v: fmt(v, 2)).values,
        'Skill local': skill['Local'].map(fmt).values,
        'Skill FL': skill['FL_similarity'].map(fmt).values,
        'DM $p$ (FL vs local)': d.loc[mse.index, 'p_DM'].map(fmt).values,
    })
    order = np.argsort(-skill['FL_similarity'].values)
    df = df.iloc[order]
    df.to_csv(f'{OUT}/table_per_country.csv', index=False)
    to_latex(df, f'{OUT}/table_per_country.tex',
             'Per-country rolling-origin performance (30-country set, levels '
             'specification, 17 one-step-ahead forecasts per country, 2006--2022). '
             'MSE in TWh$^2$. Skill $= 1 - \\mathrm{MSE}/\\mathrm{MSE}_{\\text{persistence}}$. '
             'DM: Diebold--Mariano test with HLN correction (unadjusted $p$).',
             'tab:per_country', 'lcccccccc')


def table_graphs(overall, gstats, diag, pooled):
    o = overall[(overall['CountrySet'] == MAIN_SET) & (overall['Spec'] == MAIN_SPEC)].set_index('Method')
    g = gstats[(gstats['CountrySet'] == MAIN_SET) & (gstats['Spec'] == MAIN_SPEC)]
    dg = diag[(diag['CountrySet'] == MAIN_SET) & (diag['Spec'] == MAIN_SPEC)]
    po = pooled[(pooled['CountrySet'] == MAIN_SET) & (pooled['Spec'] == MAIN_SPEC)]
    rows = []
    for m, label in GRAPH_LABELS.items():
        kind = m[len('FL_'):]
        gg = g[g['Graph'] == kind]
        if m == 'FL_similarity':
            p = '--'
        else:
            hit = po[((po['A'] == 'FL_similarity') & (po['B'] == m))
                     | ((po['A'] == m) & (po['B'] == 'FL_similarity'))]
            p = fmt(hit['p_diff'].iloc[0]) if len(hit) else '--'
        rows.append({'Graph': label,
                     'Edges': f"{gg['Edges'].mean():.0f}",
                     'Density': fmt(gg['Density'].mean(), 3),
                     'Median $\\lambda$': f"{gg['Lambda'].median():g}",
                     'Median rel. shift': fmt(dg[dg['Method'] == m]['ParamDistRel'].median(), 3),
                     'MSE (TWh$^2$)': fmt(o.loc[m, 'Mean_MSE'], 1),
                     'GM rel. MSE': fmt(o.loc[m, 'GM_RelMSE'], 3),
                     'Floats sent ($10^3$)': f"{gg['FloatsCommunicated'].sum() / 1e3:.0f}",
                     '$p$ vs sparse': p})
    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT}/table_graphs.csv', index=False)
    to_latex(df, f'{OUT}/table_graphs.tex',
             'Collaboration-graph comparison (30-country set, levels specification). '
             'Each graph is rebuilt at every origin from data up to that origin, '
             'except the full-history variant (diagnostic). $\\lambda$ is selected '
             'per origin by inner validation. Rel.\\ shift: '
             '$\\|\\theta_{FL}-\\theta_{local}\\|/\\|\\theta_{local}\\|$. Floats sent: '
             'total parameters exchanged over all 17 origins. $p$: pooled test of '
             'the mean relative loss difference against the sparse similarity graph, '
             'standard errors clustered by year.',
             'tab:graphs', 'lcccccccc')


def table_fl_vs_local(pooled):
    p = pooled[(pooled['A'] == 'FL_similarity') & (pooled['B'] == 'Local')].copy()
    p['Setting'] = p['CountrySet'].map({'europe': '30 countries', 'paper18': '18 countries'}) \
        + ', ' + p['Spec']
    df = pd.DataFrame({
        'Setting': p['Setting'],
        'Mean rel. diff.': p['MeanRelDiff'].map(fmt),
        '90\\% CI': [f'[{a:.3f}, {b:.3f}]' for a, b in zip(p['CI90_low'], p['CI90_high'])],
        '$p$ (difference)': p['p_diff'].map(fmt),
        '$p$ (TOST, $\\pm 5\\%$)': p['p_TOST'].map(fmt),
        'GM rel. MSE [95\\% CI]': [f'{g:.3f} [{a:.3f}, {b:.3f}]' for g, a, b in
                                   zip(p['GM_RelMSE'], p['GM_CI95_low'], p['GM_CI95_high'])],
        'DM: FL / local better': [f'{a} / {b}' for a, b in
                                  zip(p['N_A_better_DM05'], p['N_B_better_DM05'])],
    })
    df.to_csv(f'{OUT}/table_fl_vs_local.csv', index=False)
    to_latex(df, f'{OUT}/table_fl_vs_local.tex',
             'Federated (similarity graph) versus local-only forecasts. Mean relative '
             'difference: average over country-years of '
             '$(SE_{FL}-SE_{local})/\\mathrm{MSE}_{local,c}$ (negative favours FL), '
             'standard errors clustered by forecast year. TOST: equivalence test with '
             'margin $\\pm5\\%$ of the local MSE. GM rel.\\ MSE: geometric mean over '
             'countries of $\\mathrm{MSE}_{FL}/\\mathrm{MSE}_{local}$, moving-block '
             'bootstrap CI over years. Last column: countries with a significant '
             'Diebold--Mariano difference at 5\\% (unadjusted).',
             'tab:fl_vs_local', 'lcccccc')


def table_robustness(overall):
    methods = ['Local', 'FL_similarity', 'FL_geographic', 'Pooled_GlobalScale',
               'Pooled_FE', 'Hierarchical', 'Local_RW10', 'Drift']
    settings = [('europe', 'levels'), ('europe', 'growth'),
                ('paper18', 'levels'), ('paper18', 'growth')]
    o = overall.set_index(['CountrySet', 'Spec', 'Method'])
    rows = []
    for m in methods:
        row = {'Method': METHOD_LABELS.get(m, GRAPH_LABELS.get(m, m))}
        for cset, spec in settings:
            n = 30 if cset == 'europe' else 18
            r = o.loc[(cset, spec, m)]
            row[f'{n}, {spec}'] = (f"{r['GM_RelMSE']:.3f} "
                                   f"({int(r['N_beat_persistence'])}/{n})")
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT}/table_robustness.csv', index=False)
    to_latex(df, f'{OUT}/table_robustness.tex',
             'Robustness across specifications and country sets: geometric-mean '
             'MSE relative to persistence (below 1 beats persistence) and, in '
             'parentheses, the number of countries in which the method beats '
             'persistence.', 'tab:robustness', 'lcccc')


def table_submitted_vs_revised(overall, diag):
    o = overall.set_index(['CountrySet', 'Spec', 'Method'])
    rows = []
    for cset in ['paper18', 'europe']:
        for m in ['Localorig', 'FLorig_FixedGraph', 'FLorig_OriginGraph',
                  'Pooledorig_GlobalScale', 'Pooledorig_FE',
                  'Local', 'FL_similarity', 'Pooled_GlobalScale', 'Pooled_FE']:
            r = o.loc[(cset, 'levels', m)]
            dg = diag[(diag['CountrySet'] == cset) & (diag['Spec'] == 'levels')
                      & (diag['Method'] == m)]
            rows.append({'Set': cset, 'Method': METHOD_LABELS[m],
                         'MSE': fmt(r['Mean_MSE'], 2),
                         'Mean $R^2$ (test mean)': fmt(r['Mean_R2_test_mean'], 3),
                         'GM rel. MSE': fmt(r['GM_RelMSE'], 3),
                         'Median rel. shift': fmt(dg['ParamDistRel'].median(), 5)
                         if len(dg) else '--'})
    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT}/table_submitted_vs_revised.csv', index=False)
    to_latex(df, f'{OUT}/table_submitted_vs_revised.tex',
             'Submitted configuration reproduced in the revised pipeline, next to the '
             'revised methods (levels specification). Rel.\\ shift as in '
             'Table~\\ref{tab:graphs}.', 'tab:submitted_vs_revised', 'llcccc')


def table_heterogeneity(slope, cor):
    s = slope.copy()
    s['Setting'] = s['CountrySet'] + ', ' + s['Spec']
    df = s[['Setting', 'Countries', 'N', 'df1', 'df2', 'F', 'p']].copy()
    df['F'] = df['F'].map(lambda v: f'{v:.2f}')
    df['p'] = df['p'].map(lambda v: f'{v:.1e}')
    df.to_csv(f'{OUT}/table_slope_homogeneity.csv', index=False)
    to_latex(df, f'{OUT}/table_slope_homogeneity.tex',
             'Slope-homogeneity $F$-test: common slopes with country intercepts '
             'versus country-specific slopes (OLS, 1996--2022).',
             'tab:slope_tests', 'lcccccc')
    sig = cor[(cor['CountrySet'] == MAIN_SET) & (cor['p'] < 0.05)].sort_values('p')
    sig = sig[['Spec', 'Outcome', 'Characteristic', 'n', 'Spearman_rho', 'p', 'p_Holm']]
    sig.to_csv(f'{OUT}/table_heterogeneity_correlations.csv', index=False)


# ======================================================================
# FIGURES
# ======================================================================

def fig_lambda_path(path_summary, overall):
    ps = path_summary[(path_summary['CountrySet'] == MAIN_SET)
                      & (path_summary['Method'] == 'FLpath_similarity')]
    lam_pos = lambda lam: np.where(lam == 0, 1e-4, lam)
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.6))
    for spec, color in [('levels', BLUE), ('growth', ORANGE)]:
        g = ps[ps['Spec'] == spec].sort_values('Lambda')
        x = lam_pos(g['Lambda'].values)
        axes[0].plot(x, g['GM_RelMSE'], color=color, marker='o', ms=4,
                     label=f'{spec} (main)' if spec == 'levels' else f'{spec} (robustness)')
        axes[1].plot(x[1:], g['Median_ParamDistRel'].values[1:], color=color,
                     marker='o', ms=4)
        best = g.loc[g['GM_RelMSE'].idxmin()]
        axes[0].annotate(f"min at $\\lambda$={best['Lambda']:g}",
                         (lam_pos(np.array(best['Lambda'])), best['GM_RelMSE']),
                         textcoords='offset points',
                         xytext=(0, -12) if spec == 'levels' else (0, 8),
                         ha='center', fontsize=6.5, color=INK)
    axes[0].axhline(1.0, color=MUTED, ls='--', lw=0.8)
    axes[0].text(2e4, 1.003, 'persistence', ha='right', va='bottom', fontsize=6.5, color=MUTED)
    for ax in axes:
        ax.set_xscale('log')
        ax.set_xlabel('consensus weight $\\lambda$')
    ticks = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4]
    axes[0].set_xticks(ticks)
    axes[0].set_xticklabels(['0\n(local)', '$10^{-3}$', '', '$10^{-1}$', '', '$10^{1}$',
                             '', '$10^{3}$', ''])
    axes[0].set_ylabel('GM of MSE / MSE$_{\\mathrm{persistence}}$')
    axes[0].set_title('(a) Accuracy along the local-to-pooled path', loc='left')
    axes[1].set_yscale('log')
    axes[1].set_ylabel('median $\\|\\theta_{FL}-\\theta_{local}\\| / \\|\\theta_{local}\\|$')
    axes[1].set_title('(b) Parameter shift caused by collaboration', loc='left')
    axes[0].legend(frameon=False, loc='upper left')
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig_lambda_path.pdf')
    fig.savefig(f'{OUT}/fig_lambda_path.png')
    plt.close(fig)


def fig_convergence(trace):
    t = trace[(trace['CountrySet'] == MAIN_SET) & (trace['Spec'] == MAIN_SPEC)]
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    colors = [BLUE, ORANGE, AQUA, YELLOW]
    for color, (lam, g) in zip(colors, t.groupby('Lambda')):
        tuned = ' (selected)' if g['Tuned'].iloc[0] else ''
        ax.semilogy(g['Round'], g['DistToExact'].clip(lower=1e-14), color=color,
                    label=f'$\\lambda$ = {lam:g}{tuned}')
    ax.set_xlabel('communication round')
    ax.set_ylabel('$\\|\\Theta_k - \\Theta^\\star\\|_2$')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig_convergence.pdf')
    fig.savefig(f'{OUT}/fig_convergence.png')
    plt.close(fig)


def fig_network(origin=2021):
    panel = EnergyPanel('owid-energy-data.csv', COUNTRY_SETS[MAIN_SET])
    nodes = panel.countries
    G = build_graph('similarity', nodes, means=panel.profile_means(origin))
    region = {c: r for r, cs in REGIONS.items() for c in cs}
    region_color = dict(zip(REGIONS, [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN]))
    # Stress-based layout: graph distances become drawing distances, which
    # spreads the dense core instead of collapsing it as spring layouts do.
    pos = nx.kamada_kawai_layout(G, weight=None)
    fig, ax = plt.subplots(figsize=(7.16, 5.2))
    normal = [(u, v) for u, v, d in G.edges(data=True) if not d.get('fallback')]
    fallback = [(u, v) for u, v, d in G.edges(data=True) if d.get('fallback')]
    nx.draw_networkx_edges(G, pos, edgelist=normal, width=0.4, edge_color=GRID, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=fallback, width=1.0, style='dashed',
                           edge_color=MUTED, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=140, edgecolors='white', linewidths=1.5,
                           node_color=[region_color[region[n]] for n in G.nodes()], ax=ax)
    for n, (x, y) in pos.items():
        ax.text(x + 0.025, y, ISO3[n], ha='left', va='center', fontsize=6.5, color=INK)
    handles = [plt.Line2D([], [], marker='o', ls='', color=c, label=r, ms=6)
               for r, c in region_color.items()]
    handles.append(plt.Line2D([], [], color=MUTED, ls='--', label='connectivity fallback edge'))
    ax.legend(handles=handles, frameon=False, loc='lower left', ncol=4, fontsize=6.5)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig_network.pdf')
    fig.savefig(f'{OUT}/fig_network.png')
    plt.close(fig)
    return G.number_of_edges(), nx.density(G), len(fallback)


def fig_country_skill(summary):
    s = summary[(summary['CountrySet'] == MAIN_SET) & (summary['Spec'] == MAIN_SPEC)]
    r = s.pivot(index='Country', columns='Method', values='RelMSE_vs_Persistence')
    r = r.sort_values('FL_similarity', ascending=False)
    y = np.arange(len(r))
    fig, ax = plt.subplots(figsize=(3.5, 5.6))
    ax.hlines(y, r['Local'], r['FL_similarity'], color=GRID, lw=1.5, zorder=1)
    ax.scatter(r['Local'], y, s=22, color=ORANGE, edgecolors='white', linewidths=1,
               label='local-only', zorder=2)
    ax.scatter(r['FL_similarity'], y, s=22, color=BLUE, edgecolors='white',
               linewidths=1, label='GTVMin FL', zorder=3)
    ax.axvline(1.0, color=MUTED, ls='--', lw=0.8)
    ax.set_xscale('log')
    ax.set_yticks(y)
    ax.set_yticklabels(r.index)
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('MSE / MSE$_{\\mathrm{persistence}}$ (log scale)')
    ax.legend(frameon=False, loc='upper right')
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig_country_skill.pdf')
    fig.savefig(f'{OUT}/fig_country_skill.png')
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    setup_matplotlib()
    rng = np.random.default_rng(SEED)
    preds = pd.read_csv(f'{RES}/predictions.csv')
    overall = pd.read_csv(f'{RES}/summary_overall.csv')
    summary = pd.read_csv(f'{RES}/summary_by_country.csv')
    gstats = pd.read_csv(f'{RES}/graph_stats.csv')
    diag = pd.read_csv(f'{RES}/fl_diagnostics.csv')
    pooled = pd.read_csv(f'{RES}/stats_pooled.csv')
    dm = pd.read_csv(f'{RES}/stats_dm_by_country.csv')
    path_summary = pd.read_csv(f'{RES}/lambda_path_summary.csv')
    trace = pd.read_csv(f'{RES}/convergence_trace.csv')
    profiles = pd.read_csv(f'{RES}/heterogeneity_profiles.csv')
    slope = pd.read_csv(f'{RES}/heterogeneity_slope_tests.csv')
    cor = pd.read_csv(f'{RES}/heterogeneity_correlations.csv')

    table_main(preds, overall, rng, MAIN_SET, MAIN_SPEC, MAIN_METHODS, 'table_main',
               'Rolling-origin one-step-ahead forecasting performance, 30 European '
               'countries, levels specification (17 origins, 2005--2021, predicting '
               '2006--2022). MSE, MAE, MAPE and MASE are averaged over countries; '
               'GM rel.\\ MSE is the geometric mean over countries of '
               '$\\mathrm{MSE}/\\mathrm{MSE}_{\\text{persistence}}$ with a moving-block '
               'bootstrap 95\\% CI; $R^2_{OOS}$ uses the training-period mean as '
               'benchmark. All hyperparameters are selected at each origin from '
               'training years only.')
    table_main(preds, overall, rng, MAIN_SET, 'growth', MAIN_METHODS[:5] +
               ['Pooled_GlobalScale', 'Pooled_FE', 'Hierarchical'], 'table_main_growth',
               'As Table~\\ref{tab:table_main}, growth-rate specification (robustness).')
    table_per_country(summary, dm, profiles)
    table_graphs(overall, gstats, diag, pooled)
    table_fl_vs_local(pooled)
    table_robustness(overall)
    table_submitted_vs_revised(overall, diag)
    table_heterogeneity(slope, cor)

    fig_lambda_path(path_summary, overall)
    fig_convergence(trace)
    edges, density, n_fallback = fig_network()
    fig_country_skill(summary)
    print(f"Network (origin 2021): {edges} edges, density {density:.3f}, "
          f"{n_fallback} fallback edges")
    print(f"Wrote tables and figures to {OUT}/")


if __name__ == '__main__':
    main()
