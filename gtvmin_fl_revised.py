# Revised code for GTVMIN-FL
# Author: Nutan Gupta, University of Inland Norway
# Date: 10 November 2026

import warnings

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


# ======================================================================
# CONFIGURATION
# ======================================================================

TARGET = 'primary_energy_consumption'   # TWh
RAW_COLS = [TARGET, 'gdp', 'population']
FEATURES = ['x_energy', 'x_gdp', 'x_pop']

# 19 intended countries, of which 'Czech Republic' was
# dropped because OWID names it 'Czechia' -> 18 countries.
PAPER18 = [
    'Austria', 'Croatia', 'Cyprus', 'Denmark', 'Estonia', 'Finland',
    'Greece', 'Hungary', 'Iceland', 'Latvia', 'Lithuania', 'Luxembourg',
    'Norway', 'Poland', 'Romania', 'Slovakia', 'Slovenia', 'Switzerland',
]
EU27 = [
    'Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czechia',
    'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece',
    'Hungary', 'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg',
    'Malta', 'Netherlands', 'Poland', 'Portugal', 'Romania', 'Slovakia',
    'Slovenia', 'Spain', 'Sweden',
]
COUNTRY_SETS = {
    'paper18': PAPER18,
    'original19': PAPER18 + ['Czechia'],
    # EU-27 + EFTA members covered by OWID + United Kingdom. The inclusion
    # rule (complete data over the window) is applied in EnergyPanel.
    'europe': EU27 + ['Iceland', 'Norway', 'Switzerland', 'United Kingdom'],
}

# Every country belongs to exactly one region.
REGIONS = {
    'Nordic': ['Denmark', 'Finland', 'Iceland', 'Norway', 'Sweden'],
    'Baltic': ['Estonia', 'Latvia', 'Lithuania'],
    'Central': ['Austria', 'Czechia', 'Germany', 'Hungary', 'Poland',
                'Slovakia', 'Slovenia', 'Switzerland'],
    'Western': ['Belgium', 'France', 'Ireland', 'Luxembourg', 'Netherlands',
                'United Kingdom'],
    'Southern': ['Croatia', 'Cyprus', 'Greece', 'Italy', 'Malta', 'Portugal',
                 'Spain'],
    'SouthEastern': ['Bulgaria', 'Romania'],
}
ADJACENT_REGIONS = [
    ('Nordic', 'Baltic'), ('Nordic', 'Western'), ('Nordic', 'Central'),
    ('Baltic', 'Central'), ('Central', 'Western'), ('Central', 'Southern'),
    ('Central', 'SouthEastern'), ('Southern', 'SouthEastern'),
    ('Western', 'Southern'),
]
INTRA_REGION_WEIGHT, INTER_REGION_WEIGHT = 0.9, 0.6

SIMILARITY_THRESHOLD = 0.8
FALLBACK_WEIGHT = 0.5
BETA_GRID = [0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
LAMBDA_GRID = [0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
JITTER = 1e-10
# Stopping rule of the federated solver: max parameter change per round
# <= FL_TOL * (1 + max|theta|). Deviation from the exact GTVMin solution
# is recorded in graph_stats so the effect of this tolerance is visible.
FL_TOL = 1e-6

# previous configuration, used only to reproduce the original FL.
ORIGINAL_FL = dict(learning_rate=0.005, num_rounds=30, lambda_param=0.05,
                   personalization=0.3, ridge_alpha=5.0, clip_bound=10.0)


# ======================================================================
# DATA
# ======================================================================

class EnergyPanel:
    """
    Country panel with a common evaluation window. A country is included
    if it has complete primary energy, GDP and population data for every
    year from (first_target_year - 2) to last_year; this is the explicit
    country-selection rule.
    """

    def __init__(self, data_path, countries, first_target_year=1996,
                 last_year=2022):
        df = pd.read_csv(data_path, usecols=['country', 'year'] + RAW_COLS)
        self.first_target_year = first_target_year
        self.last_year = last_year
        window = np.arange(first_target_year - 2, last_year + 1)

        self.raw, self.excluded = {}, {}
        for c in countries:
            d = (df[df['country'] == c].set_index('year')
                 .reindex(window)[RAW_COLS])
            missing = d.index[d.isna().any(axis=1)].tolist()
            if missing:
                self.excluded[c] = f"missing years {missing}"
                continue
            self.raw[c] = d
        self.countries = list(self.raw)
        self.designs = {spec: {c: self._design(c, spec) for c in self.countries}
                        for spec in ('levels', 'growth')}

    def _design(self, country, spec):
        d = self.raw[country]
        years = np.arange(self.first_target_year, self.last_year + 1)

        def at(series, lag):
            return series.reindex(years - lag).values

        E, gdp, pop = d[TARGET], d['gdp'], d['population']
        out = pd.DataFrame({'year': years, 'E': at(E, 0), 'E_prev': at(E, 1)})
        if spec == 'levels':
            # specification: lag-1 levels, target in TWh.
            X = np.column_stack([at(E, 1), at(gdp, 1), at(pop, 1)])
            y = at(E, 0)
        elif spec == 'growth':
            # Growth specification: percent log-growth, all lagged one year.
            lE, lg, lp = np.log(E), np.log(gdp), np.log(pop)
            y = 100 * (at(lE, 0) - at(lE, 1))
            X = 100 * np.column_stack([at(lE, 1) - at(lE, 2),
                                       at(lg, 1) - at(lg, 2),
                                       at(lp, 1) - at(lp, 2)])
        else:
            raise ValueError(spec)
        out[FEATURES] = X
        out['y'] = y
        return out

    def profile_means(self, max_year=None):
        """Mean lag-1 (energy, GDP, population) levels per country, using
        target years <= max_year only (None = full history, i.e. the leaky
        construction of the previous version)."""
        rows = []
        for c in self.countries:
            d = self.designs['levels'][c]
            if max_year is not None:
                d = d[d['year'] <= max_year]
            rows.append(d[FEATURES].mean().values)
        return np.array(rows)


class LocalTransform:
    """
    Model-unit transform for one country, fit on its training rows only.
    levels: per-country standardization of X and y (as submitted).
    growth: identity -- percent growth rates are already unit-free and
            comparable across countries, so parameters can be shared.
    """

    def __init__(self, spec):
        self.spec = spec

    def fit(self, X, y):
        if self.spec == 'levels':
            self.sx = StandardScaler().fit(X)
            self.sy = StandardScaler().fit(y.reshape(-1, 1))
        return self

    def tx(self, X):
        return self.sx.transform(X) if self.spec == 'levels' else X

    def ty(self, y):
        return (self.sy.transform(y.reshape(-1, 1)).ravel()
                if self.spec == 'levels' else y)

    def inv(self, y):
        return (self.sy.inverse_transform(np.reshape(y, (-1, 1))).ravel()
                if self.spec == 'levels' else np.asarray(y, dtype=float))


def to_twh(spec, pred, E_prev):
    """Convert a raw-target-unit prediction (TWh or % growth) to TWh."""
    return pred if spec == 'levels' else E_prev * np.exp(pred / 100.0)


# ======================================================================
# RIDGE AND GTVMin
# ======================================================================

def penalize_intercept(spec):
    """Growth spec: the intercept (drift) is penalized too, so that every
    parameter shrinks toward 0 and beta -> inf gives exactly the persistence
    forecast (R2: shrinkage should pull toward the random walk, not toward
    the mean). Levels spec: intercept unpenalized, as submitted."""
    return spec == 'growth'


def ridge_system(X, y, beta, pen_intercept=False):
    """
    Quadratic form of Loss(theta) = (1/n)||y - Z theta||^2 + beta theta' P theta
    with Z = [X, 1] and P = diag(1, .., 1, pen_intercept):
        Loss(theta) = theta' A theta - 2 c' theta + y'y/n.
    Returns (A, c, const).
    """
    n = len(y)
    Z = np.hstack([X, np.ones((n, 1))])
    pen_mask = np.r_[np.ones(X.shape[1]), float(pen_intercept)]
    A = Z.T @ Z / n + beta * np.diag(pen_mask) + JITTER * np.eye(Z.shape[1])
    c = Z.T @ y / n
    return A, c, float(y @ y / n)


def solve_ridge(X, y, beta, pen_intercept=False):
    A, c, _ = ridge_system(X, y, beta, pen_intercept)
    return np.linalg.solve(A, c)


def predict_linear(theta, X):
    return X @ theta[:-1] + theta[-1]


def weight_matrix(graph, nodes):
    return nx.to_numpy_array(graph, nodelist=nodes, weight='weight')


class GTVMin:
    """
    GTVMin with squared-L2 variation:
        f(Theta) = sum_i [theta_i' A_i theta_i - 2 c_i' theta_i + const_i]
                   + lam * sum_{(i,j) in E} W_ij ||theta_i - theta_j||^2.
    Stationarity: A_i theta_i + lam sum_j W_ij (theta_i - theta_j) = c_i,
    i.e. (blockdiag(A) + lam L (x) I) Theta = c with L the graph Laplacian.
    """

    def __init__(self, A, c, const, W, lam):
        self.A, self.c, self.const = np.asarray(A), np.asarray(c), np.asarray(const)
        self.W, self.lam = W, lam
        self.m, self.d = self.c.shape

    def objective(self, Theta):
        local = (np.einsum('mi,mij,mj->m', Theta, self.A, Theta)
                 - 2 * np.einsum('mi,mi->m', self.c, Theta) + self.const).sum()
        sq = ((Theta[:, None, :] - Theta[None, :, :]) ** 2).sum(-1)
        return local + self.lam * 0.5 * (self.W * sq).sum()

    def exact(self):
        L = np.diag(self.W.sum(1)) - self.W
        H = np.zeros((self.m * self.d, self.m * self.d))
        for i in range(self.m):
            H[i*self.d:(i+1)*self.d, i*self.d:(i+1)*self.d] = self.A[i]
        H += self.lam * np.kron(L, np.eye(self.d))
        return np.linalg.solve(H, self.c.ravel()).reshape(self.m, self.d)

    def federated(self, Theta0=None, tol=1e-10, max_iter=200000, trace=False):
        """
        Block-Jacobi message passing: each round every node receives its
        neighbours' current parameters and solves its local proximal problem
            theta_i <- argmin Loss_i(theta) + lam sum_j W_ij ||theta - theta_j||^2
                     = (A_i + lam d_i I)^{-1} (c_i + lam sum_j W_ij theta_j).
        Converges to the exact GTVMin solution because blockdiag(A) is
        positive definite and the signless Laplacian D + W is PSD.
        """
        deg = self.W.sum(1)
        Minv = np.linalg.inv(self.A + self.lam * deg[:, None, None] * np.eye(self.d))
        Theta = (np.linalg.solve(self.A, self.c[..., None])[..., 0]
                 if Theta0 is None else Theta0.copy())
        history = [self.objective(Theta)] if trace else None
        thetas = [Theta.copy()] if trace else None
        converged, it = False, 0
        for it in range(1, max_iter + 1):
            new = np.einsum('mij,mj->mi', Minv, self.c + self.lam * self.W @ Theta)
            delta = np.abs(new - Theta).max()
            Theta = new
            if trace:
                history.append(self.objective(Theta))
                thetas.append(Theta.copy())
            if delta <= tol * (1 + np.abs(Theta).max()):
                converged = True
                break
        info = {'iterations': it, 'converged': converged}
        if trace:
            info['objective'] = history
            info['thetas'] = thetas
        return Theta, info


# ======================================================================
# GRAPHS
# ======================================================================

def cosine_similarity_matrix(means):
    Z = StandardScaler().fit_transform(means)
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Zn = Z / norms
    return Zn @ Zn.T


def _connect_components(G, nodes, S):
    """Deterministically join components by the highest-similarity
    cross-component pair (weight FALLBACK_WEIGHT), until connected."""
    idx = {n: k for k, n in enumerate(nodes)}
    while not nx.is_connected(G):
        comp_of = {}
        for k, comp in enumerate(nx.connected_components(G)):
            for n in comp:
                comp_of[n] = k
        best, pair = -np.inf, None
        for a in nodes:
            for b in nodes:
                if comp_of[a] < comp_of[b] and S[idx[a], idx[b]] > best:
                    best, pair = S[idx[a], idx[b]], (a, b)
        G.add_edge(*pair, weight=FALLBACK_WEIGHT, fallback=True)


def build_graph(kind, nodes, means=None, thetas=None, knn=3):
    """
    kind:
      similarity            cosine similarity of standardized mean lag-1
                            profiles > threshold (means must use data up to
                            the origin only)
      similarity_fullhistory  same, with full-history means (leaky; diagnostic)
      complete_similarity   all pairs, weight max(sim, 0.1)
      complete_equal        all pairs, weight 1
      geographic            region membership (intra 0.9, adjacent 0.6)
      param_knn             k-nearest neighbours in local-parameter space,
                            Gaussian-kernel weights; needs only the model
                            parameters that FL exchanges anyway
    """
    G = nx.Graph()
    G.add_nodes_from(nodes)
    m = len(nodes)
    if kind in ('similarity', 'similarity_fullhistory', 'complete_similarity'):
        S = cosine_similarity_matrix(means)
        for i in range(m):
            for j in range(i + 1, m):
                if kind == 'complete_similarity':
                    G.add_edge(nodes[i], nodes[j], weight=max(S[i, j], 0.1))
                elif S[i, j] > SIMILARITY_THRESHOLD:
                    G.add_edge(nodes[i], nodes[j], weight=S[i, j])
        _connect_components(G, nodes, S)
    elif kind == 'complete_equal':
        for i in range(m):
            for j in range(i + 1, m):
                G.add_edge(nodes[i], nodes[j], weight=1.0)
    elif kind == 'geographic':
        region = {c: r for r, cs in REGIONS.items() for c in cs}
        adjacent = {frozenset(p) for p in ADJACENT_REGIONS}
        for i in range(m):
            for j in range(i + 1, m):
                ri, rj = region[nodes[i]], region[nodes[j]]
                if ri == rj:
                    G.add_edge(nodes[i], nodes[j], weight=INTRA_REGION_WEIGHT)
                elif frozenset((ri, rj)) in adjacent:
                    G.add_edge(nodes[i], nodes[j], weight=INTER_REGION_WEIGHT)
        _connect_components(G, nodes, np.zeros((m, m)))
    elif kind == 'param_knn':
        Z = StandardScaler().fit_transform(thetas)
        D2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
        sigma2 = np.median(D2[np.triu_indices(m, 1)])
        K = np.exp(-D2 / sigma2)
        for i in range(m):
            for j in np.argsort(D2[i])[1:knn + 1]:
                G.add_edge(nodes[i], nodes[j], weight=K[i, j])
        _connect_components(G, nodes, K)
    else:
        raise ValueError(kind)
    return G


# ======================================================================
# NESTED (INNER) VALIDATION
# ======================================================================

def inner_splits(years, n_val=5, min_fit=6):
    """One-step-ahead inner validation on the last n_val training years:
    for each validation year v, fit on years < v and predict v."""
    years = np.asarray(years)
    out = []
    for v in years[-n_val:]:
        fit = years < v
        if fit.sum() >= min_fit:
            out.append((fit, years == v))
    return out


def beta_validation_errors(Xraw, yraw, years, spec, grid, n_val, min_fit):
    """Inner one-step-ahead squared errors (model units) per beta value."""
    errs = np.zeros(len(grid))
    for fit, val in inner_splits(years, n_val, min_fit):
        tf = LocalTransform(spec).fit(Xraw[fit], yraw[fit])
        Xf, yf = tf.tx(Xraw[fit]), tf.ty(yraw[fit])
        Xv, yv = tf.tx(Xraw[val]), tf.ty(yraw[val])
        for k, b in enumerate(grid):
            theta = solve_ridge(Xf, yf, b, penalize_intercept(spec))
            errs[k] += ((predict_linear(theta, Xv) - yv) ** 2).sum()
    return errs


def tune_beta_local(Xraw, yraw, years, spec, grid, n_val, min_fit):
    """Per-country beta from that country's own inner validation."""
    errs = beta_validation_errors(Xraw, yraw, years, spec, grid, n_val, min_fit)
    return grid[int(np.argmin(errs))]


def tune_beta_network(train, nodes, spec, grid, n_val, min_fit):
    """One beta for all nodes: each node reports only its inner validation
    error per grid value (a scalar per value, no data), and the beta with
    the smallest network total is used. Far less noisy than per-country
    selection from a handful of validation years."""
    errs = sum(beta_validation_errors(*train[n], spec, grid, n_val, min_fit)
               for n in nodes)
    return grid[int(np.argmin(errs))]


def node_systems(train, spec, betas, nodes, mask=None):
    """Per-node quadratic systems (A, c, const) in model units, plus the
    validation design for each node if a (fit, val) mask pair is given."""
    A, c, const, val = [], [], [], []
    for n in nodes:
        Xr, yr, yrs = train[n]
        fit = np.ones(len(yr), bool) if mask is None else mask[0]
        tf = LocalTransform(spec).fit(Xr[fit], yr[fit])
        a, cc, k = ridge_system(tf.tx(Xr[fit]), tf.ty(yr[fit]), betas[n],
                                penalize_intercept(spec))
        A.append(a); c.append(cc); const.append(k)
        if mask is not None:
            val.append((tf.tx(Xr[mask[1]]), tf.ty(yr[mask[1]])))
    return np.array(A), np.array(c), np.array(const), val


def tune_lambda(train, spec, betas, nodes, W, grid, n_val, min_fit):
    """Network-wide lambda chosen by inner validation (sum of squared
    one-step errors in model units over all nodes)."""
    years = train[nodes[0]][2]
    errs = np.zeros(len(grid))
    for mask in inner_splits(years, n_val, min_fit):
        A, c, const, val = node_systems(train, spec, betas, nodes, mask)
        for k, lam in enumerate(grid):
            Theta = GTVMin(A, c, const, W, lam).exact()
            errs[k] += sum(((predict_linear(Theta[i], Xv) - yv) ** 2).sum()
                           for i, (Xv, yv) in enumerate(val))
    return grid[int(np.argmin(errs))], errs


# ======================================================================
# CENTRALIZED (POOLED) BASELINES
# ======================================================================

class PooledModel:
    """
    Single centralized model on pooled training data.
      scale='global': one scaler fit on the pooled raw data (levels; the
                      submitted 'global standardization' baseline). For
                      growth rates no scaling is applied.
      scale='local' : each country's data in its own LocalTransform units,
                      then pooled.
      fe=True       : country-specific intercepts, UNPENALIZED (the
                      earlier version code shrank the dummies with Ridge).
    """

    def __init__(self, spec, scale='global', fe=False):
        self.spec, self.scale, self.fe = spec, scale, fe

    def _stack(self, data, nodes):
        Xs, ys, ids = [], [], []
        self.tfs = {}
        for k, n in enumerate(nodes):
            Xr, yr = data[n][0], data[n][1]
            if self.scale == 'local':
                self.tfs[n] = LocalTransform(self.spec).fit(Xr, yr)
                Xr, yr = self.tfs[n].tx(Xr), self.tfs[n].ty(yr)
            Xs.append(Xr); ys.append(yr); ids += [k] * len(yr)
        return np.vstack(Xs), np.concatenate(ys), np.array(ids)

    def _design(self, X, ids):
        if self.fe:
            D = np.zeros((len(X), self.n_nodes))
            D[np.arange(len(X)), ids] = 1.0
            return np.hstack([X, D])
        return np.hstack([X, np.ones((len(X), 1))])

    def fit(self, data, nodes, beta):
        self.nodes, self.n_nodes = list(nodes), len(nodes)
        X, y, ids = self._stack(data, nodes)
        self.gx = self.gy = None
        if self.scale == 'global' and self.spec == 'levels':
            self.gx = StandardScaler().fit(X)
            self.gy = StandardScaler().fit(y.reshape(-1, 1))
            X, y = self.gx.transform(X), self.gy.transform(y.reshape(-1, 1)).ravel()
        Z = self._design(X, ids)
        p = X.shape[1]
        # Country dummies (fe) are never penalized; a single common
        # intercept follows the same rule as the local models.
        pen_icpt = 0.0 if self.fe else float(penalize_intercept(self.spec))
        pen = np.r_[np.ones(p), np.full(Z.shape[1] - p, pen_icpt)]
        A = Z.T @ Z / len(y) + beta * np.diag(pen) + JITTER * np.eye(Z.shape[1])
        self.theta = np.linalg.solve(A, Z.T @ y / len(y))
        return self

    def predict(self, node, Xraw):
        """Prediction in raw target units (TWh for levels, % for growth)."""
        k = self.nodes.index(node)
        if self.scale == 'local':
            X = self.tfs[node].tx(Xraw)
        elif self.gx is not None:
            X = self.gx.transform(Xraw)
        else:
            X = Xraw
        pred = self._design(X, np.full(len(X), k)) @ self.theta
        if self.scale == 'local':
            return self.tfs[node].inv(pred)
        if self.gy is not None:
            return self.gy.inverse_transform(pred.reshape(-1, 1)).ravel()
        return pred


def tune_beta_pooled(train, nodes, spec, scale, fe, grid, n_val, min_fit):
    years = train[nodes[0]][2]
    errs = np.zeros(len(grid))
    for fit, val in inner_splits(years, n_val, min_fit):
        sub = {n: (train[n][0][fit], train[n][1][fit]) for n in nodes}
        for k, b in enumerate(grid):
            model = PooledModel(spec, scale, fe).fit(sub, nodes, b)
            for n in nodes:
                Xv, yv = train[n][0][val], train[n][1][val]
                # Compare in each country's own local units so that large
                # countries do not dominate the selection criterion.
                tf = LocalTransform(spec).fit(train[n][0][fit], train[n][1][fit])
                errs[k] += ((tf.ty(model.predict(n, Xv)) - tf.ty(yv)) ** 2).sum()
    return grid[int(np.argmin(errs))]


class HierarchicalModel:
    """
    Mixed-effects partial pooling (R2, R4.7): y = x'(b + u_c) + (a + v_c) + e,
    random intercept and slopes per country, fit by REML on data in each
    country's LocalTransform units. Falls back to fewer random slopes if
    the full covariance does not converge; the variant used is recorded.
    """

    RE_FORMULAS = ['~' + '+'.join(FEATURES), '~x_energy', '~1']

    def __init__(self, spec):
        self.spec = spec

    def fit(self, data, nodes):
        import statsmodels.formula.api as smf
        frames, self.tfs = [], {}
        for n in nodes:
            Xr, yr = data[n][0], data[n][1]
            tf = LocalTransform(self.spec).fit(Xr, yr)
            self.tfs[n] = tf
            f = pd.DataFrame(tf.tx(Xr), columns=FEATURES)
            f['y'], f['g'] = tf.ty(yr), n
            frames.append(f)
        df = pd.concat(frames, ignore_index=True)
        self.result, self.re_formula = None, None
        for re_f in self.RE_FORMULAS:
            try:
                # Non-convergence is handled by the fallback chain below,
                # so statsmodels' ConvergenceWarnings are silenced here.
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    r = smf.mixedlm('y ~ ' + '+'.join(FEATURES), df, groups=df['g'],
                                    re_formula=re_f).fit(reml=True, method=['lbfgs'])
                if r.converged and np.all(np.isfinite(r.fe_params)):
                    self.result, self.re_formula = r, re_f
                    break
            except Exception:
                continue
        return self

    def predict(self, node, Xraw):
        tf = self.tfs[node]
        x = pd.Series(tf.tx(Xraw)[0], index=FEATURES)
        if self.result is None:
            return tf.inv([0.0])
        fe = self.result.fe_params
        pred = fe['Intercept'] + sum(fe[f] * x[f] for f in FEATURES)
        for name, val in self.result.random_effects[node].items():
            pred += val * (x[name] if name in FEATURES else 1.0)
        return tf.inv([pred])


# ======================================================================
#  FL (for reproduction and parameter-movement audit)
# ======================================================================

def original_fl(X_train, y_train, graph, nodes, learning_rate, num_rounds,
                lambda_param, personalization, ridge_alpha, clip_bound):
    """Exact re-implementation of the previous algorithm (sklearn Ridge,
    alpha=5; eta*lambda consensus step; rho blend; clipping). Returns the
    final and local parameters and whether clipping ever changed a value."""
    weights, original = {}, {}
    for n in nodes:
        m = Ridge(alpha=ridge_alpha).fit(X_train[n], y_train[n])
        weights[n] = np.r_[m.coef_, m.intercept_]
        original[n] = weights[n].copy()
    clip_active = False
    for _ in range(num_rounds):
        new = {}
        for n in nodes:
            nbrs = [b for b in graph.neighbors(n) if b in weights]
            w_i = np.clip(weights[n], -clip_bound, clip_bound)
            clip_active |= bool(np.any(w_i != weights[n]))
            if not nbrs:
                new[n] = w_i
                continue
            diff, tot = np.zeros_like(w_i), 0.0
            for b in nbrs:
                ew = graph[n][b]['weight']
                diff += ew * (np.clip(weights[b], -clip_bound, clip_bound) - w_i)
                tot += ew
            cons = w_i + learning_rate * lambda_param * diff / tot
            new[n] = ((1 - personalization) * cons
                      + personalization * np.clip(original[n], -clip_bound, clip_bound))
        weights = {n: np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                   for n, v in new.items()}
    return weights, original, clip_active


# ======================================================================
# ROLLING-ORIGIN EVALUATION
# ======================================================================

class RollingOriginEvaluator:
    """
    Expanding-window, one-step-ahead evaluation. At each origin t0 every
    model, scaler, hyperparameter and collaboration graph is built from
    target years <= t0 only, and evaluated on year t0 + 1.
    """

    def __init__(self, panel, spec, graph_kinds=('similarity',), min_train=10,
                 n_val=5, min_fit=6, beta_grid=BETA_GRID, lambda_grid=LAMBDA_GRID,
                 beta_scope='network', lambda_path=None, path_graphs=('similarity',),
                 include_original=False, include_hierarchical=True,
                 rolling_window=None, verbose=True):
        self.panel, self.spec = panel, spec
        self.beta_scope = beta_scope
        self.graph_kinds = list(graph_kinds)
        self.min_train, self.n_val, self.min_fit = min_train, n_val, min_fit
        self.beta_grid, self.lambda_grid = list(beta_grid), list(lambda_grid)
        self.lambda_path, self.path_graphs = lambda_path, list(path_graphs)
        self.include_original = include_original and spec == 'levels'
        self.include_hierarchical = include_hierarchical
        self.rolling_window = rolling_window
        self.verbose = verbose
        self.nodes = panel.countries

    def origins(self):
        first = self.panel.first_target_year + self.min_train - 1
        return list(range(first, self.panel.last_year))

    def run(self):
        preds, path, diag, gstats = [], [], [], []
        full_means = self.panel.profile_means(None)
        prev_edges = {}
        for t0 in self.origins():
            if self.verbose:
                print(f"  [{self.spec}] origin {t0} -> {t0 + 1}")
            nodes = self.nodes
            train, test = {}, {}
            for n in nodes:
                d = self.panel.designs[self.spec][n]
                tr, te = d[d['year'] <= t0], d[d['year'] == t0 + 1]
                train[n] = (tr[FEATURES].values, tr['y'].values, tr['year'].values)
                test[n] = te.iloc[0]
                assert len(tr) >= self.min_train and len(te) == 1

            def add(method, n, pred_raw, store=preds, **extra):
                t = test[n]
                lv = self.panel.designs['levels'][n]
                store.append({
                    'Spec': self.spec, 'Method': method, 'Country': n,
                    'Origin': t0, 'Year': t0 + 1, 'Actual': t['E'],
                    'Predicted': float(to_twh(self.spec, np.ravel(pred_raw)[0], t['E_prev'])),
                    'Persistence': t['E_prev'],
                    'HistMean': lv[lv['year'] <= t0]['E'].mean(), **extra})

            # ---- benchmarks ----
            for n in nodes:
                lv = self.panel.designs['levels'][n]
                tr = lv[lv['year'] <= t0]
                drift = np.mean(np.log(tr['E'].values) - np.log(tr['E_prev'].values))
                t = test[n]
                for name, val in [('Persistence', t['E_prev']),
                                  ('Drift', t['E_prev'] * np.exp(drift))]:
                    preds.append({'Spec': self.spec, 'Method': name, 'Country': n,
                                  'Origin': t0, 'Year': t0 + 1, 'Actual': t['E'],
                                  'Predicted': val, 'Persistence': t['E_prev'],
                                  'HistMean': tr['E'].mean()})

            # ---- local-only, beta by inner validation (network-wide or
            #      per country, see beta_scope) ----
            pen_icpt = penalize_intercept(self.spec)
            if self.beta_scope == 'network':
                b = tune_beta_network(train, nodes, self.spec, self.beta_grid,
                                      self.n_val, self.min_fit)
                betas = {n: b for n in nodes}
            else:
                betas = {n: tune_beta_local(*train[n], self.spec, self.beta_grid,
                                            self.n_val, self.min_fit) for n in nodes}
            theta_local, tfs = {}, {}
            for n in nodes:
                Xr, yr, yrs = train[n]
                tfs[n] = LocalTransform(self.spec).fit(Xr, yr)
                theta_local[n] = solve_ridge(tfs[n].tx(Xr), tfs[n].ty(yr), betas[n],
                                             pen_icpt)
                Xte = tfs[n].tx(test[n][FEATURES].values.reshape(1, -1).astype(float))
                add('Local', n, tfs[n].inv(predict_linear(theta_local[n], Xte)),
                    Beta=betas[n])

            if self.rolling_window:
                rw = {n: tuple(a[-self.rolling_window:] for a in train[n]) for n in nodes}
                b_rw = tune_beta_network(rw, nodes, self.spec, self.beta_grid,
                                         self.n_val, self.min_fit)
                for n in nodes:
                    Xr, yr, _ = rw[n]
                    tf = LocalTransform(self.spec).fit(Xr, yr)
                    th = solve_ridge(tf.tx(Xr), tf.ty(yr), b_rw, pen_icpt)
                    Xte = tf.tx(test[n][FEATURES].values.reshape(1, -1).astype(float))
                    add(f'Local_RW{self.rolling_window}', n,
                        tf.inv(predict_linear(th, Xte)), Beta=b_rw)

            # ---- GTVMin federated learning ----
            A, c, const, _ = node_systems(train, self.spec, betas, nodes)
            Theta_loc = np.array([theta_local[n] for n in nodes])
            origin_means = self.panel.profile_means(t0)
            graphs = {}
            for kind in sorted(set(self.graph_kinds) | set(
                    self.path_graphs if self.lambda_path else [])):
                means = full_means if kind == 'similarity_fullhistory' else origin_means
                graphs[kind] = build_graph(kind, nodes, means=means, thetas=Theta_loc)

            for kind in self.graph_kinds:
                G = graphs[kind]
                W = weight_matrix(G, nodes)
                lam, _ = tune_lambda(train, self.spec, betas, nodes, W,
                                     self.lambda_grid, self.n_val, self.min_fit)
                solver = GTVMin(A, c, const, W, lam)
                Theta, info = solver.federated(Theta0=Theta_loc, tol=FL_TOL)
                exact_dev = float(np.abs(Theta - solver.exact()).max())
                edges = {frozenset(e) for e in G.edges()}
                n_edges = G.number_of_edges()
                gstats.append({
                    'Spec': self.spec, 'Graph': kind, 'Origin': t0,
                    'Edges': n_edges, 'Density': nx.density(G),
                    'FallbackEdges': sum(1 for *_, d in G.edges(data=True)
                                         if d.get('fallback')),
                    'Jaccard_prev': (len(edges & prev_edges[kind]) /
                                     len(edges | prev_edges[kind])
                                     if kind in prev_edges else np.nan),
                    'Lambda': lam, 'Iterations': info['iterations'],
                    'Converged': info['converged'], 'MaxDevFromExact': exact_dev,
                    # floats sent: each round, every edge carries theta both ways
                    'FloatsCommunicated': info['iterations'] * 2 * n_edges * A.shape[1],
                })
                prev_edges[kind] = edges
                for i, n in enumerate(nodes):
                    Xte = tfs[n].tx(test[n][FEATURES].values.reshape(1, -1).astype(float))
                    add(f'FL_{kind}', n, tfs[n].inv(predict_linear(Theta[i], Xte)),
                        Beta=betas[n], Lambda=lam)
                    dist = np.linalg.norm(Theta[i] - Theta_loc[i])
                    diag.append({'Spec': self.spec, 'Method': f'FL_{kind}',
                                 'Country': n, 'Origin': t0, 'Lambda': lam,
                                 'Degree': G.degree(n),
                                 'ParamDist': dist,
                                 'ParamDistRel': dist / max(np.linalg.norm(Theta_loc[i]), 1e-12)})

            # ---- lambda path (diagnostic only; NOT used for selection) ----
            if self.lambda_path:
                for kind in self.path_graphs:
                    W = weight_matrix(graphs[kind], nodes)
                    for lam in self.lambda_path:
                        Theta = GTVMin(A, c, const, W, lam).exact()
                        for i, n in enumerate(nodes):
                            Xte = tfs[n].tx(test[n][FEATURES].values.reshape(1, -1).astype(float))
                            dist = np.linalg.norm(Theta[i] - Theta_loc[i])
                            add(f'FLpath_{kind}', n,
                                tfs[n].inv(predict_linear(Theta[i], Xte)), store=path,
                                Lambda=lam, ParamDist=dist,
                                ParamDistRel=dist / max(np.linalg.norm(Theta_loc[i]), 1e-12))

            # ---- centralized baselines ----
            pooled_variants = [('Pooled_GlobalScale', 'global', False),
                               ('Pooled_FE', 'global', True)]
            if self.spec == 'levels':
                pooled_variants.append(('Pooled_LocalScale', 'local', False))
            for name, scale, fe in pooled_variants:
                b = tune_beta_pooled(train, nodes, self.spec, scale, fe,
                                     self.beta_grid, self.n_val, self.min_fit)
                model = PooledModel(self.spec, scale, fe).fit(train, nodes, b)
                for n in nodes:
                    add(name, n, model.predict(
                        n, test[n][FEATURES].values.reshape(1, -1).astype(float)), Beta=b)

            if self.include_hierarchical:
                hm = HierarchicalModel(self.spec).fit(train, nodes)
                for n in nodes:
                    add('Hierarchical', n, hm.predict(
                        n, test[n][FEATURES].values.reshape(1, -1).astype(float)),
                        REFormula=hm.re_formula)

            # ----fixed full-history vs per-origin graph ----
            if self.include_original:
                Xs = {n: tfs[n].tx(train[n][0]) for n in nodes}
                ys = {n: tfs[n].ty(train[n][1]) for n in nodes}
                for gname, means in [('FixedGraph', full_means),
                                     ('OriginGraph', origin_means)]:
                    G = build_graph('similarity', nodes, means=means)
                    w, w0, clip_active = original_fl(Xs, ys, G, nodes, **ORIGINAL_FL)
                    for n in nodes:
                        Xte = tfs[n].tx(test[n][FEATURES].values.reshape(1, -1).astype(float))
                        add(f'FLorig_{gname}', n, tfs[n].inv(predict_linear(w[n], Xte)))
                        dist = np.linalg.norm(w[n] - w0[n])
                        diag.append({'Spec': self.spec, 'Method': f'FLorig_{gname}',
                                     'Country': n, 'Origin': t0,
                                     'Lambda': ORIGINAL_FL['lambda_param'],
                                     'Degree': G.degree(n), 'ParamDist': dist,
                                     'ParamDistRel': dist / max(np.linalg.norm(w0[n]), 1e-12),
                                     'ClipActive': clip_active})
                for n in nodes:
                    m = Ridge(alpha=ORIGINAL_FL['ridge_alpha']).fit(Xs[n], ys[n])
                    Xte = tfs[n].tx(test[n][FEATURES].values.reshape(1, -1).astype(float))
                    add('Localorig', n, tfs[n].inv(m.predict(Xte)))

                # Earlier centralized baselines, reproduced verbatim:
                # one global scaler, sklearn Ridge(alpha=5); FE adds
                # drop-first dummies that Ridge also shrinks.
                Xp = np.vstack([train[n][0] for n in nodes])
                yp = np.concatenate([train[n][1] for n in nodes])
                gsx = StandardScaler().fit(Xp)
                gsy = StandardScaler().fit(yp.reshape(-1, 1))
                ys_g = gsy.transform(yp.reshape(-1, 1)).ravel()
                g_model = Ridge(alpha=ORIGINAL_FL['ridge_alpha']).fit(gsx.transform(Xp), ys_g)
                ids = np.concatenate([[n] * len(train[n][1]) for n in nodes])
                dummies = pd.get_dummies(ids, drop_first=True)
                fe_model = Ridge(alpha=ORIGINAL_FL['ridge_alpha']).fit(
                    np.hstack([gsx.transform(Xp), dummies.values.astype(float)]), ys_g)
                for n in nodes:
                    Xte = gsx.transform(test[n][FEATURES].values.reshape(1, -1).astype(float))
                    add('Pooledorig_GlobalScale', n,
                        gsy.inverse_transform(g_model.predict(Xte).reshape(-1, 1)).ravel())
                    drow = np.array([[1.0 if n == col else 0.0 for col in dummies.columns]])
                    add('Pooledorig_FE', n, gsy.inverse_transform(
                        fe_model.predict(np.hstack([Xte, drow])).reshape(-1, 1)).ravel())

        return {'predictions': pd.DataFrame(preds), 'lambda_path': pd.DataFrame(path),
                'fl_diagnostics': pd.DataFrame(diag), 'graph_stats': pd.DataFrame(gstats)}


# ======================================================================
# METRICS
# ======================================================================

def mase_scales(panel, last_train_year):
    """In-sample MAE of the naive (persistence) forecast over the initial
    training period, per country (Hyndman & Koehler, 2006)."""
    out = {}
    for c in panel.countries:
        d = panel.designs['levels'][c]
        d = d[d['year'] <= last_train_year]
        out[c] = np.mean(np.abs(d['E'] - d['E_prev']))
    return out


def summarize(pred_df, scales, group_cols=('Spec', 'Method')):
    """Per-country metrics on the pooled one-step-ahead forecasts."""
    rows = []
    keys = list(group_cols) + ['Country']
    for key, g in pred_df.groupby(keys):
        a, p = g['Actual'].values, g['Predicted'].values
        e, e_pers = a - p, a - g['Persistence'].values
        mse, mse_pers = np.mean(e ** 2), np.mean(e_pers ** 2)
        sst_test = np.sum((a - a.mean()) ** 2)
        sst_hist = np.sum((a - g['HistMean'].values) ** 2)
        country = key[-1]
        rows.append({**dict(zip(keys, key)), 'N': len(g),
                     'MSE': mse, 'RMSE': np.sqrt(mse), 'MAE': np.mean(np.abs(e)),
                     'MAPE': 100 * np.mean(np.abs(e / a)),
                     'MASE': np.mean(np.abs(e)) / scales[country],
                     'RelMSE_vs_Persistence': mse / mse_pers,
                     'Skill_vs_Persistence': 1 - mse / mse_pers,
                     'R2_test_mean': 1 - np.sum(e ** 2) / sst_test,
                     'R2_OOS_hist_mean': 1 - np.sum(e ** 2) / sst_hist})
    return pd.DataFrame(rows)


def aggregate(summary, group_cols=('Spec', 'Method')):
    """Network-level summary. GM_RelMSE is the geometric mean over countries
    of MSE / MSE_persistence (< 1 means better than persistence)."""
    def agg(g):
        return pd.Series({
            'Countries': len(g),
            'Mean_MSE': g['MSE'].mean(),
            'Mean_MAE': g['MAE'].mean(),
            'Mean_MAPE': g['MAPE'].mean(),
            'Mean_MASE': g['MASE'].mean(),
            'GM_RelMSE': np.exp(np.log(g['RelMSE_vs_Persistence']).mean()),
            'Mean_Skill': g['Skill_vs_Persistence'].mean(),
            'Median_Skill': g['Skill_vs_Persistence'].median(),
            'N_beat_persistence': int((g['Skill_vs_Persistence'] > 0).sum()),
            'Mean_R2_test_mean': g['R2_test_mean'].mean(),
            'Mean_R2_OOS': g['R2_OOS_hist_mean'].mean(),
        })
    return summary.groupby(list(group_cols)).apply(agg, include_groups=False).reset_index()
