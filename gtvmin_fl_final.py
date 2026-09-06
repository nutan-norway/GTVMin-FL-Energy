import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import networkx as nx
import warnings
warnings.filterwarnings('ignore')


# ======================================================================
# FEATURE DEFINITION -- leakage-free, small p
# ======================================================================
#
# We predict primary_energy_consumption at year t using ONLY information
# available at year t-1 (one-step-ahead forecasting). No same-year
# component of the target is ever used. 
#
# AR term (lag of the target itself) is the strongest legitimate
# predictor in most macro time series and is NOT leakage: it is
# information already realized before the year being predicted.

LAG_FEATURE_COLUMNS = [
    'primary_energy_consumption',   # autoregressive term (lag of the target)
]

OPTIONAL_EXOGENOUS_COLUMNS = ['population', 'gdp']

TARGET_COLUMN = 'primary_energy_consumption'


class TemporalFederatedLearningNetwork:
    """
    Leakage-free, chronologically-split, lag-based version of the FL
    network.
    """

    def __init__(self, data_path, european_only=True, graph_type='similarity',
                 min_train_years=15, test_frac=0.2, val_frac=0.1,
                 similarity_threshold=0.8, start_year=None):
        self.data_path = data_path
        self.graph_type = graph_type
        self.min_train_years = min_train_years
        self.test_frac = test_frac
        self.val_frac = val_frac
        self.similarity_threshold = similarity_threshold
        self.start_year = start_year  # e.g. 1995: restrict to a more

        self.european_countries = [
            'Austria', 'Croatia', 'Cyprus', 'Czech Republic', 'Denmark',
            'Estonia', 'Finland', 'Greece', 'Hungary', 'Iceland', 'Latvia',
            'Lithuania', 'Luxembourg', 'Norway', 'Poland',
            'Romania', 'Slovakia', 'Slovenia', 'Switzerland',
        ]
        self.geographic_groups = {
            'Nordic': ['Norway', 'Finland', 'Denmark', 'Iceland'],
            'Baltic': ['Estonia', 'Latvia', 'Lithuania', 'Poland'],
            'Central': ['Austria', 'Switzerland', 'Czech Republic', 'Slovakia', 'Hungary'],
            'Southern': ['Greece', 'Croatia', 'Slovenia', 'Cyprus'],
            'Eastern': ['Romania', 'Poland'],
            'Western': ['Luxembourg']
        }

        self._load_and_engineer(european_only)
        self._build_graph()
        self._split_chronologically()
    
    # ------------------------------------------------------------
    

    # ------------------------------------------------------------
    def _load_and_engineer(self, european_only):
        df = pd.read_csv(self.data_path)
        if european_only:
            df = df[df['country'].isin(self.european_countries)]
        if self.start_year is not None:
            df = df[df['year'] >= self.start_year]
            print(f"  [note] restricting to years >= {self.start_year} "
                  f"to reduce exposure to structural breaks (pre-1990 "
                  f"transition economies, oil shocks, etc.)")

        self.feature_columns = list(LAG_FEATURE_COLUMNS)
        for col in OPTIONAL_EXOGENOUS_COLUMNS:
            if col in df.columns:
                self.feature_columns.append(col)
            else:
                print(f"  [note] optional exogenous column '{col}' not found "
                      f"in CSV; proceeding without it.")

        needed_raw_cols = list(set(self.feature_columns + [TARGET_COLUMN]))

        self.country_data = {}
        for country in df['country'].unique():
            cdf = df[df['country'] == country].copy()
            cdf = cdf.sort_values('year')
            cdf = cdf.dropna(subset=needed_raw_cols)

            # Build lag-1 features: X_t = raw_features at year t-1,
            # y_t = target at year t. This guarantees no same-year
            # accounting-identity leakage and gives genuine one-step
            # forecasting semantics.
            lagged = cdf[['year'] + self.feature_columns].copy()
            lagged.columns = ['year'] + [f'{c}_lag1' for c in self.feature_columns]
            lagged = lagged.rename(columns={'year': 'feature_year'})
            lagged['year'] = lagged['feature_year'] + 1  # aligns to target year
            lagged = lagged.drop(columns=['feature_year'])

            merged = pd.merge(cdf[['year', TARGET_COLUMN]], lagged, on='year', how='inner')
            merged = merged.dropna()

            if len(merged) >= self.min_train_years + 3:  # enough for train+val+test
                self.country_data[country] = merged.sort_values('year').reset_index(drop=True)

        self.lag_feature_columns = [f'{c}_lag1' for c in self.feature_columns]
        self.nodes = list(self.country_data.keys())

        print(f"Data loaded for {len(self.nodes)} countries "
              f"(dropped for insufficient post-lag history: "
              f"{set(df['country'].unique()) & set(self.european_countries) - set(self.nodes)})")
        print(f"Feature set ({len(self.lag_feature_columns)} features, lag-1): "
              f"{self.lag_feature_columns}")

        # p<<n diagnostic
        p = len(self.lag_feature_columns) + 1  # + intercept
        for c in self.nodes:
            n = len(self.country_data[c])
            n_train = int(round(n * (1 - self.test_frac - self.val_frac)))
            ratio = n_train / p
            flag = "" if ratio >= 5 else " "
            print(f"  {c:12s} total years={n:3d}  train~{n_train:3d}  "
                  f"n_train/p={ratio:4.1f}{flag}")

    # ------------------------------------------------------------
    def _build_graph(self):
        self.graph = nx.Graph()
        for node in self.nodes:
            self.graph.add_node(node)
        self.node_models = {n: LinearRegression() for n in self.nodes}

        if self.graph_type == 'similarity':
            mean_matrix = np.array([
                self.country_data[c][self.lag_feature_columns].mean().values
                for c in self.nodes
            ])
            feature_scaler = StandardScaler()
            mean_matrix_scaled = feature_scaler.fit_transform(mean_matrix)

            for i, c1 in enumerate(self.nodes):
                m1 = mean_matrix_scaled[i]
                for j in range(i + 1, len(self.nodes)):
                    c2 = self.nodes[j]
                    m2 = mean_matrix_scaled[j]
                    n1, n2 = np.linalg.norm(m1), np.linalg.norm(m2)
                    if n1 > 0 and n2 > 0:
                        sim = np.dot(m1 / n1, m2 / n2)
                        if sim > self.similarity_threshold:
                            self.graph.add_edge(c1, c2, weight=sim)
            self._ensure_connectivity()
        elif self.graph_type == 'complete_equal':
            for i, c1 in enumerate(self.nodes):
                for c2 in self.nodes[i+1:]:
                    self.graph.add_edge(c1, c2, weight=1.0)
        elif self.graph_type == 'complete':
            # Complete graph, similarity-weighted (standardized, as in
            # 'similarity', but with NO threshold -- every pair connected).
            mean_matrix = np.array([
                self.country_data[c][self.lag_feature_columns].mean().values
                for c in self.nodes
            ])
            mean_matrix_scaled = StandardScaler().fit_transform(mean_matrix)
            for i, c1 in enumerate(self.nodes):
                m1 = mean_matrix_scaled[i]
                for j in range(i + 1, len(self.nodes)):
                    c2 = self.nodes[j]
                    m2 = mean_matrix_scaled[j]
                    n1, n2 = np.linalg.norm(m1), np.linalg.norm(m2)
                    if n1 > 0 and n2 > 0:
                        sim = np.dot(m1 / n1, m2 / n2)
                        self.graph.add_edge(c1, c2, weight=max(sim, 0.1))
        elif self.graph_type == 'geographic':
            for group_name, countries in self.geographic_groups.items():
                group_nodes = [c for c in countries if c in self.nodes]
                for i, c1 in enumerate(group_nodes):
                    for c2 in group_nodes[i+1:]:
                        self.graph.add_edge(c1, c2, weight=0.9)
            cross_connections = [
                ('Nordic', 'Baltic', 0.7), ('Central', 'Southern', 0.7),
                ('Central', 'Eastern', 0.6), ('Eastern', 'Baltic', 0.6)
            ]
            for g1, g2, w in cross_connections:
                n1 = [c for c in self.geographic_groups[g1] if c in self.nodes]
                n2 = [c for c in self.geographic_groups[g2] if c in self.nodes]
                for c1 in n1:
                    for c2 in n2:
                        if not self.graph.has_edge(c1, c2):
                            self.graph.add_edge(c1, c2, weight=w)
            self._ensure_connectivity()

        print(f"Graph ({self.graph_type}): {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")

    def _ensure_connectivity(self):
        if not nx.is_connected(self.graph):
            comps = list(nx.connected_components(self.graph))
            for i in range(len(comps) - 1):
                a, b = list(comps[i])[0], list(comps[i+1])[0]
                self.graph.add_edge(a, b, weight=0.5)

    # ------------------------------------------------------------
    def _split_chronologically(self):
        """
        Chronological (NOT random) split: earliest years -> train,
        middle years -> validation, most recent years -> test.
        This is required for a genuine forecasting evaluation and
        avoids leaking future information into training.
        """
        self.train_data, self.val_data, self.test_data, self.scalers = {}, {}, {}, {}

        for country in self.nodes:
            d = self.country_data[country]
            n = len(d)
            n_test = max(2, int(round(n * self.test_frac)))
            n_val = max(2, int(round(n * self.val_frac)))
            n_train = n - n_test - n_val
            if n_train < len(self.lag_feature_columns) + 2:
                print(f"  [warn] {country}: only {n_train} train rows after chronological "
                      f"split; consider lowering min_train_years or feature count.")

            train_df = d.iloc[:n_train]
            val_df = d.iloc[n_train:n_train+n_val]
            test_df = d.iloc[n_train+n_val:]

            X_train_raw = train_df[self.lag_feature_columns].values
            y_train_raw = train_df[TARGET_COLUMN].values
            X_val_raw = val_df[self.lag_feature_columns].values
            y_val_raw = val_df[TARGET_COLUMN].values
            X_test_raw = test_df[self.lag_feature_columns].values
            y_test_raw = test_df[TARGET_COLUMN].values

            scaler_X, scaler_y = StandardScaler(), StandardScaler()
            X_train = scaler_X.fit_transform(X_train_raw)
            y_train = scaler_y.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
            X_val = scaler_X.transform(X_val_raw) if len(X_val_raw) else X_val_raw
            y_val = (scaler_y.transform(y_val_raw.reshape(-1, 1)).flatten()
                     if len(y_val_raw) else y_val_raw)
            X_test = scaler_X.transform(X_test_raw)
            y_test = scaler_y.transform(y_test_raw.reshape(-1, 1)).flatten()

            self.scalers[country] = {'X': scaler_X, 'y': scaler_y}
            self.train_data[country] = (X_train, y_train)
            self.val_data[country] = (X_val, y_val)
            self.test_data[country] = (X_test, y_test)

        # Keep raw (unscaled) pooled data available for the fair
        # centralized baselines below.
        self.raw_pooled = {
            'train': {c: (self.country_data[c][self.lag_feature_columns].values[:len(self.train_data[c][0])],
                           self.country_data[c][TARGET_COLUMN].values[:len(self.train_data[c][0])])
                      for c in self.nodes},
        }


# ======================================================================
# FAIR CENTRALIZED BASELINES
# ======================================================================

class FairCentralizedBaselines:
    """
    Two centralized baselines that avoid confounding "pooling is harmful"
    with "per-country standardization before pooling is harmful":

      (a) global_standardization: ONE StandardScaler fit on the pooled
          RAW training data across all countries (not per-country
          scalers), then a single LinearRegression.
      (b) fixed_effects: same global scaling for the continuous
          features, PLUS one-hot country dummy variables, so the model
          shares slopes across countries but learns a country-specific
          intercept. Still a single, centralized model (no per-node
          personalization), but a substantially stronger and fairer
          comparator than naive pooling with mismatched local scalers.
    """

    def __init__(self, network: TemporalFederatedLearningNetwork):
        self.network = network
        self.countries = network.nodes
        self.feat_cols = network.lag_feature_columns

    def _pooled_raw_train(self):
        Xs, ys, country_ids = [], [], []
        for c in self.countries:
            X, y = self.network.raw_pooled['train'][c]
            Xs.append(X)
            ys.append(y)
            country_ids += [c] * len(y)
        return np.vstack(Xs), np.concatenate(ys), np.array(country_ids)

    def train_global_standardization(self):
        X_raw, y_raw, _ = self._pooled_raw_train()
        self.scaler_X_global = StandardScaler().fit(X_raw)
        self.scaler_y_global = StandardScaler().fit(y_raw.reshape(-1, 1))
        X_scaled = self.scaler_X_global.transform(X_raw)
        y_scaled = self.scaler_y_global.transform(y_raw.reshape(-1, 1)).flatten()
        self.global_model = LinearRegression().fit(X_scaled, y_scaled)

    def train_fixed_effects(self):
        X_raw, y_raw, country_ids = self._pooled_raw_train()
        self.fe_scaler_X = StandardScaler().fit(X_raw)
        self.fe_scaler_y = StandardScaler().fit(y_raw.reshape(-1, 1))
        X_scaled = self.fe_scaler_X.transform(X_raw)
        y_scaled = self.fe_scaler_y.transform(y_raw.reshape(-1, 1)).flatten()

        dummies = pd.get_dummies(country_ids, drop_first=True).values.astype(float)
        self.fe_dummy_categories = sorted(set(country_ids))[1:]  # matches drop_first order approx.
        X_fe = np.hstack([X_scaled, dummies])
        self.fe_model = LinearRegression().fit(X_fe, y_scaled)
        self._fe_dummy_frame_cols = pd.get_dummies(country_ids, drop_first=True).columns.tolist()

    def evaluate(self, which='global'):
        results = {}
        for country in self.countries:
            X_test_raw = (self.network.country_data[country][self.feat_cols]
                          .values[-len(self.network.test_data[country][0]):])
            y_test_raw = (self.network.country_data[country][TARGET_COLUMN]
                          .values[-len(self.network.test_data[country][0]):])

            if which == 'global':
                Xs = self.scaler_X_global.transform(X_test_raw)
                y_pred_s = self.global_model.predict(Xs)
                y_pred = self.scaler_y_global.inverse_transform(
                    y_pred_s.reshape(-1, 1)).flatten()
            elif which == 'fixed_effects':
                Xs = self.fe_scaler_X.transform(X_test_raw)
                dummy_row = pd.DataFrame(
                    [[1.0 if country == col else 0.0 for col in self._fe_dummy_frame_cols]],
                    columns=self._fe_dummy_frame_cols)
                dummy_block = np.repeat(dummy_row.values, len(Xs), axis=0)
                X_fe_test = np.hstack([Xs, dummy_block])
                y_pred_s = self.fe_model.predict(X_fe_test)
                y_pred = self.fe_scaler_y.inverse_transform(
                    y_pred_s.reshape(-1, 1)).flatten()
            else:
                raise ValueError(which)

            mse = mean_squared_error(y_test_raw, y_pred)
            mae = mean_absolute_error(y_test_raw, y_pred)
            try:
                r2 = r2_score(y_test_raw, y_pred)
                r2 = max(min(r2, 1.0), -1e6)  # cap raised for diagnosis; was -10.0
            except Exception:
                r2 = -1.0
            results[country] = {'mse': mse, 'r2': r2, 'mae': mae}
        return results


# ======================================================================
# STANDARD LOCAL-ONLY AND FEDERATED (GTVMin) TRAINING
# ======================================================================

class CentralizedBaseline:
    """Local-only (no collaboration) and naive-global baseline."""

    def __init__(self, fl_network, ridge_alpha=5.0):
        self.network = fl_network
        self.ridge_alpha = ridge_alpha
        self.global_model = Ridge(alpha=ridge_alpha)
        self.node_models = {}

    def train_local_only(self):
        for node in self.network.nodes:
            X_train, y_train = self.network.train_data[node]
            model = Ridge(alpha=self.ridge_alpha)
            model.fit(X_train, y_train)
            self.node_models[node] = model

    def evaluate(self):
        results = {}
        for node in self.network.nodes:
            model = self.node_models[node]
            X_test, y_test = self.network.test_data[node]
            y_pred = model.predict(X_test)

            scaler_y = self.network.scalers[node]['y']
            y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
            y_pred_original = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

            mse = mean_squared_error(y_test_original, y_pred_original)
            mae = mean_absolute_error(y_test_original, y_pred_original)
            try:
                r2 = r2_score(y_test_original, y_pred_original)
                r2 = max(min(r2, 1.0), -1e6)  # cap raised for diagnosis; was -10.0
            except Exception:
                r2 = -1.0
            results[node] = {'mse': mse, 'r2': r2, 'mae': mae}
        return results


# ======================================================================
# DIAGNOSTIC: raw predicted-vs-actual for a specific country
# ======================================================================

def diagnose_country(network, model_dict, country, label=""):
    """model_dict: e.g. local_baseline.node_models or net.node_models (post-FL)"""
    model = model_dict[country]
    X_test, y_test = network.test_data[country]
    y_pred = model.predict(X_test)

    scaler_y = network.scalers[country]['y']
    y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_orig = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    n_test = len(y_test)
    test_years = network.country_data[country]['year'].values[-n_test:]

    print(f"\n--- {label} raw predicted vs. actual: {country} ---")
    for yr, actual, pred in zip(test_years, y_test_orig, y_pred_orig):
        print(f"  {yr}: actual={actual:12.2f}  predicted={pred:14.2f}  "
              f"error={pred-actual:14.2f}")

    # Also show the raw (unlagged) gdp/population/target history for this
    # country to check for a structural break / scale discontinuity.
    raw = network.country_data[country][
        ['year', 'primary_energy_consumption_lag1'] +
        ([c for c in ['gdp_lag1', 'population_lag1'] if c in network.lag_feature_columns])
    ]
    print(f"\n--- {label} full lagged history: {country} "
          f"(check for scale jumps/structural breaks) ---")
    print(raw.to_string(index=False))


class EnhancedGTVMinFL:
    """GTVMin-based personalized federated learning, with Ridge as the
    local learner to control coefficient instability from collinear, 
    trending lag features under low n/p, and a tighter weight-clipping 
    bound appropriate to the now much smaller expected coefficient scale.

    IMPORTANT: weights are fit ONCE per node to initialize (round 0),
    then PERSIST across all subsequent communication rounds. Each round
    only blends the currently held weights with neighbors' currently
    held weights -- nodes are never re-fit from raw data again after
    initialization.
    """

    def __init__(self, fl_network, variation_measure='l2', learning_rate=0.005,
                 num_rounds=30, lambda_param=0.05, personalization=0.3,
                 ridge_alpha=5.0, clip_bound=10.0):
        self.network = fl_network
        self.variation_measure = variation_measure
        self.learning_rate = learning_rate
        self.num_rounds = num_rounds
        self.lambda_param = lambda_param
        self.personalization = personalization
        self.ridge_alpha = ridge_alpha
        self.clip_bound = clip_bound
        self.original_weights = {}

    def train(self):
        node_weights = {}

        # ---- ONE-TIME local initialization (not repeated per round) ----
        for node in self.network.nodes:
            self.network.node_models[node] = Ridge(alpha=self.ridge_alpha)
            X_train, y_train = self.network.train_data[node]
            model = self.network.node_models[node]
            model.fit(X_train, y_train)
            node_weights[node] = np.concatenate([model.coef_, [model.intercept_]])
            self.original_weights[node] = node_weights[node].copy()

        # ---- iterative consensus rounds: weights persist, no re-fit ----
        for round_num in range(self.num_rounds):
            new_weights = {}
            for node in self.network.nodes:
                neighbors = list(self.network.graph.neighbors(node))
                if neighbors:
                    w_i = np.clip(node_weights[node], -self.clip_bound, self.clip_bound)
                    weighted_diff = np.zeros_like(w_i)
                    total_weight = 0
                    for neighbor in neighbors:
                        w_j = np.clip(node_weights[neighbor], -self.clip_bound, self.clip_bound)
                        edge_weight = self.network.graph[node][neighbor]['weight']
                        total_weight += edge_weight
                        weighted_diff += edge_weight * (w_j - w_i)  # L2
                    if total_weight > 0:
                        weighted_diff /= total_weight

                    consensus_weights = w_i + self.learning_rate * self.lambda_param * weighted_diff
                    original_w = np.clip(self.original_weights[node], -self.clip_bound, self.clip_bound)
                    personalized_weights = ((1 - self.personalization) * consensus_weights
                                             + self.personalization * original_w)
                    new_weights[node] = personalized_weights
                else:
                    new_weights[node] = np.clip(node_weights[node], -self.clip_bound, self.clip_bound)

            for node in self.network.nodes:
                node_weights[node] = np.nan_to_num(new_weights[node], nan=0.0,
                                                     posinf=0.0, neginf=0.0)

        # ---- write final, post-consensus weights into models for evaluate() ----
        for node in self.network.nodes:
            model = self.network.node_models[node]
            w = node_weights[node]
            model.coef_ = w[:-1]
            model.intercept_ = w[-1]

    def evaluate(self):
        results = {}
        for node in self.network.nodes:
            model = self.network.node_models[node]
            X_test, y_test = self.network.test_data[node]
            y_pred = model.predict(X_test)

            scaler_y = self.network.scalers[node]['y']
            y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
            y_pred_original = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

            mse = mean_squared_error(y_test_original, y_pred_original)
            mae = mean_absolute_error(y_test_original, y_pred_original)
            try:
                r2 = r2_score(y_test_original, y_pred_original)
                r2 = max(min(r2, 1.0), -1e6)  # cap raised for diagnosis; was -10.0
            except Exception:
                r2 = -1.0
            results[node] = {'mse': mse, 'r2': r2, 'mae': mae}
        return results


# ======================================================================
# ROLLING-ORIGIN (EXPANDING WINDOW) EVALUATION
#
# Standard practice in applied forecasting: instead of one arbitrary
# train/test split (dominated by whichever years happen
# to fall in that window, including COVID), train on all years up to
# an "origin" year, predict the single next year, then slide the origin
# forward one year at a time and repeat. Every available year is
# evaluated exactly once, as a genuine one-step-ahead prediction, with
# no discretion about which years are included – avoiding both the
# small-test-set instability AND any appearance of cherry-picking a
# favorable window.
# ======================================================================

class RollingOriginEvaluator:
    def __init__(self, net: TemporalFederatedLearningNetwork, min_train=10,
                 ridge_alpha=5.0, fl_learning_rate=0.005, fl_num_rounds=30,
                 fl_lambda=0.05, fl_personalization=0.3, clip_bound=10.0):
        self.net = net
        self.min_train = min_train
        self.ridge_alpha = ridge_alpha
        self.fl_learning_rate = fl_learning_rate
        self.fl_num_rounds = fl_num_rounds
        self.fl_lambda = fl_lambda
        self.fl_personalization = fl_personalization
        self.clip_bound = clip_bound
        self.feat_cols = net.lag_feature_columns

    def _origins(self):
        all_years = pd.concat([self.net.country_data[c]['year']
                                for c in self.net.nodes])
        first_year, last_year = int(all_years.min()), int(all_years.max())
        first_origin = first_year + self.min_train - 1
        return list(range(first_origin, last_year))  # predict origin+1 each time

    def _country_slice(self, country, origin):
        d = self.net.country_data[country]
        train = d[d['year'] <= origin]
        test = d[d['year'] == origin + 1]
        return train, test

    def run(self):
        records = []
        for origin in self._origins():
            # ---- gather this origin's train/test data for every country ----
            train_raw, test_raw = {}, {}
            for c in self.net.nodes:
                tr, te = self._country_slice(c, origin)
                if len(tr) >= self.min_train and len(te) == 1:
                    train_raw[c] = tr
                    test_raw[c] = te
            active = list(train_raw.keys())
            if len(active) < 2:
                continue

            # ---- per-country scalers, fit on THIS origin's train only ----
            scalers, X_train, y_train, X_test, y_test_actual = {}, {}, {}, {}, {}
            for c in active:
                sx, sy = StandardScaler(), StandardScaler()
                Xtr = sx.fit_transform(train_raw[c][self.feat_cols].values)
                ytr = sy.fit_transform(
                    train_raw[c][[TARGET_COLUMN]].values).flatten()
                Xte = sx.transform(test_raw[c][self.feat_cols].values)
                scalers[c] = {'X': sx, 'y': sy}
                X_train[c], y_train[c] = Xtr, ytr
                X_test[c] = Xte
                y_test_actual[c] = test_raw[c][TARGET_COLUMN].values[0]

            # ---- Local_Only (Ridge) ----
            local_models = {}
            for c in active:
                m = Ridge(alpha=self.ridge_alpha).fit(X_train[c], y_train[c])
                local_models[c] = m
                pred_s = m.predict(X_test[c])
                pred = scalers[c]['y'].inverse_transform(
                    pred_s.reshape(-1, 1)).flatten()[0]
                records.append({'Method': 'Local_Only', 'Country': c,
                                 'Origin': origin, 'Actual': y_test_actual[c],
                                 'Predicted': pred})

            # ---- FL_Similarity (Ridge + GTVMin consensus, fixed graph) ----
            # IMPORTANT: weights are fit ONCE per node to initialize, then
            # PERSIST across rounds. Each round only blends the currently
            # held weights with neighbors' currently held weights -- it
            # never re-derives from raw data again. Re-fitting every round
            # (as an earlier version of this code did) discards the
            # consensus update immediately, since Ridge is a closed-form
            # solver that jumps straight back to the local optimum,
            # silently reducing "federated" training to local-only.
            fl_models = {c: Ridge(alpha=self.ridge_alpha) for c in active}
            node_weights, original_weights = {}, {}
            for c in active:
                fl_models[c].fit(X_train[c], y_train[c])  # ONE-TIME initialization
                node_weights[c] = np.concatenate(
                    [fl_models[c].coef_, [fl_models[c].intercept_]])
                original_weights[c] = node_weights[c].copy()

            for _ in range(self.fl_num_rounds):
                new_weights = {}
                for c in active:
                    neighbors = [n for n in self.net.graph.neighbors(c) if n in active]
                    if neighbors:
                        w_i = np.clip(node_weights[c], -self.clip_bound, self.clip_bound)
                        diff = np.zeros_like(w_i)
                        total_w = 0
                        for nb in neighbors:
                            w_j = np.clip(node_weights[nb], -self.clip_bound, self.clip_bound)
                            ew = self.net.graph[c][nb]['weight']
                            total_w += ew
                            diff += ew * (w_j - w_i)
                        if total_w > 0:
                            diff /= total_w
                        consensus = w_i + self.fl_learning_rate * self.fl_lambda * diff
                        orig = np.clip(original_weights[c], -self.clip_bound, self.clip_bound)
                        new_weights[c] = ((1 - self.fl_personalization) * consensus
                                           + self.fl_personalization * orig)
                    else:
                        new_weights[c] = np.clip(node_weights[c], -self.clip_bound, self.clip_bound)
                # persist -- NO re-fit here. node_weights carries forward
                # into the next round's consensus step, enabling genuine
                # multi-round / multi-hop diffusion across the graph.
                for c in active:
                    node_weights[c] = np.nan_to_num(new_weights[c], nan=0.0,
                                                      posinf=0.0, neginf=0.0)

            for c in active:
                w = node_weights[c]
                fl_models[c].coef_ = w[:-1]
                fl_models[c].intercept_ = w[-1]

            for c in active:
                pred_s = fl_models[c].predict(X_test[c])
                pred = scalers[c]['y'].inverse_transform(
                    pred_s.reshape(-1, 1)).flatten()[0]
                records.append({'Method': 'FL_Similarity', 'Country': c,
                                 'Origin': origin, 'Actual': y_test_actual[c],
                                 'Predicted': pred})

            # ---- Centralized_GlobalScale (Ridge, single pooled scaler) ----
            X_pool_raw = np.vstack([train_raw[c][self.feat_cols].values for c in active])
            y_pool_raw = np.concatenate([train_raw[c][TARGET_COLUMN].values for c in active])
            gsx = StandardScaler().fit(X_pool_raw)
            gsy = StandardScaler().fit(y_pool_raw.reshape(-1, 1))
            g_model = Ridge(alpha=self.ridge_alpha).fit(
                gsx.transform(X_pool_raw),
                gsy.transform(y_pool_raw.reshape(-1, 1)).flatten())
            for c in active:
                Xte_g = gsx.transform(test_raw[c][self.feat_cols].values)
                pred = gsy.inverse_transform(
                    g_model.predict(Xte_g).reshape(-1, 1)).flatten()[0]
                records.append({'Method': 'Centralized_GlobalScale', 'Country': c,
                                 'Origin': origin, 'Actual': y_test_actual[c],
                                 'Predicted': pred})

            # ---- Centralized_FixedEffects (Ridge, global scale + dummies) ----
            country_ids = np.concatenate(
                [[c] * len(train_raw[c]) for c in active])
            dummies_train = pd.get_dummies(country_ids, drop_first=True)
            X_fe_train = np.hstack([gsx.transform(X_pool_raw), dummies_train.values.astype(float)])
            fe_model = Ridge(alpha=self.ridge_alpha).fit(
                X_fe_train, gsy.transform(y_pool_raw.reshape(-1, 1)).flatten())
            dummy_cols = dummies_train.columns.tolist()
            for c in active:
                Xte_g = gsx.transform(test_raw[c][self.feat_cols].values)
                dummy_row = np.array([[1.0 if c == col else 0.0 for col in dummy_cols]])
                Xte_fe = np.hstack([Xte_g, dummy_row])
                pred = gsy.inverse_transform(
                    fe_model.predict(Xte_fe).reshape(-1, 1)).flatten()[0]
                records.append({'Method': 'Centralized_FixedEffects', 'Country': c,
                                 'Origin': origin, 'Actual': y_test_actual[c],
                                 'Predicted': pred})

            # ---- Naive persistence ----
            for c in active:
                pred = test_raw[c]['primary_energy_consumption_lag1'].values[0]
                records.append({'Method': 'Naive_Persistence', 'Country': c,
                                 'Origin': origin, 'Actual': y_test_actual[c],
                                 'Predicted': pred})

        return pd.DataFrame(records)

    def summarize(self, results_df):
        rows = []
        for (method, country), g in results_df.groupby(['Method', 'Country']):
            mse = mean_squared_error(g['Actual'], g['Predicted'])
            mae = mean_absolute_error(g['Actual'], g['Predicted'])
            try:
                r2 = r2_score(g['Actual'], g['Predicted'])
                r2 = max(min(r2, 1.0), -1e6)
            except Exception:
                r2 = -1.0
            rows.append({'Method': method, 'Country': country, 'N_origins': len(g),
                         'MSE': mse, 'R2': r2, 'MAE': mae})
        return pd.DataFrame(rows)


# ======================================================================
# Full comparison in one run
# ======================================================================
if __name__ == "__main__":
    data_path = "owid-energy-data.csv"

    print("=== Building leakage-free, chronologically-split network ===")
    net = TemporalFederatedLearningNetwork(data_path, graph_type='similarity',
                                            min_train_years=15, start_year=1995)
    print("=== All edges with weight < 0.8 (candidate fallback edges) ===")
    for u, v, data in net.graph.edges(data=True):
        if data['weight'] < 0.8:
            print(f"  {u} -- {v}: weight={data['weight']:.4f}")

    print("\n=== Edges specifically touching the low-degree periphery nodes ===")
    periphery = ['Hungary', 'Norway', 'Switzerland', 'Finland', 'Denmark',
                 'Austria', 'Romania', 'Greece', 'Poland']
    for u, v, data in net.graph.edges(data=True):
        if u in periphery or v in periphery:
            print(f"  {u} -- {v}: weight={data['weight']:.4f}")

    def plot_network_structure(net, save_path="network_structure1.png", seed=42):
        G = net.graph
        n = G.number_of_nodes()

        # weight=None: layout treats all edges as equal strength, avoiding
        # the tightly-connected subgroup collapsing into an unreadable clump.
        # Larger k and more iterations give nodes more room to spread out.
        pos = nx.spring_layout(G, seed=seed, weight=None,
                                k=5.0 / np.sqrt(n), iterations=500)

        fig, ax = plt.subplots(figsize=(14, 12))

        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        max_w = max(edge_weights) if edge_weights else 1.0
        edge_widths = [0.5 + 2.0 * (w / max_w) for w in edge_weights]

        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.3,
                                edge_color='gray', ax=ax)
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=800,
                                edgecolors='black', linewidths=1.0, ax=ax)
        nx.draw_networkx_labels(
            G, pos, font_size=10, font_weight='bold', ax=ax,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=0.5)
        )

        ax.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved network figure to {save_path}")
        print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}, "
              f"Density: {nx.density(G):.3f}")
        degrees = dict(G.degree())
        print("Degree by country (sorted):")
        for c, d in sorted(degrees.items(), key=lambda x: -x[1]):
            print(f"  {c:12s} degree={d}")

    plot_network_structure(net)
                                            
    print("\n=== Fair centralized baselines ===")
    fair = FairCentralizedBaselines(net)

    fair.train_global_standardization()
    global_results = fair.evaluate('global')

    fair.train_fixed_effects()
    fe_results = fair.evaluate('fixed_effects')

    print("\n=== Local-only baseline (no collaboration) ===")
    
    local_baseline = CentralizedBaseline(net)
    local_baseline.train_local_only()
    local_results = local_baseline.evaluate()

    print("\n=== Federated learning (GTVMin, similarity graph) ===")
    fl_model = EnhancedGTVMinFL(net, learning_rate=0.005, num_rounds=30,
                                 lambda_param=0.05, personalization=0.3)
    fl_model.train()
    fl_results = fl_model.evaluate()

    # ---- Combine into one comparison table ----
    rows = []
    for method_name, results in [
        ('Centralized_GlobalScale', global_results),
        ('Centralized_FixedEffects', fe_results),
        ('Local_Only', local_results),
        ('FL_Similarity', fl_results),
    ]:
        for country, m in results.items():
            rows.append({'Method': method_name, 'Country': country,
                         'MSE': m['mse'], 'R2': m['r2'], 'MAE': m['mae']})

    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv("baseline_comparison.csv", index=False)

    avg_metrics = comparison_df.groupby('Method')[['MSE', 'R2', 'MAE']].mean()
    print("\n=== Average performance across all four methods ===")
    print(avg_metrics)

    print("\nSaved per-country results to baseline_comparison.csv")

    # ------------------------------------------------------------
    # Quick collinearity diagnostic on the 6 lag features, to confirm
    # (or rule out) the multicollinearity-among-trending-regressors
    # hypothesis
    # ------------------------------------------------------------
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        pooled_X = np.vstack([net.train_data[c][0] for c in net.nodes])
        vif_data = pd.DataFrame({
            'Feature': net.lag_feature_columns,
            'VIF': [variance_inflation_factor(pooled_X, i)
                    for i in range(pooled_X.shape[1])]
        }).sort_values('VIF', ascending=False)
        print("\n=== VIF on pooled (standardized) lag features ===")
        print(vif_data.to_string(index=False))
        print("(VIF > 10 indicates high collinearity; > 5 is moderate)")
    except ImportError:
        print("\n[statsmodels not installed; skipping VIF check -- "
              "pip install statsmodels to run it]")

    # ------------------------------------------------------------
    # Diagnostic 1: does the test window include the COVID-19 shock
    # (2020-2021), which no historical linear trend could anticipate?
    # ------------------------------------------------------------
    print("\n=== Test-set year ranges per country ===")
    for c in net.nodes:
        n_test = len(net.test_data[c][0])
        test_years = net.country_data[c]['year'].values[-n_test:]
        covid_flag = "  <-- includes 2020/2021" if any(y in (2020, 2021) for y in test_years) else ""
        print(f"  {c:12s} test years: {test_years.min()}-{test_years.max()} "
              f"({n_test} pts){covid_flag}")

    # ------------------------------------------------------------
    # Diagnostic 2: per-country breakdown for Local_Only and FL, sorted
    # by MSE descending
    # ------------------------------------------------------------
    print("\n=== Local_Only per-country results, sorted by MSE (desc) ===")
    local_df = comparison_df[comparison_df['Method'] == 'Local_Only'].sort_values(
        'MSE', ascending=False)
    print(local_df.to_string(index=False))

    print("\n=== FL_Similarity per-country results, sorted by MSE (desc) ===")
    fl_df = comparison_df[comparison_df['Method'] == 'FL_Similarity'].sort_values(
        'MSE', ascending=False)
    print(fl_df.to_string(index=False))

    # ------------------------------------------------------------
    # Diagnostic 3: inspect Poland directly
    # ------------------------------------------------------------
    diagnose_country(net, local_baseline.node_models, 'Poland', label='Local_Only')
    diagnose_country(net, local_baseline.node_models, 'Lithuania', label='Local_Only')

    # ------------------------------------------------------------
    # Naive persistence baseline: predict y_t = y_{t-1} (the AR lag
    # term itself, with implicit coefficient 1 and no other features).
    # Standard benchmark in applied forecasting; tests whether the
    # fitted models add any value over "assume no change."
    # ------------------------------------------------------------
    print("\n=== Naive persistence baseline (y_t = y_{t-1}) ===")
    persistence_results = {}
    for country in net.nodes:
        n_test = len(net.test_data[country][0])
        y_test_actual = net.country_data[country][TARGET_COLUMN].values[-n_test:]
        y_pred_persistence = net.country_data[country]['primary_energy_consumption_lag1'].values[-n_test:]
        mse = mean_squared_error(y_test_actual, y_pred_persistence)
        mae = mean_absolute_error(y_test_actual, y_pred_persistence)
        try:
            r2 = r2_score(y_test_actual, y_pred_persistence)
            r2 = max(min(r2, 1.0), -1e6)
        except Exception:
            r2 = -1.0
        persistence_results[country] = {'mse': mse, 'r2': r2, 'mae': mae}

    persistence_df = pd.DataFrame([
        {'Method': 'Naive_Persistence', 'Country': c, 'MSE': v['mse'],
         'R2': v['r2'], 'MAE': v['mae']}
        for c, v in persistence_results.items()
    ])
    avg_persistence = persistence_df[['MSE', 'R2', 'MAE']].mean()
    print(persistence_df.sort_values('MSE', ascending=False).to_string(index=False))
    print(f"\nAverage: MSE={avg_persistence['MSE']:.4f}, R2={avg_persistence['R2']:.4f}, "
          f"MAE={avg_persistence['MAE']:.4f}")
    

    # ==================================================================
    # ROLLING-ORIGIN (EXPANDING WINDOW) EVALUATION
    # ==================================================================
    print("\n" + "=" * 70)
    print("ROLLING-ORIGIN EVALUATION")
    print("=" * 70)
    evaluator = RollingOriginEvaluator(net, min_train=10, ridge_alpha=5.0,
                                        fl_learning_rate=0.005, fl_num_rounds=30,
                                        fl_lambda=0.05, fl_personalization=0.3)
    rolling_results_df = evaluator.run()
    rolling_results_df.to_csv("rolling_origin_predictions.csv", index=False)

    summary = evaluator.summarize(rolling_results_df)
    summary.to_csv("rolling_origin_summary.csv", index=False)
    print("\n=== Rolling-origin summary (aggregated over all origins) ===")
    print(summary.to_string(index=False))

    overall = summary.groupby('Method')[['MSE', 'R2', 'MAE']].mean()
    print("\n=== Rolling-origin OVERALL average across countries ===")
    print(overall)
    print("\nSaved per-prediction detail to rolling_origin_predictions.csv")

    # ==================================================================
    # TABLE 3: parameter sensitivity, rolling-origin
    # ==================================================================
    print("\n" + "=" * 70)
    print("TABLE 3: PARAMETER SENSITIVITY, ROLLING-ORIGIN")
    print("=" * 70)

    def run_rolling_avg(net_obj, **fl_kwargs):
        ev = RollingOriginEvaluator(net_obj, min_train=10, ridge_alpha=5.0,
                                     **fl_kwargs)
        df = ev.run()
        fl_only = df[df['Method'] == 'FL_Similarity']
        summ = ev.summarize(fl_only)
        return summ[['MSE', 'R2', 'MAE']].mean()

    lambda_values = [0.01, 0.05, 0.5, 1.0]
    table3_rows = []
    for lam in lambda_values:
        avg = run_rolling_avg(net, fl_learning_rate=0.005, fl_num_rounds=30,
                               fl_lambda=lam, fl_personalization=0.3)
        table3_rows.append({'Parameter': 'lambda', 'Value': lam,
                            'Avg_MSE': avg['MSE'], 'Avg_R2': avg['R2']})
        print(f"  lambda={lam}: MSE={avg['MSE']:.2f}, R2={avg['R2']:.4f}")

    rho_values = [0.0, 0.3, 0.5, 0.7, 1.0]
    for rho in rho_values:
        avg = run_rolling_avg(net, fl_learning_rate=0.005, fl_num_rounds=30,
                               fl_lambda=0.05, fl_personalization=rho)
        table3_rows.append({'Parameter': 'rho', 'Value': rho,
                            'Avg_MSE': avg['MSE'], 'Avg_R2': avg['R2']})
        print(f"  rho={rho}: MSE={avg['MSE']:.2f}, R2={avg['R2']:.4f}")

    table3_df = pd.DataFrame(table3_rows)
    table3_df.to_csv("table3_parameter_sensitivity_rolling.csv", index=False)
    print("\nSaved to table3_parameter_sensitivity_rolling.csv")
    

    # ==================================================================
    # TABLE 4: graph structure comparison, rolling-origin
    # ==================================================================
    print("\n" + "=" * 70)
    print("TABLE 4 (regenerated): GRAPH STRUCTURE COMPARISON, ROLLING-ORIGIN")
    print("=" * 70)

    graph_types = ['complete_equal', 'complete', 'similarity', 'geographic']
    table4_rows = []
    for gtype in graph_types:
        net_g = TemporalFederatedLearningNetwork(
            data_path, graph_type=gtype, min_train_years=15, start_year=1995)
        avg = run_rolling_avg(net_g, fl_learning_rate=0.005, fl_num_rounds=30,
                               fl_lambda=0.05, fl_personalization=0.3)
        table4_rows.append({
            'Graph_Type': gtype,
            'Edges': net_g.graph.number_of_edges(),
            'Density': (2 * net_g.graph.number_of_edges()
                        / (len(net_g.nodes) * (len(net_g.nodes) - 1))),
            'Avg_MSE': avg['MSE'], 'Avg_R2': avg['R2'],
        })
        print(f"  {gtype}: edges={table4_rows[-1]['Edges']}, "
              f"MSE={avg['MSE']:.2f}, R2={avg['R2']:.4f}")

    table4_df = pd.DataFrame(table4_rows)
    table4_df.to_csv("table4_graph_comparison_rolling.csv", index=False)
    print("\nSaved to table4_graph_comparison_rolling.csv")