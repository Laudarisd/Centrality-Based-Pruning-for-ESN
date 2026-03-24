# -*- coding: utf-8 -*-
"""
Core ESN pruning logic and model definitions.

This module supports 10 pruning models:

1) indeg
   C_in(i) = |I_i| = sum of positive incoming weights to node i.

2) outdeg
   C_out(i) = |O_i| = sum of positive outgoing weights from node i.

3) total
   C_total(i) = |I_i| + |O_i|.

4) between (aliases: bet, cb)
   Betweenness centrality on the positive-weight graph W_plus.

5) close (aliases: closeness, cc)
   Closeness centrality on the positive-weight graph W_plus.

6) c1  (paper C1)
   c1(i) = (|I_i_plus| - |I_i_minus|) / (|I_i_plus| + |I_i_minus|).

7) c2  (paper C2)
   c2(i) = (|I_i_plus| + |O_i_plus| - |I_i_minus| - |O_i_minus|)
           / (|I_i_plus| + |O_i_plus| + |I_i_minus| + |O_i_minus|).

8) c3  (paper C3)
   c3(i) = |I_i_plus| + |O_i_plus| + |I_i_minus| + |O_i_minus|.

9) c4
   c4(i) = |I_i_plus| - |I_i_minus|.

10) c5
   c5(i) = |I_i_plus| + |O_i_plus| - |I_i_minus| - |O_i_minus|.
"""

import copy
import csv
import logging
import os
import numpy as np
from scipy import linalg
import networkx as nx
import betweenness_model
import closeness_model

# Model names exposed for experiments.
# c1/c2/c3 are paper models, and c4/c5 are additional variants.
MODEL_NAMES = (
    "indeg",
    "outdeg",
    "total",
    "bet",
    "between",
    "cb",
    "close",
    "closeness",
    "cc",
    "c1",
    "c2",
    "c3",
    "c4",
    "c5",
)


# Load and normalize 1D series data from .npy or .csv.
def load_data(data_path="elect.npy"):
    ext = os.path.splitext(data_path)[1].lower()

    # Read NumPy arrays and reduce to one signal dimension.
    if ext == ".npy":
        data = np.load(data_path)
        if data.ndim == 1:
            series = data.astype(float)
        elif data.ndim >= 2:
            # Keep previous behavior (second column) when available.
            col_idx = 1 if data.shape[1] > 1 else 0
            series = data[:, col_idx].astype(float)
        else:
            raise ValueError(f"Unsupported npy shape: {data.shape}")

    # Read CSV and extract the first numeric value per row as 1D series.
    elif ext == ".csv":
        values = []
        with open(data_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                for cell in row:
                    try:
                        values.append(float(cell))
                        break
                    except ValueError:
                        continue
        if not values:
            raise ValueError(f"No numeric values found in CSV: {data_path}")
        series = np.array(values, dtype=float)
    else:
        raise ValueError(f"Unsupported data type: {data_path}. Use .npy or .csv")

    # Normalize to [0, 1] for ESN training stability.
    min_v = np.min(series)
    max_v = np.max(series)
    if max_v == min_v:
        raise ValueError("Input series is constant; normalization is undefined.")
    series = (series - min_v) / (max_v - min_v)
    return series


# Build reservoir and input weights with the configured seeding and scaling logic.
def init_reservoir_weights(res_size, in_size, seed, spectral_radius=1.0):
    rng = np.random.RandomState(seed)

    W = rng.rand(res_size, res_size) - 0.5
    rho_w = np.max(np.abs(linalg.eig(W)[0]))
    # Preserve Echo State Property tendency by scaling to the configured spectral radius.
    W *= spectral_radius / rho_w

    Win = (rng.rand(res_size, 1 + in_size) - 0.5) * 1
    return W, Win


# Split reservoir weights into positive/negative components and degree-like sums.
def compute_weight_components(W):
    W2 = copy.copy(W)
    W3 = copy.copy(W)

    # Keep positive and negative parts separate.
    W2[W2 < 0] = 0
    W3[W2 > 0] = 0
    W3 = -W3

    # Aggregate positive/negative in/out contributions.
    in1 = W2.sum(axis=1)  # In-degree strength: row-wise sum of positive weights.
    out1 = W2.sum(axis=0)  # Out-degree strength: column-wise sum of positive weights.
    in2 = W3.sum(axis=1)
    out2 = W3.sum(axis=0)
    return W2, in1, out1, in2, out2


# Compute indegree-style model scores.
def score_indeg(in1, out1, in2, out2, W2):
    # Return node scores for indegree-based pruning.
    return in1


# Compute outdegree-style model scores.
def score_outdeg(in1, out1, in2, out2, W2):
    # Return node scores for outdegree-based pruning.
    return out1


# Compute total degree-style model scores.
def score_total(in1, out1, in2, out2, W2):
    return in1 + out1


# Compute betweenness-centrality model scores.
def score_between(in1, out1, in2, out2, W2):
    graph = nx.from_numpy_array(W2)
    values = betweenness_model.betweenness_centrality(graph, weight="weight")
    return np.array(list(values.values()))


# Compute closeness-centrality model scores.
def score_close(in1, out1, in2, out2, W2):
    graph = nx.from_numpy_array(W2)
    values = closeness_model.closeness_centrality(graph, distance="weight")
    return np.array(list(values.values()))


# Compute C1 model scores: (|I+| - |I-|) / (|I+| + |I-|).
def score_c1(in1, out1, in2, out2, W2):
    return np.divide(in1 - in2, in1 + in2)


# Compute C2 model scores: (|I+|+|O+|-|I-|-|O-|) / (|I+|+|O+|+|I-|+|O-|).
def score_c2(in1, out1, in2, out2, W2):
    return np.divide(in1 + out1 - in2 - out2, in1 + out1 + in2 + out2)


# Compute C3 model scores: |I+| + |O+| + |I-| + |O-|.
def score_c3(in1, out1, in2, out2, W2):
    return in1 + out1 + in2 + out2


# Compute C4 model scores: |I+| - |I-|.
def score_c4(in1, out1, in2, out2, W2):
    return in1 - in2


# Compute C5 model scores: |I+| + |O+| - |I-| - |O-|.
def score_c5(in1, out1, in2, out2, W2):
    return in1 + out1 - in2 - out2


# Map model names to their scoring functions.
MODEL_SCORE_FUNCTIONS = {
    "indeg": score_indeg,
    "outdeg": score_outdeg,
    "total": score_total,
    "bet": score_between,
    "between": score_between,
    "cb": score_between,
    "close": score_close,
    "closeness": score_close,
    "cc": score_close,
    "c1": score_c1,
    "c2": score_c2,
    "c3": score_c3,
    "c4": score_c4,
    "c5": score_c5,
}

logger = logging.getLogger(__name__)

REQUIRED_CONFIG_KEYS = (
    "train_len",
    "test_len",
    "init_len",
    "in_size",
    "out_size",
    "leak_rate",
    "tau",
    "reg",
)


# Validate required ESN config fields before simulation.
def validate_config(config):
    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in config]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")


# Summarize key ESN reservoir/input properties for logging.
def summarize_esn_state(W, Win):
    if W.size > 0:
        spectral_radius = float(np.max(np.abs(linalg.eigvals(W))))
        density = float(np.count_nonzero(W) / W.size)
        mean_abs_w = float(np.mean(np.abs(W)))
    else:
        spectral_radius = 0.0
        density = 0.0
        mean_abs_w = 0.0

    mean_abs_win = float(np.mean(np.abs(Win))) if Win.size > 0 else 0.0
    return {
        "reservoir_size": int(W.shape[0]),
        "spectral_radius": spectral_radius,
        "density": density,
        "mean_abs_w": mean_abs_w,
        "mean_abs_win": mean_abs_win,
    }


# Compute node ranking for pruning based on the selected centrality/model option.
def centrality_rank(W, model_name):
    W2, in1, out1, in2, out2 = compute_weight_components(W)
    if model_name not in MODEL_SCORE_FUNCTIONS:
        raise ValueError(f"Unknown model_name: {model_name}. Expected one of: {MODEL_NAMES}")

    logger.debug("Computing centrality rank | model=%s", model_name)
    values = MODEL_SCORE_FUNCTIONS[model_name](in1, out1, in2, out2, W2)
    return np.array(values).argsort()


# Compute ESN prediction and MSE exactly following Samsung2.err structure.
def esn_err(W, Win, res_size, data, config):
    cfg = dict(config)
    validate_config(cfg)

    train_len = cfg["train_len"]
    test_len = cfg["test_len"]
    init_len = cfg["init_len"]
    in_size = cfg["in_size"]
    out_size = cfg["out_size"]
    leak_rate = cfg["leak_rate"]
    tau = cfg["tau"]
    reg = cfg["reg"]

    Yt = data[init_len + tau : init_len + train_len + tau]  # Target shifted by tau steps.

    # Collect reservoir states during teacher forcing (input-driven state evolution).
    X = np.zeros((1 + in_size + res_size, train_len))
    x = np.zeros((res_size, 1))
    for t in range(train_len):
        u = np.array([[data[init_len + t]]])
        # Leaky-integrator ESN state update.
        x = (1 - leak_rate) * x + leak_rate * np.tanh(
            np.float64(np.dot(Win, np.vstack((1, u))) + np.dot(W, x))
        )
        X[:, t] = np.vstack((1, u, x))[:, 0]

    # Closed-form ridge regression for stable readout fitting.
    X_T = X.T
    Wout = np.dot(
        np.dot(Yt, X_T),
        linalg.inv(np.dot(X, X_T) + reg * np.eye(1 + in_size + res_size)),
    )

    # Run the test window and produce predictions with the learned readout.
    Y = np.zeros((out_size, test_len))
    x = np.zeros((res_size, 1))
    for t in range(test_len):
        u = np.array([data[init_len + train_len + t]])
        # Same leaky dynamics used during evaluation.
        x = (1 - leak_rate) * x + leak_rate * np.tanh(
            np.float64(np.dot(Win, np.vstack((1, u))) + np.dot(W, x))
        )
        y = np.dot(Wout, np.vstack((1, u, x)))
        Y[:, t] = y

    # Compute test MSE with the same shifted target indexing.
    error_len = test_len
    mse = (
        np.sum(
            np.square(
                data[init_len + train_len + tau : init_len + train_len + error_len + tau]
                - Y[0, 0:error_len]
            )
        )
        / error_len
    )

    return Y, mse


# Prune one node using the current ranked list and return updated tensors/ranks.
def prune_once(W, Win, rank):
    W_new = np.delete(W, rank[0], 0)
    W_new = np.delete(W_new, rank[0], 1)
    Win_new = np.delete(Win, rank[0], 0)

    rank_new = np.array(rank, copy=True)
    rank_new[rank_new > rank_new[0]] -= 1
    rank_list = list(rank_new)
    rank_list.remove(rank_new[0])
    rank_new = np.array(rank_list)

    return W_new, Win_new, rank_new


# Run one full pruning experiment and return all outputs needed for saving/reporting.
def run_single_experiment(res_size, num_prune, seed, model_name, data, config):
    cfg = dict(config)
    validate_config(cfg)

    # Seeded initialization keeps runs reproducible.
    W, Win = init_reservoir_weights(
        res_size=res_size,
        in_size=cfg["in_size"],
        seed=seed,
        spectral_radius=cfg.get("spectral_radius", 1.0),
    )
    W_work = copy.copy(W)
    rank = centrality_rank(W, model_name)
    before_state = summarize_esn_state(W, Win)
    logger.info(
        "Experiment started | model=%s seed=%d reservoir_size=%d num_prune=%d",
        model_name,
        seed,
        res_size,
        num_prune,
    )
    logger.info(
        "ESN before pruning | model=%s seed=%d size=%d spectral_radius=%.6f density=%.6f mean_abs_w=%.6f mean_abs_win=%.6f",
        model_name,
        seed,
        before_state["reservoir_size"],
        before_state["spectral_radius"],
        before_state["density"],
        before_state["mean_abs_w"],
        before_state["mean_abs_win"],
    )

    errors = []
    best_error = 10
    best_index = 0
    best_prediction = None

    for i in range(num_prune):
        Y, err = esn_err(W_work, Win, res_size - i, data=data, config=cfg)

        if err < best_error:
            best_error = err
            best_index = i
            best_prediction = Y

        W_work, Win, rank = prune_once(W_work, Win, rank)
        errors.append(err)

        # Early stop when pruning degrades error beyond threshold.
        if errors[i] > 1.3 * errors[0]:
            logger.info(
                "Early stop triggered | model=%s seed=%d prune_step=%d error=%.6f threshold=%.6f",
                model_name,
                seed,
                i,
                errors[i],
                1.3 * errors[0],
            )
            break

    logger.info(
        "Experiment completed | model=%s seed=%d best_error=%.6f best_index=%d num_pruned=%d",
        model_name,
        seed,
        best_error,
        best_index,
        len(errors),
    )
    after_state = summarize_esn_state(W_work, Win)
    logger.info(
        "ESN after pruning | model=%s seed=%d size=%d spectral_radius=%.6f density=%.6f mean_abs_w=%.6f mean_abs_win=%.6f",
        model_name,
        seed,
        after_state["reservoir_size"],
        after_state["spectral_radius"],
        after_state["density"],
        after_state["mean_abs_w"],
        after_state["mean_abs_win"],
    )
    return {
        "errors": errors,
        "best_error": best_error,
        "best_index": best_index,
        "num_pruned": len(errors),
        "optimal_remaining_nodes": num_prune - best_index,
        "best_prediction": best_prediction,
    }


# Build target and prediction slices for plotting.
def get_plot_series(data, best_prediction, config):
    cfg = dict(config)
    validate_config(cfg)

    init_len = cfg["init_len"]
    train_len = cfg["train_len"]
    test_len = cfg["test_len"]
    tau = cfg["tau"]
    warmup = int(cfg.get("plot_warmup", 50))
    if test_len <= 1:
        warmup = 0
    else:
        warmup = max(0, min(warmup, test_len - 1))

    target = data[warmup + init_len + train_len + tau : init_len + train_len + test_len + tau]
    pred = best_prediction[:, warmup:].T

    # Keep target/prediction lengths aligned for clean plotting.
    common_len = min(len(target), len(pred))
    return target[:common_len], pred[:common_len]
