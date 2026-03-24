# -*- coding: utf-8 -*-

import argparse
import csv
import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

import matplotlib.pyplot as plt
import numpy as np

from core_logic import get_plot_series, load_data, run_single_experiment

DEFAULT_DATA_PATH = "./data/traffic.npy"
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_LOG_PATH = os.path.join("log", "log.log")

DEFAULT_MODELS = "indeg"
DEFAULT_RESERVOIR_SIZES = [100, 200, 300, 400, 500, 600, 700, 800, 900]
DEFAULT_PRUNE_PERCENT = 0.40
DEFAULT_NUM_PRUNE = None
DEFAULT_SEED_START = 0
DEFAULT_RUNS = 10
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_INIT_RATIO = 0.10
DEFAULT_TRAIN_RATIO = 0.80
DEFAULT_TEST_RATIO = 0.10

# Main ESN experiment parameters are controlled here.
BASE_ESN_CONFIG = {
    "in_size": 1,
    "out_size": 1,
    "leak_rate": 0.5,
    "tau": 2,
    "reg": 1e-4,
    "spectral_radius": 0.99,
    "plot_warmup": 50,
    "init_len": 0,
    "train_len": 0,
    "test_len": 0,
}


# Configure console and file logging for reproducible experiment tracking.
def setup_logging(log_path, log_level):
    log_dir = os.path.dirname(log_path) or "."
    os.makedirs(log_dir, exist_ok=True)
    level = getattr(logging, log_level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


# Parse reservoir-size list like "100,200,300" into integer list.
def parse_reservoir_sizes(value):
    return [int(x.strip()) for x in value.split(",") if x.strip()]


# Build dynamic ESN split config from ratios (on n - tau usable points).
def build_dynamic_config(data_length, base_config, init_ratio, train_ratio, test_ratio):
    if abs((init_ratio + train_ratio + test_ratio) - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.0.")

    tau = base_config["tau"]
    usable_length = data_length - tau
    if usable_length < 10:
        raise ValueError(f"Data is too short for dynamic split: length={data_length}, tau={tau}")

    init_len = int(init_ratio * usable_length)
    train_len = int(train_ratio * usable_length)
    test_len = usable_length - init_len - train_len

    # Ensure test window is at least one step.
    if test_len < 1:
        test_len = 1
        train_len = max(1, train_len - 1)

    cfg = dict(base_config)
    cfg.update(
        {
            "init_len": init_len,
            "train_len": train_len,
            "test_len": test_len,
        }
    )
    return cfg


# Parse command-line arguments for training and output settings.
def parse_args():
    parser = argparse.ArgumentParser(description="Train ESN pruning models and save outputs.")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH, help="Path to input .npy or .csv data file")
    parser.add_argument(
        "--models",
        default=DEFAULT_MODELS,
        help="Comma-separated model names (e.g., indeg,outdeg,total,between,close,c1,c2,c3,c4,c5; aliases: bet/cb and closeness/cc)",
    )
    parser.add_argument(
        "--reservoir-sizes",
        default=",".join(str(x) for x in DEFAULT_RESERVOIR_SIZES),
        help="Comma-separated reservoir sizes (default: 100..900)",
    )
    parser.add_argument(
        "--prune-percent",
        type=float,
        default=DEFAULT_PRUNE_PERCENT,
        help="Fraction of reservoir nodes to prune (0.40 means 40%%)",
    )
    parser.add_argument(
        "--num-prune",
        type=int,
        default=DEFAULT_NUM_PRUNE,
        help="Maximum pruning iterations (overrides prune-percent when set)",
    )
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START, help="First random seed")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="Number of seeds to run")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory to save plots and CSV")
    parser.add_argument("--log-path", default=DEFAULT_LOG_PATH, help="Path to log file")
    parser.add_argument("--log-level", default=DEFAULT_LOG_LEVEL, help="Logging level: DEBUG, INFO, WARNING, ERROR")
    parser.add_argument("--init-ratio", type=float, default=DEFAULT_INIT_RATIO, help="Initialization split ratio")
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO, help="Training split ratio")
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO, help="Testing split ratio")
    parser.add_argument("--tau", type=int, default=BASE_ESN_CONFIG["tau"], help="Prediction horizon tau")
    parser.add_argument(
        "--spectral-radius",
        type=float,
        default=BASE_ESN_CONFIG["spectral_radius"],
        help="Reservoir spectral radius target after scaling",
    )
    parser.add_argument(
        "--plot-warmup",
        type=int,
        default=BASE_ESN_CONFIG["plot_warmup"],
        help="Number of initial test points to skip in prediction plots",
    )
    return parser.parse_args()


# Save one error-curve plot with the requested naming convention.
def save_error_plot(errors, model_name, reservoir_size, pruned_index, seed, output_dir):
    plt.figure(2)
    plt.clf()
    # Plot pruning error trajectory (same behavior as current plot).
    plt.plot(errors, "r", label="Pruning error")
    # Add base ESN reference line (before pruning) as dashed line.
    plt.axhline(errors[0], color="k", linestyle="--", linewidth=1.2, label="Base ESN")
    plt.title("Pruning Error Curve")
    plt.xlabel("Pruned Nodes")
    plt.ylabel("MSE")
    plt.legend()

    file_name = f"{model_name}_{reservoir_size}_pruned_{pruned_index}_seed_{seed}_error.png"
    plt.savefig(os.path.join(output_dir, file_name))


# Save one target-vs-prediction plot with the requested naming convention.
def save_prediction_plot(target, prediction, model_name, reservoir_size, pruned_index, seed, output_dir):
    target_arr = np.asarray(target).reshape(-1)
    pred_arr = np.asarray(prediction).reshape(-1)
    common_len = min(len(target_arr), len(pred_arr))
    target_arr = target_arr[:common_len]
    pred_arr = pred_arr[:common_len]

    plt.figure(1)
    plt.clf()
    plt.plot(target_arr, "g")
    plt.plot(pred_arr, "b")
    plt.title("Target and generated signals $y(n)$ starting")
    plt.legend(["Target", "Prediction"])

    # Use percentile-based limits to reduce visual distortion from rare spikes.
    merged = np.concatenate([target_arr, pred_arr])
    finite = merged[np.isfinite(merged)]
    if finite.size > 0:
        lo = float(np.percentile(finite, 1))
        hi = float(np.percentile(finite, 99))
        if hi > lo:
            pad = 0.08 * (hi - lo)
            plt.ylim(lo - pad, hi + pad)

    file_name = f"{model_name}_{reservoir_size}_pruned_{pruned_index}_seed_{seed}_pred.png"
    plt.savefig(os.path.join(output_dir, file_name))


# Run all requested experiments and keep rows for one final all-at-once CSV write.
def run_training(args):
    logger = logging.getLogger(__name__)

    # Create a unique run directory to avoid overwriting results across runs.
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(run_output_dir, exist_ok=True)

    # Support both .npy and .csv and enforce 1D series inside load_data.
    data = load_data(args.data_path)

    # Override base config in train.py so experiment control is centralized here.
    base_config = dict(BASE_ESN_CONFIG)
    base_config["tau"] = args.tau
    base_config["spectral_radius"] = args.spectral_radius
    base_config["plot_warmup"] = args.plot_warmup

    # Dynamic split from dataset length using configured ratios.
    dynamic_config = build_dynamic_config(
        len(data),
        base_config,
        args.init_ratio,
        args.train_ratio,
        args.test_ratio,
    )

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    reservoir_sizes = parse_reservoir_sizes(args.reservoir_sizes)

    logger.info(
        "Training started | models=%s reservoir_sizes=%s prune_percent=%.3f runs=%d seed_start=%d data_path=%s run_output_dir=%s",
        model_names,
        reservoir_sizes,
        args.prune_percent,
        args.runs,
        args.seed_start,
        args.data_path,
        run_output_dir,
    )
    logger.info(
        "Dynamic split | init_ratio=%.2f train_ratio=%.2f test_ratio=%.2f init_len=%d train_len=%d test_len=%d tau=%d spectral_radius=%.3f",
        args.init_ratio,
        args.train_ratio,
        args.test_ratio,
        dynamic_config["init_len"],
        dynamic_config["train_len"],
        dynamic_config["test_len"],
        dynamic_config["tau"],
        dynamic_config["spectral_radius"],
    )

    all_rows = []

    # Loop through reservoir-size list and run each model/seed combination.
    for reservoir_size in reservoir_sizes:
        computed_num_prune = max(1, int(reservoir_size * args.prune_percent))
        num_prune = args.num_prune if args.num_prune is not None else computed_num_prune

        logger.info(
            "Reservoir loop | reservoir_size=%d num_prune=%d",
            reservoir_size,
            num_prune,
        )

        for model_name in model_names:
            for seed in range(args.seed_start, args.seed_start + args.runs):
                logger.info(
                    "Run started | model=%s reservoir_size=%d seed=%d",
                    model_name,
                    reservoir_size,
                    seed,
                )

                result = run_single_experiment(
                    res_size=reservoir_size,
                    num_prune=num_prune,
                    seed=seed,
                    model_name=model_name,
                    data=data,
                    config=dynamic_config,
                )

                errors = result["errors"]
                best_error = result["best_error"]
                best_index = result["best_index"]
                num_pruned = result["num_pruned"]

                # Save plots with reservoir size in name so list-runs do not overwrite each other.
                save_error_plot(
                    errors=errors,
                    model_name=model_name,
                    reservoir_size=reservoir_size,
                    pruned_index=best_index,
                    seed=seed,
                    output_dir=run_output_dir,
                )

                target, pred = get_plot_series(data, result["best_prediction"], config=dynamic_config)
                if len(target) > 0 and len(pred) > 0:
                    logger.debug(
                        "Plot range | model=%s reservoir_size=%d seed=%d target_min=%.6f target_max=%.6f pred_min=%.6f pred_max=%.6f",
                        model_name,
                        reservoir_size,
                        seed,
                        float(np.min(target)),
                        float(np.max(target)),
                        float(np.min(pred)),
                        float(np.max(pred)),
                    )
                save_prediction_plot(
                    target=target,
                    prediction=pred,
                    model_name=model_name,
                    reservoir_size=reservoir_size,
                    pruned_index=best_index,
                    seed=seed,
                    output_dir=run_output_dir,
                )

                # Append per-step rows; CSV will be saved once at the end.
                for prune_index, err in enumerate(errors):
                    all_rows.append(
                        {
                            "model": model_name,
                            "reservoir_size": reservoir_size,
                            "seed": seed,
                            "prune_index": prune_index,
                            "error": err,
                            "best_error": best_error,
                            "best_pruned_index": best_index,
                            "num_pruned": num_pruned,
                            "optimal_remaining_nodes": result["optimal_remaining_nodes"],
                            "init_len": dynamic_config["init_len"],
                            "train_len": dynamic_config["train_len"],
                            "test_len": dynamic_config["test_len"],
                            "tau": dynamic_config["tau"],
                        }
                    )

                print(
                    f"model={model_name} reservoir={reservoir_size} seed={seed} initial_error={errors[0]:.6f} "
                    f"best_error={best_error:.6f} best_index={best_index} num_pruned={num_pruned}"
                )
                logger.info(
                    "Run completed | model=%s reservoir_size=%d seed=%d initial_error=%.6f best_error=%.6f best_index=%d num_pruned=%d",
                    model_name,
                    reservoir_size,
                    seed,
                    errors[0],
                    best_error,
                    best_index,
                    num_pruned,
                )

    # Write one aggregated CSV file after all runs complete.
    csv_path = os.path.join(run_output_dir, "all_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "reservoir_size",
                "seed",
                "prune_index",
                "error",
                "best_error",
                "best_pruned_index",
                "num_pruned",
                "optimal_remaining_nodes",
                "init_len",
                "train_len",
                "test_len",
                "tau",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Saved CSV: {csv_path}")
    logger.info("Training finished | csv_path=%s total_rows=%d", csv_path, len(all_rows))


# Entrypoint for command-line execution.
def main():
    args = parse_args()
    setup_logging(args.log_path, args.log_level)
    logging.getLogger(__name__).info("Log file initialized at %s", args.log_path)
    run_training(args)


if __name__ == "__main__":
    main()
