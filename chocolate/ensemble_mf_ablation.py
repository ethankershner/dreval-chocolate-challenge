"""
The EnsembleMF ablation: run EnsembleMFStandardized on the leaderboard splits and score it against EnsembleMF.

``run`` trains EnsembleMFStandardized with DrEval's experiment runner on all 10 leaderboard splits and stores the
predictions in ``results/emf_standardized`` under the original split numbers. EnsembleMF is a small network, so one
split does not keep the GPU busy; ``run`` therefore trains several splits at once in separate worker processes,
which costs no extra time per split (about 8 minutes each on one GPU, so about 30 minutes for all 10 with 3 workers).

``test_table`` and ``score`` put every model's test predictions for one split side by side and compute DrEval's
normalized Pearson correlation, so that averages of models can be scored exactly as DrEval scores a model.

Run a worker by hand (from the repo root): ``python -m chocolate.ensemble_mf_ablation <worker_run_id> <split> ...``
"""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .run import BASELINE, DATASET, TEST_MODE, load_response

MODEL = "EnsembleMFStandardized"
RUN_ID = "emf_standardized"
KEYS = ["cell_line_name", "pubchem_id"]


def _mode_dir(results: str, run_id: str) -> Path:
    return Path(results) / run_id / DATASET / TEST_MODE


def done_splits(results: str = "results", run_id: str = RUN_ID) -> list[int]:
    """
    :param results: results directory
    :param run_id: run folder of the ablation
    :returns: leaderboard splits for which predictions of both the model and the baseline are stored
    """
    d = _mode_dir(results, run_id)
    return [k for k in range(10) if all((d / m / "predictions" / f"predictions_split_{k}.csv").exists() for m in [MODEL, BASELINE])]


def worker(run_id: str, splits: list[int], splits_dir: str = "reference/splits", results: str = "results", path_data: str = "data") -> None:
    """
    Run DrEval on a subset of the leaderboard splits in a run folder of its own.

    DrEval numbers splits by their sorted position in the splits folder, so the chosen splits are copied in as
    0..k-1; ``split_map.txt`` records the original numbers for ``gather``.

    :param run_id: worker run folder
    :param splits: leaderboard split numbers
    :param splits_dir: folder with the leaderboard split CSVs
    :param results: results directory
    :param path_data: data directory
    """
    from drevalpy.experiment import drug_response_experiment

    from .ensemble_mf_standardized import EnsembleMFStandardized

    target = _mode_dir(results, run_id) / "splits"
    if not target.exists():
        target.mkdir(parents=True)
        for new, old in enumerate(splits):
            for f in Path(splits_dir).glob(f"cv_split_{old}_*.csv"):
                shutil.copy(f, target / f.name.replace(f"cv_split_{old}_", f"cv_split_{new}_"))
        (target.parent / "split_map.txt").write_text("\n".join(f"{n} {o}" for n, o in enumerate(splits)))
    drug_response_experiment(
        models=[EnsembleMFStandardized],
        baselines=[],  # DrEval always adds the NaiveMeanEffectsPredictor the normalized metrics need
        response_data=load_response(path_data),
        test_mode=TEST_MODE,
        run_id=run_id,
        path_out=results,
        path_data=path_data,
        overwrite=False,
    )


def gather(worker_run_id: str, results: str = "results", run_id: str = RUN_ID) -> list[int]:
    """
    Copy a worker's predictions into the ablation run folder under their original split numbers.

    :param worker_run_id: worker run folder (with split_map.txt)
    :param results: results directory
    :param run_id: run folder of the ablation
    :returns: the leaderboard splits copied
    """
    src, dst = _mode_dir(results, worker_run_id), _mode_dir(results, run_id)
    copied = []
    for line in (src / "split_map.txt").read_text().split("\n"):
        new, old = map(int, line.split())
        files = {m: src / m / "predictions" / f"predictions_split_{new}.csv" for m in [MODEL, BASELINE]}
        if not all(f.exists() for f in files.values()):
            continue
        for m, f in files.items():
            (dst / m / "predictions").mkdir(parents=True, exist_ok=True)
            shutil.copy(f, dst / m / "predictions" / f"predictions_split_{old}.csv")
        copied.append(old)
    return copied


def run(n_workers: int = 3, results: str = "results", run_id: str = RUN_ID) -> list[int]:
    """
    Train EnsembleMFStandardized on every leaderboard split that has no stored predictions yet.

    The missing splits are dealt round-robin to ``n_workers`` processes that share the GPU. Their predictions are
    gathered into ``results/<run_id>`` and the worker folders are removed.

    :param n_workers: number of splits trained at the same time
    :param results: results directory
    :param run_id: run folder of the ablation
    :returns: the splits that were trained
    :raises RuntimeError: if a worker fails
    """
    todo = [k for k in range(10) if k not in done_splits(results, run_id)]
    if not todo:
        print(f"All 10 splits are stored in {_mode_dir(results, run_id)}; nothing to train.")
        return []
    groups = [todo[i::n_workers] for i in range(n_workers) if todo[i::n_workers]]
    jobs = []
    for i, group in enumerate(groups):
        wid = f"{run_id}_worker{i}"
        log = open(Path(results) / f"{wid}.log", "w", encoding="utf-8")
        cmd = [sys.executable, "-m", "chocolate.ensemble_mf_ablation", wid, *map(str, group)]
        jobs.append((wid, log, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)))
        print(f"worker {i}: splits {group}")
    failed = []
    for wid, log, proc in jobs:
        proc.wait()
        log.close()
        if proc.returncode != 0:
            failed.append(wid)
            continue
        gather(wid, results, run_id)
        shutil.rmtree(Path(results) / wid)
    if failed:
        raise RuntimeError(f"Workers {failed} failed; see results/<worker>.log")
    return todo


def _read(path: Path, name: str) -> pd.DataFrame:
    """
    Read one DrEval prediction file as one row per (cell line, drug).

    DrEval's prediction files repeat a few pairs (identical rows); ``c`` counts how often.

    :param path: prediction CSV
    :param name: column name for the predictions
    :returns: table with KEYS, response, the predictions and c
    """
    d = pd.read_csv(path, dtype={"pubchem_id": str})
    counts = d.groupby(KEYS).size().rename("c").reset_index()
    d = d.drop_duplicates(KEYS)[KEYS + ["response", "predictions"]].rename(columns={"predictions": name})
    return d.merge(counts, on=KEYS)


def test_table(
    k: int,
    results: str = "results",
    final_run: str = "final",
    run_id: str = RUN_ID,
    leaderboard: str = "reference/leaderboard_EnsembleMF_true_vs_pred.csv",
) -> pd.DataFrame:
    """
    All test predictions of one leaderboard split, one row per (cell line, drug).

    :param k: leaderboard split
    :param results: results directory
    :param final_run: run folder with PerDrugRidge, MultiTaskMLP and the baseline
    :param run_id: run folder of the ablation
    :param leaderboard: published EnsembleMF predictions (true_vs_pred rows of EnsembleMF)
    :returns: columns response, naive (the NaiveMeanEffectsPredictor), ridge, mlp, emf (published EnsembleMF),
        emf_std (EnsembleMFStandardized) and c (how often DrEval's files repeat the pair)
    """
    f = _mode_dir(results, final_run)
    t = _read(f / BASELINE / "predictions" / f"predictions_split_{k}.csv", "naive")
    for name, model in [("ridge", "PerDrugRidge"), ("mlp", "MultiTaskMLP")]:
        t = t.merge(_read(f / model / "predictions" / f"predictions_split_{k}.csv", name).drop(columns=["response", "c"]), on=KEYS)
    std = _read(_mode_dir(results, run_id) / MODEL / "predictions" / f"predictions_split_{k}.csv", "emf_std")
    t = t.merge(std.drop(columns=["response", "c"]), on=KEYS)
    lb = pd.read_csv(leaderboard, dtype={"pubchem_id": str})
    lb = lb[(lb.rand_setting == "predictions") & (lb.CV_split == k)].drop_duplicates(KEYS)
    t = t.merge(lb[KEYS + ["y_pred"]].rename(columns={"y_pred": "emf"}), on=KEYS, how="left")
    return t.dropna(subset=["response", "naive"]).reset_index(drop=True)


def score(t: pd.DataFrame, pred) -> float:
    """
    DrEval's normalized Pearson correlation of a prediction column (or any combination of columns).

    DrEval subtracts the NaiveMeanEffectsPredictor from response and prediction and correlates the rest over all test
    pairs of the split, pooled across drugs. It merges two files that both repeat a pair c times, so that pair counts
    c**2 times; the weights reproduce this exactly.

    :param t: table from ``test_table``
    :param pred: predictions aligned with ``t``
    :returns: normalized Pearson correlation
    """
    w = t.c.to_numpy(float) ** 2
    a = (t.response - t.naive).to_numpy()
    b = (np.asarray(pred, dtype=float) - t.naive.to_numpy())
    a, b = a - np.average(a, weights=w), b - np.average(b, weights=w)
    return float(np.sum(w * a * b) / np.sqrt(np.sum(w * a * a) * np.sum(w * b * b)))


if __name__ == "__main__":
    worker(sys.argv[1], [int(s) for s in sys.argv[2:]])
