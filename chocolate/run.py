"""Run models through DrEval on the leaderboard splits and build DrEval's evaluation report."""

import shutil
from pathlib import Path

import pandas as pd
from drevalpy.datasets.loader import load_dataset
from drevalpy.experiment import drug_response_experiment
from drevalpy.models import MODEL_FACTORY
from drevalpy.models.drp_model import DRPModel
from drevalpy.visualization.create_report import create_report

DATASET = "CTRPv2"
TEST_MODE = "LCO"
METRIC = "Pearson: normalized"
BASELINE = "NaiveMeanEffectsPredictor"  # the reference the normalized metrics subtract


def load_response(path_data: str = "data"):
    """
    Load the CTRPv2 response dataset exactly as the ``drevalpy`` command line does by default.

    :param path_data: data directory; the dataset is downloaded from Zenodo if it is missing
    :returns: DrugResponseDataset with CurveCurator ln(IC50) as the response
    """
    return load_dataset(dataset_name=DATASET, path_data=path_data, measure="LN_IC50", curve_curator=True)


def finished(run_dir: Path, model: str) -> int:
    """
    Count the splits for which DrEval has saved predictions of a model.

    :param run_dir: run folder
    :param model: model name
    :returns: number of prediction files
    """
    return len(list((run_dir / DATASET / TEST_MODE / model / "predictions").glob("predictions_split_*.csv")))


def run_dreval(
    run_id: str,
    models: list[type[DRPModel]],
    path_data: str = "data",
    path_out: str = "results",
    splits_dir: str = "reference/splits",
) -> Path:
    """
    Train and test models on the precomputed leaderboard splits with DrEval's experiment runner.

    DrEval reuses splits that already sit in ``<path_out>/<run_id>/CTRPv2/LCO/splits``, so the leaderboard splits are
    copied there first. DrEval skips every (model, split) whose predictions already exist, so an interrupted run
    resumes and a finished run is not repeated.

    :param run_id: name of the result folder
    :param models: model classes to run; the NaiveMeanEffectsPredictor baseline is always added
    :param path_data: data directory
    :param path_out: results directory
    :param splits_dir: folder with the unzipped leaderboard splits (cv_split_<k>_<role>.csv)
    :returns: path of the run folder
    :raises FileNotFoundError: if the leaderboard splits are missing
    """
    run_dir = Path(path_out) / run_id
    mode_dir = run_dir / DATASET / TEST_MODE
    n_splits = len(list(Path(splits_dir).glob("cv_split_*_test.csv")))
    if n_splits == 0:
        raise FileNotFoundError(f"No leaderboard splits in {splits_dir}; unzip splits.zip from the DrEval Zenodo record.")
    names = [m.get_model_name() for m in models] + [BASELINE]
    if all(finished(run_dir, n) == n_splits for n in names):
        print(f"Run '{run_id}' already has predictions of every model for all {n_splits} splits; not repeating it.")
        return run_dir
    if not (mode_dir / "splits").exists():
        mode_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(splits_dir, mode_dir / "splits")
    drug_response_experiment(
        models=models,
        baselines=[MODEL_FACTORY[BASELINE]],
        response_data=load_response(path_data),
        test_mode=TEST_MODE,
        run_id=run_id,
        path_out=path_out,
        path_data=path_data,
        overwrite=False,
    )
    return run_dir


def evaluate(run_id: str, path_data: str = "data", path_out: str = "results") -> pd.DataFrame:
    """
    Score a run with DrEval's evaluation suite and write its HTML report.

    :param run_id: name of the result folder
    :param path_data: data directory
    :param path_out: results directory
    :returns: DrEval's evaluation table for the plain predictions (one row per model and split)
    """
    out = Path(path_out) / run_id / "evaluation_results.csv"
    if not out.exists():
        create_report(run_id=run_id, dataset=DATASET, path_data=path_data, result_path=path_out)
    table = pd.read_csv(out, index_col=0)
    return table[(table.rand_setting == "predictions") & (table.test_mode == TEST_MODE)]
