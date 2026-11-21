"""Data access, scoring and the experiments behind the notebook's 'why does it work' section.

The experiments reuse the models' own code (``build_network``/``train_network`` of the MLP and ``ridge_path`` of the
ridge), so they train exactly what the benchmarked DrEval models train.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from drevalpy.models.utils import load_and_select_gene_features

from .multitask_mlp import build_network, train_network
from .per_drug_ridge import RIDGE_ALPHAS, ridge_path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cached(path: str, compute):
    """
    Return the pickled result at ``path`` if it exists, otherwise compute and store it.

    :param path: cache file; delete it to recompute
    :param compute: function without arguments that produces the result
    :returns: the result
    """
    file = Path(path)
    if file.exists():
        return pd.read_pickle(file)
    result = compute()
    file.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(result, file)
    return result


# --------------------------------------------------------------------------- data
def load_expression(path_data: str = "data", gene_list: str = "landmark_genes") -> pd.DataFrame:
    """
    Load the model's input: arcsinh-transformed expression of the landmark genes.

    :param path_data: data directory
    :param gene_list: name of the DrEval gene list
    :returns: cell lines x genes table
    """
    features = load_and_select_gene_features("gene_expression", gene_list, path_data, "CTRPv2")
    ids = np.array(features.identifiers)
    matrix = features.get_feature_matrix(view="gene_expression", identifiers=ids).astype(np.float64)
    return pd.DataFrame(np.arcsinh(matrix), index=ids, columns=features.meta_info["gene_expression"])


def load_response_table(path_data: str = "data") -> pd.DataFrame:
    """
    Load all screened (cell line, drug) pairs of CTRPv2, including pairs without an IC50.

    :param path_data: data directory
    :returns: long table with cell_line_name, pubchem_id, drug_name, tissue, response (ln IC50, NaN if undefined)
    """
    cols = ["cell_line_name", "pubchem_id", "drug_name", "tissue", "LN_IC50_curvecurator"]
    df = pd.read_csv(Path(path_data) / "CTRPv2" / "CTRPv2.csv", usecols=cols, dtype={"pubchem_id": str}, low_memory=False)
    return df.rename(columns={"LN_IC50_curvecurator": "response"})


def to_matrix(df: pd.DataFrame, cells: np.ndarray, drugs: np.ndarray) -> np.ndarray:
    """
    Pivot a long response table into a cell lines x drugs matrix with NaN for unmeasured pairs.

    :param df: long table with cell_line_name, pubchem_id, response
    :param cells: row order
    :param drugs: column order
    :returns: (len(cells), len(drugs)) matrix
    """
    ci = pd.Index(cells).get_indexer(df.cell_line_name)
    di = pd.Index(drugs).get_indexer(df.pubchem_id)
    ok = (ci >= 0) & (di >= 0)
    out = np.full((len(cells), len(drugs)), np.nan)
    out[ci[ok], di[ok]] = df.response.to_numpy()[ok]
    return out


class Fold:
    """One leaderboard split: training = train + validation files, test = test file."""

    def __init__(self, splits_dir: str, k: int, expression: pd.DataFrame, run_dir: str | None = None) -> None:
        """
        Build the matrices for one split.

        :param splits_dir: folder with cv_split_<k>_<role>.csv
        :param k: split number
        :param expression: cell lines x genes table from ``load_expression``
        :param run_dir: DrEval run folder; if given, the reference for the normalized score is read from the
            NaiveMeanEffectsPredictor predictions DrEval saved there, so scores match DrEval's report exactly
        """

        def read(role: str) -> pd.DataFrame:
            return pd.read_csv(Path(splits_dir) / f"cv_split_{k}_{role}.csv", dtype={"pubchem_id": str})

        train, test = pd.concat([read("train"), read("validation")]), read("test")
        # the reference the 'normalized' metrics subtract: drug mean + tissue effect (NaiveMeanEffectsPredictor)
        overall = train.response.mean()
        tissue_effect = (train.groupby("tissue").response.mean() - overall).to_dict()
        drug_mean_all = train.groupby("pubchem_id").response.mean()
        test_tissue = test.drop_duplicates("cell_line_name").set_index("cell_line_name").tissue
        train = train[train.cell_line_name.isin(expression.index)]
        test = test[test.cell_line_name.isin(expression.index)]
        self.k = k
        self.drugs = np.unique(train.pubchem_id)
        self.train_cells = np.unique(train.cell_line_name)
        # the cell lines DrEval holds out for early stopping (part of the validation set)
        stop = set(read("early_stopping").cell_line_name)
        self.stop_rows = np.array([c in stop for c in self.train_cells])
        self.test_cells = np.unique(test.cell_line_name)
        self.test_tissue = test_tissue.reindex(self.test_cells).to_numpy()
        self.y_train = to_matrix(train, self.train_cells, self.drugs)
        self.y_test = to_matrix(test, self.test_cells, self.drugs)
        self.drug_mean = np.nanmean(self.y_train, axis=0)
        effect = np.array([tissue_effect.get(t, 0.0) for t in self.test_tissue])
        self.baseline = drug_mean_all.reindex(self.drugs).to_numpy()[None, :] + effect[:, None]
        # Pair list for scoring. DrEval joins predictions and reference on (drug, cell line), so a pair that is
        # listed twice in the data is counted four times; the same join is used here to reproduce its numbers.
        ref = test[["cell_line_name", "pubchem_id"]].copy()
        ref["ref"] = self.baseline[
            pd.Index(self.test_cells).get_indexer(test.cell_line_name), pd.Index(self.drugs).get_indexer(test.pubchem_id)
        ]
        if run_dir is not None:
            self.baseline = dreval_predictions(run_dir, "NaiveMeanEffectsPredictor", self)
            path = Path(run_dir) / "CTRPv2" / "LCO" / "NaiveMeanEffectsPredictor" / "predictions"
            naive = pd.read_csv(path / f"predictions_split_{k}.csv", dtype={"pubchem_id": str})
            ref = naive[["cell_line_name", "pubchem_id"]].assign(ref=naive.predictions)
        pairs = test[["cell_line_name", "pubchem_id", "response"]].merge(ref, on=["cell_line_name", "pubchem_id"], how="left")
        pairs = pairs[pairs.pubchem_id.isin(self.drugs)]
        self._row = pd.Index(self.test_cells).get_indexer(pairs.cell_line_name)
        self._col = pd.Index(self.drugs).get_indexer(pairs.pubchem_id)
        self._true, self._ref = pairs.response.to_numpy(), pairs.ref.to_numpy()
        x_train = expression.loc[self.train_cells].to_numpy()
        mu, sd = x_train.mean(0), x_train.std(0)
        ok = sd > 1e-8
        self.x_train = ((x_train[:, ok] - mu[ok]) / sd[ok]).astype(np.float32)
        self.x_test = ((expression.loc[self.test_cells].to_numpy()[:, ok] - mu[ok]) / sd[ok]).astype(np.float32)

    def score(self, pred: np.ndarray, cols: np.ndarray | None = None) -> float:
        """
        Normalized Pearson: correlation of (truth - reference) with (prediction - reference) over test pairs.

        :param pred: (n_test_cells, n_cols) predictions
        :param cols: drug columns the predictions refer to; all drugs if None
        :returns: the correlation
        """
        cols = np.arange(len(self.drugs)) if cols is None else np.asarray(cols)
        where = np.full(len(self.drugs), -1)
        where[cols] = np.arange(len(cols))
        keep = where[self._col] >= 0
        p = pred[self._row[keep], where[self._col[keep]]]
        y, b = self._true[keep], self._ref[keep]
        m = ~np.isnan(p) & ~np.isnan(b)
        return float(np.corrcoef((y - b)[m], (p - b)[m])[0, 1])


# --------------------------------------------------------------------------- models for the ablations
def fit_multitask(
    fold: Fold, cols: np.ndarray | None = None, n_seeds: int = 10, return_nets: bool = False, rows: np.ndarray | None = None
):
    """
    Train the multi-task MLP on a subset of drugs and predict the test cell lines, exactly as the DrEval model does.

    Training uses the split's training cell lines except DrEval's early-stopping cell lines, on which each network
    is stopped early. Genes and drugs are standardized on the training cell lines.

    :param fold: the split
    :param cols: drug columns to use as outputs; all drugs if None
    :param n_seeds: number of networks to average
    :param return_nets: also return the trained networks
    :param rows: boolean mask of cell lines to use (training and early-stopping); all if None
    :returns: (n_test_cells, n_cols) predictions, and the networks if requested
    """
    cols = np.arange(len(fold.drugs)) if cols is None else np.asarray(cols)
    use = np.ones(len(fold.train_cells), bool) if rows is None else np.asarray(rows)
    tr, st = use & ~fold.stop_rows, use & fold.stop_rows
    y, y_stop = fold.y_train[tr][:, cols], fold.y_train[st][:, cols]
    mean, sd = np.nanmean(y, axis=0), np.nanstd(y, axis=0)
    sd[~(sd > 0)] = 1.0
    mu, sg = fold.x_train[tr].mean(0), fold.x_train[tr].std(0)
    sg[~(sg > 1e-8)] = 1.0

    def tensors(x, yy):
        t = (yy - mean[None, :]) / sd[None, :]
        return (torch.as_tensor((x - mu) / sg, dtype=torch.float32, device=DEVICE),
                torch.as_tensor(np.nan_to_num(t), dtype=torch.float32, device=DEVICE),
                torch.as_tensor(~np.isnan(t), dtype=torch.float32, device=DEVICE))

    x, target, mask = tensors(fold.x_train[tr], y)
    x_stop, target_stop, mask_stop = tensors(fold.x_train[st], y_stop)
    x_test = torch.as_tensor((fold.x_test - mu) / sg, dtype=torch.float32, device=DEVICE)
    out, nets = 0.0, []
    for s_ in range(n_seeds):
        torch.manual_seed(s_)
        net, _ = train_network(build_network(x.shape[1], len(cols)).to(DEVICE), x, target, mask, x_stop, target_stop, mask_stop)
        with torch.no_grad():
            out = out + net(x_test).double().cpu().numpy()
        nets.append(net)
    pred = out / n_seeds * sd[None, :] + mean[None, :]
    return (pred, nets) if return_nets else pred


def fit_single_task_mlps(fold: Fold, cols: np.ndarray, n_seeds: int = 5) -> np.ndarray:
    """
    Train one MLP per drug, same architecture and settings, each on the cell lines measured for that drug.

    :param fold: the split
    :param cols: drug columns
    :param n_seeds: number of networks to average per drug
    :returns: (n_test_cells, n_cols) predictions
    """
    out = np.zeros((len(fold.test_cells), len(cols)))
    for j, c in enumerate(cols):
        rows = ~np.isnan(fold.y_train[:, c])
        if not (rows & fold.stop_rows).any() or not (rows & ~fold.stop_rows).any():
            out[:, j] = np.nanmean(fold.y_train[:, c])  # no early-stopping cell line measured for this drug
            continue
        out[:, j] = fit_multitask(fold, np.array([c]), n_seeds=n_seeds, rows=rows)[:, 0]
    return out


def fit_per_drug_ridge(fold: Fold, cols: np.ndarray | None = None, return_path: bool = False):
    """
    One ridge regression per drug on the same genes, penalty chosen per drug by exact leave-one-out error.

    Calls ``ridge_path``, the same function the PerDrugRidge DrEval model uses.

    :param fold: the split
    :param cols: drug columns; all drugs if None
    :param return_path: also return the predictions for every penalty and the index chosen per drug
    :returns: (n_test_cells, n_cols) predictions; with ``return_path`` also (n_penalties, n_test_cells, n_cols)
        predictions and (n_cols,) chosen indices (-1 for drugs with too few measurements)
    """
    cols = np.arange(len(fold.drugs)) if cols is None else np.asarray(cols)
    w, b, chosen = ridge_path(fold.x_train.astype(np.float64), fold.y_train[:, cols])
    path = np.einsum("ng,agd->and", fold.x_test.astype(np.float64), w) + b[:, None, :]
    out = path[np.maximum(chosen, 0), :, np.arange(len(cols))].T
    return (out, path, chosen) if return_path else out


# --------------------------------------------------------------------------- analyses
def cotraining_curve(
    fold: Fold, targets: np.ndarray, n_total: list[int], n_seeds: int = 5, seed: int = 0
) -> dict[str, float]:
    """
    Score a fixed set of target drugs while the number of drugs trained alongside them grows.

    :param fold: the split
    :param targets: drug columns that are scored
    :param n_total: total numbers of output drugs to try (targets plus randomly chosen other drugs)
    :param n_seeds: networks averaged per setting
    :param seed: seed for choosing the other drugs
    :returns: setting name -> normalized Pearson on the target drugs
    """
    rng = np.random.default_rng(seed)
    others = rng.permutation(np.setdiff1d(np.arange(len(fold.drugs)), targets))
    res = {
        "per-drug ridge": fold.score(fit_per_drug_ridge(fold, targets), targets),
        "1 (one network per drug)": fold.score(fit_single_task_mlps(fold, targets, n_seeds), targets),
    }
    for n in n_total:
        cols = np.concatenate([targets, others[: max(n - len(targets), 0)]])
        res[str(len(cols))] = fold.score(fit_multitask(fold, cols, n_seeds)[:, : len(targets)], targets)
    return res


def split_general_specific(fold: Fold, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Split predictions into a part shared by all drugs of a cell line and a drug-specific rest.

    The shared part is the cell line's mean predicted deviation from the reference, over all drugs.

    :param fold: the split
    :param pred: (n_test_cells, n_drugs) predictions
    :returns: (general-only predictions, predicted general level per cell line)
    """
    level = np.nanmean(pred - fold.baseline, axis=1)
    return fold.baseline + level[:, None], level


def observed_general_level(fold: Fold) -> np.ndarray:
    """
    The measured counterpart of the general level: a test cell line's mean deviation from the reference.

    :param fold: the split
    :returns: (n_test_cells,) mean of (truth - reference) over the measured drugs
    """
    return np.nanmean(fold.y_test - fold.baseline, axis=1)


def per_drug_pearson(fold: Fold, pred: np.ndarray, min_n: int = 10) -> np.ndarray:
    """
    Pearson correlation per drug across the test cell lines, after subtracting the reference.

    :param fold: the split
    :param pred: (n_test_cells, n_drugs) predictions
    :param min_n: drugs with fewer measured test cell lines get NaN
    :returns: (n_drugs,) correlations
    """
    out = np.full(len(fold.drugs), np.nan)
    for d in range(len(fold.drugs)):
        m = ~np.isnan(fold.y_test[:, d])
        if m.sum() >= min_n:
            a, b = (fold.y_test[m, d] - fold.baseline[m, d]), (pred[m, d] - fold.baseline[m, d])
            if a.std() > 0 and b.std() > 0:
                out[d] = np.corrcoef(a, b)[0, 1]
    return out


def dreval_predictions(run_dir: str, model: str, fold: Fold) -> np.ndarray:
    """
    Read the predictions DrEval saved for one split as a test cell lines x drugs matrix.

    :param run_dir: run folder, e.g. results/multitask_mlp
    :param model: model name
    :param fold: the split
    :returns: (n_test_cells, n_drugs) matrix, NaN where no prediction was requested
    """
    path = Path(run_dir) / "CTRPv2" / "LCO" / model / "predictions" / f"predictions_split_{fold.k}.csv"
    df = pd.read_csv(path, dtype={"pubchem_id": str})
    df = df.drop(columns="response").rename(columns={"predictions": "response"})
    return to_matrix(df, fold.test_cells, fold.drugs)


def load_all_predictions(run_dir: str, models: list[str], folds: list[Fold]) -> dict[str, list[np.ndarray]]:
    """
    Read the predictions DrEval saved for several models.

    :param run_dir: run folder
    :param models: model names
    :param folds: the splits
    :returns: model -> list of (n_test_cells, n_drugs) matrices, one per split
    """
    return {m: [dreval_predictions(run_dir, m, f) for f in folds] for m in models}


def normalized_pairs(fold: Fold, preds: dict[str, np.ndarray]) -> pd.DataFrame:
    """
    The view the challenge metric scores: every test pair's prediction and truth minus the naive reference.

    :param fold: the split
    :param preds: model -> (n_test_cells, n_drugs) predictions
    :returns: one row per scored pair, one column per model plus "measured"
    """
    out = pd.DataFrame({m: p[fold._row, fold._col] - fold._ref for m, p in preds.items()})
    out["measured"] = fold._true - fold._ref
    return out.dropna()


def per_drug_table(folds: list[Fold], preds: dict[str, list[np.ndarray]], min_n: int = 10) -> pd.DataFrame:
    """
    Per-drug correlation of each model across test cell lines (reference subtracted), averaged over splits.

    :param folds: the splits
    :param preds: model -> list of prediction matrices, one per split
    :param min_n: minimum number of measured test cell lines for a drug to be scored in a split
    :returns: one row per drug: mean correlation per model, response spread and number of training cell lines
    """
    parts = []
    for i, f in enumerate(folds):
        parts.append(pd.DataFrame({
            "drug": f.drugs,
            "response sd": np.nanstd(f.y_train, axis=0),
            "training cell lines": (~np.isnan(f.y_train)).sum(0),
            **{m: per_drug_pearson(f, p[i], min_n) for m, p in preds.items()},
        }))
    return pd.concat(parts).groupby("drug").mean()


def published_predictions(path: str, fold: Fold) -> np.ndarray:
    """
    Read a model's published leaderboard predictions for one split as a test cell lines x drugs matrix.

    :param path: CSV with the model's rows of the leaderboard's true_vs_pred.csv
    :param fold: the split
    :returns: (n_test_cells, n_drugs) matrix, NaN where the model made no prediction
    """
    df = pd.read_csv(path, dtype={"pubchem_id": str})
    df = df[df.CV_split == fold.k].rename(columns={"y_pred": "response"})
    return to_matrix(df, fold.test_cells, fold.drugs)
