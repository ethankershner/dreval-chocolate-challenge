"""
EnsembleMFStandardized: DrEval's EnsembleMF trained on per-drug standardized ln(IC50).

Identical to EnsembleMF in every respect except the target. Each drug's training responses are shifted and scaled
to mean 0 and standard deviation 1 (statistics from the training pairs only) before training, and predictions are
mapped back to ln(IC50) with the same statistics. Early stopping uses the same transform. Its settings in
hyperparameters.yaml are those of DrEval's own EnsembleMF entry.

Used in ensemble_mf_ablation.ipynb to test whether EnsembleMF's raw-scale loss, which lets drugs with a wide response
spread dominate, explains part of its gap to the models here.
"""

import os

import joblib
import numpy as np

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models.EnsembleMF.ensemble_mf import EnsembleMF


class EnsembleMFStandardized(EnsembleMF):
    """EnsembleMF with per-drug standardized targets."""

    def __init__(self) -> None:
        """Initialize; drug statistics are set in ``train``."""
        super().__init__()
        self.drug_stats: dict[str, tuple[float, float]] = {}
        self.raw_mean: float = 0.0

    @classmethod
    def get_model_name(cls) -> str:
        """:returns: the model name "EnsembleMFStandardized"."""
        return "EnsembleMFStandardized"

    def _stats(self, drug_ids) -> tuple[np.ndarray, np.ndarray]:
        """
        Per-pair mean and scale of the drug's training responses.

        :param drug_ids: drug id per pair
        :returns: (means, scales); drugs without statistics get the overall training mean and scale 1
        """
        s = [self.drug_stats.get(d, (self.raw_mean, 1.0)) for d in drug_ids]
        return np.array([a for a, _ in s]), np.array([b for _, b in s])

    def _standardize(self, data: DrugResponseDataset) -> DrugResponseDataset:
        """
        Return a copy of ``data`` with each drug's responses standardized by the training statistics.

        :param data: response dataset in ln(IC50)
        :returns: copy with standardized responses
        """
        out = data.copy()
        mean, std = self._stats(data.drug_ids)
        out._response = (np.asarray(data.response, dtype=float) - mean) / std
        return out

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
        model_checkpoint_dir: str = "checkpoints",
    ) -> None:
        """
        Fit per-drug statistics on the training responses, then train EnsembleMF on standardized targets.

        :param output: training responses
        :param cell_line_input: cell-line features
        :param drug_input: drug features
        :param output_earlystopping: early-stopping responses
        :param model_checkpoint_dir: unused
        """
        y = np.asarray(output.response, dtype=float)
        drugs = np.asarray(output.drug_ids)
        self.raw_mean = float(np.nanmean(y))
        self.drug_stats = {}
        for d in np.unique(drugs):
            v = y[drugs == d]
            sd = float(np.nanstd(v))
            self.drug_stats[d] = (float(np.nanmean(v)), sd if sd > 1e-6 else 1.0)
        es = self._standardize(output_earlystopping) if output_earlystopping is not None else None
        super().train(self._standardize(output), cell_line_input, drug_input, es, model_checkpoint_dir)

    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        """
        Predict standardized responses with EnsembleMF and map them back to ln(IC50).

        :param cell_line_ids: cell-line ids
        :param drug_ids: drug ids
        :param cell_line_input: cell-line features
        :param drug_input: drug features
        :returns: predicted ln(IC50)
        """
        z = super().predict(cell_line_ids, drug_ids, cell_line_input, drug_input)
        mean, std = self._stats(drug_ids)
        return z * std + mean

    def save(self, directory: str) -> None:
        """
        Save the ensemble and the drug statistics.

        :param directory: target directory
        """
        super().save(directory)
        joblib.dump({"drug_stats": self.drug_stats, "raw_mean": self.raw_mean}, os.path.join(directory, "drug_stats.pkl"))

    @classmethod
    def load(cls, directory: str) -> "EnsembleMFStandardized":
        """
        Restore a model saved with ``save``.

        :param directory: directory with the saved files
        :returns: the restored model
        """
        instance = super().load(directory)
        stats = joblib.load(os.path.join(directory, "drug_stats.pkl"))
        instance.drug_stats, instance.raw_mean = stats["drug_stats"], stats["raw_mean"]
        return instance
