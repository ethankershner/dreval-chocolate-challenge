# dreval-chocolate-challenge

Two models and their average for the [DrEval](https://drevalpy.readthedocs.io) chocolate challenge. The task is to predict drug response (ln IC50) for unseen cancer cell lines on CTRPv2 (leave-cell-line-out), and to significantly outperform `EnsembleMF`, the best model on the leaderboard.

## Result

On the 10 precomputed leaderboard splits, run and scored with DrEval (normalized Pearson, the challenge metric):

| Model | Score | Splits ahead of EnsembleMF | Conover p vs EnsembleMF |
|---|---|---|---|
| EnsembleMF (leaderboard) | 0.376 | | |
| `PerDrugRidge` | 0.407 | 10 of 10 | 0.003 |
| `MultiTaskMLP` | 0.407 | 9 of 10 | 0.077 |
| **`AverageEnsemble`** | **0.423** | **10 of 10** | **0.003** |

No setting is tuned: the ridge picks its penalty per drug inside the training data, and the network uses conventional defaults with early stopping. `AverageEnsemble`, the unweighted mean of the two, ranks first among the 35 models on every split and significantly outperforms `EnsembleMF`. So does the ridge on its own; the network alone is ahead on 9 of 10 splits but not significantly. The p-values come from the Friedman test with Conover post-hoc and Benjamini-Hochberg correction, with each model added to the published leaderboard on its own.

The notebook explores why:

- **Each drug on its own scale.** Both models fit each drug on its own scale, while `EnsembleMF`'s single loss on raw ln IC50 is dominated by the drugs with the widest spread. `EnsembleMF` is level on those drugs (0.006 behind) and 0.03–0.07 behind on the rest.
- **Ridge works through strong shrinkage.**
- **The default network matches the ridge but errs differently.** With default settings its benefit from sharing across drugs is not clear.
- **Averaging helps because ridge and the network make different errors.** Their predictions correlate 0.85, and the average beats each of them on 9 of 10 splits.

## EnsembleMF ablation

`ensemble_mf_ablation.ipynb` tests the first point directly. `EnsembleMFStandardized` is DrEval's `EnsembleMF` with identical settings, trained on per-drug standardized targets so that every drug weighs the same in its loss.

| | Score |
|---|---|
| EnsembleMF (leaderboard) | 0.376 |
| `EnsembleMFStandardized` | 0.382 (ahead on 8 of 10 splits) |
| `PerDrugRidge` | 0.407 |

- **Drug by drug, the effect follows the hypothesis.** Standardization raises the per-drug correlation by 0.046 on the fifth of drugs with the narrowest spread, and by less as the spread grows. On the widest fifth it lowers it by 0.013.
- **The leaderboard score barely moves.** The metric pools all test pairs, so the widest fifth of drugs carries about half of it. Those are the drugs the raw-scale loss already favours.
- **Most of the gap is unexplained.** The gain of 0.006 is about a fifth of the ridge's lead.
- **No help as a third ensemble member.** Adding it to the unweighted average of ridge and MLP gives 0.421 against 0.423.

## The models

Both models use only the expression of the 867 L1000 landmark genes, arcsinh-transformed and standardized on the training cell lines. None uses drug features. Each has a single fixed configuration, so DrEval does no hyperparameter tuning. The ridge trains on train + validation of every split; the MLP trains on train + 60 validation cell lines and stops early on the other 20, which DrEval holds out for that purpose.

| Model | What it is |
|---|---|
| `PerDrugRidge` | One ridge regression per drug. The penalty is chosen per drug from 14 values by exact leave-one-out error on the training cell lines. |
| `MultiTaskMLP` | One network with an output per drug, trained on per-drug standardized targets with a squared error over measured pairs only. Conventional default settings: two hidden layers of 512 ReLU units, dropout 0.2, Adam (learning rate 0.001, no weight decay), batch size 64, early stopping with patience 10. 10 seeds are averaged. |
| `AverageEnsemble` | Trains `PerDrugRidge` and `MultiTaskMLP` and returns the unweighted mean of their predictions. |

## Contents

| Path | What it is |
|---|---|
| `chocolate_challenge.ipynb` | Entry point: data, the models, the DrEval run, performance, and why the models beat `EnsembleMF` |
| `chocolate/per_drug_ridge.py`, `multitask_mlp.py`, `ensemble.py` | The three models as DrEval `DRPModel` classes |
| `chocolate/hyperparameters.yaml` | Their configurations, one per model |
| `chocolate/features.py` | The shared input: landmark-gene expression |
| `chocolate/run.py` | Runs DrEval's `drug_response_experiment` on the leaderboard splits and DrEval's `create_report` |
| `chocolate/analysis.py` | A scorer that reproduces DrEval's normalized Pearson exactly, plus the experiments in section 5 |
| `chocolate/plots.py` | Longer figure code |
| `ensemble_mf_ablation.ipynb` | The EnsembleMF ablation: per-drug standardized targets |
| `chocolate/ensemble_mf_standardized.py` | `EnsembleMFStandardized`, DrEval's `EnsembleMF` on per-drug standardized targets |
| `chocolate/ensemble_mf_ablation.py` | Trains it on the leaderboard splits (several splits at once) and scores model combinations exactly as DrEval does |
| `reference/leaderboard_evaluation_results.csv` | Published leaderboard scores (from the DrEval Zenodo record) |

DrEval is used unmodified from PyPI (`drevalpy==1.7.0`). The models are passed to DrEval as classes, so DrEval's source is not changed.

## Running it

```bash
uv venv --python 3.13 .venv
```

```bash
uv pip install --python .venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cu128
```

```bash
uv pip install --python .venv/Scripts/python.exe -e .
```

Then:

1. **Splits:** download `splits.zip` from the [DrEval Zenodo record](https://doi.org/10.5281/zenodo.12633909) and unzip it so the CSV files sit in `reference/splits/`.
2. **Published EnsembleMF predictions** (section 5 of the notebook): from the leaderboard archive on the same record, take `true_vs_pred.csv`. Keep the rows with `algorithm == "EnsembleMF"` and `rand_setting == "predictions"`, and save them as `reference/leaderboard_EnsembleMF_true_vs_pred.csv`.
3. **Data:** DrEval downloads CTRPv2 into `data/` on first use. Check that `data/meta/gene_lists/landmark_genes.csv` lists about 870 genes (see the notes below).
4. **Notebook:** open `chocolate_challenge.ipynb` and run all cells.

The DrEval run takes about 15 minutes on one GPU. It is skipped if `results/final` already holds predictions of every model for all 10 splits. The experiments in section 5 are cached in `results/analysis/`; recomputing them takes a few minutes.

5. **Ablation:** after the main notebook has run, open `ensemble_mf_ablation.ipynb` and run all cells. It needs the published EnsembleMF predictions from step 2. Training takes about 30 minutes on one GPU, running 3 splits at once. It is skipped for every split already stored in `results/emf_standardized`.

## Notes for the organisers

- **Pair counting in normalized metrics.** DrEval joins predictions to the `NaiveMeanEffectsPredictor` reference on (drug, cell line), so a pair listed twice in the data is counted four times. This affects every model equally. The notebook's scorer reproduces it, so its numbers match DrEval's report exactly.
- **Gene lists.** The `meta.zip` on Zenodo dated 2026-11-03 has truncated gene lists (`landmark_genes` has 13 genes). This repo used the full lists from the earlier version.
- **Per-drug models passed in from outside.** DrEval decides where a single-drug model's results go by looking its name up in `SINGLE_DRUG_MODEL_FACTORY`, not by `is_single_drug_model`. A per-drug model passed to `drug_response_experiment` as a class, without being in that dictionary, makes every drug overwrite the same prediction file. The per-drug ridge here avoids this by being an ordinary model that fits its drugs internally.
