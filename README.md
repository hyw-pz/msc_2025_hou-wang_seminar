# EBM + LASSO Benchmark

A research project replicating and extending the evaluation of LASSO post-processing for **Explainable Boosting Machines (EBMs)**, as described in Greenwell et al. (2023). The codebase covers empirical benchmarks across real-world datasets and a controlled simulation study.

---

## Motivation

Explainable Boosting Machines produce accurate models with built-in interpretability via additive shape functions. However, in high-dimensional settings, a trained EBM can contain hundreds of terms (main effects + pairwise interactions), which undermines the glass-box nature and introduces significant inference latency.

This project applies **LASSO regularisation on the EBM term-contribution matrix** to automatically prune redundant terms — yielding a sparser model while preserving predictive performance.

### Pipeline

```
Raw data
   │
   ▼
EBM fit ──► eval_terms() ──► Term contribution matrix ──► LASSO path
                                                               │
                                         alpha selection (min metric on val fold)
                                                               │
                                         Sparse EBM ◄── ebm.scale() + sweep()
```

The non-negativity constraint (`positive=True`) ensures LASSO coefficients can only scale terms down or remove them entirely — never flip their learned direction.

---

## Notebooks

### `01_als_benchmark.ipynb` — ALS Dataset (Full Walkthrough)

Uses the ALS dataset from *Computer Age Statistical Inference* (Hastie & Efron). Target: `dFRS` (disease progression rate). Reproduces the original paper's train/test split.

Covers:
- Baseline benchmark: XGBoost · LightGBM · RandomForest · LASSO (5-fold CV)
- EBM + LASSO cross-validation
- Full LASSO regularisation path on the designated train/test split
- Sparsity vs. test MSE plot
- Sparse EBM construction via `ebm.scale()` + `ebm.sweep()`
- Visualisations: shape functions, interaction heat-maps, feature importance, reason plots

### `02_multi_dataset_benchmark.ipynb` — Multi-Dataset Comparison

Runs EBM + LASSO against all baseline models across 6 OpenML regression datasets.

| Dataset | OpenML ID | Target | Notes |
|---|---|---|---|
| Superconductivity | 43174 | Critical temperature | Dense, no categoricals |
| Mercedes Benz | 42570 | Test time | Mixed types, one-hot encoded |
| House Prices | 42563 | Sale price | Log-transformed target |
| Topo 2.1 | 422 | `oz267` | |
| Yolanda | 42705 | Target | Subsampled to 5 000 rows |
| Student Performance | 42572 | Grade | |

---

## Simulation Study

The simulation investigates whether LASSO post-processing can recover the true underlying effects in a controlled sparse additive regression setting.

### Data Generating Process

Data are generated from a sparse additive model with 100 covariates: 10 relevant variables, 10 correlated redundant variables (paired with the relevant ones), and 80 independent noise variables. All covariates are truncated to [−1, 1].

**Main effects** include linear, quadratic, exponential, and sinusoidal forms:

$$g_\text{main}(x) = 0.55x_1 - 0.45x_2 + 0.40(x_3 - 0.2)^2 + 0.20e^{-2x_4} + 0.30\sin(\pi x_5) - 0.38x_6^2 + 0.15(x_7 + x_8 + x_9 + x_{10})$$

**Interaction effects** include bilinear, sinusoidal, and Gaussian-shaped interactions:

$$g_\text{int}(x) = x_1 x_2 + x_3 x_4 + x_5 x_6 + 0.5\sin(\pi x_9 x_{10}) + \exp\!\left(-2(x_7+0.2)^2 - \tfrac{1}{2}\!\left(3x_8 + 1.2(x_7+0.2)^2 - 0.4\right)^2\right)$$

The response is $y = g_\text{main}(x) + g_\text{int}(x) + \varepsilon$, $\varepsilon \sim \mathcal{N}(0, 0.4^2)$.

### Experimental Setup

- Sample sizes: $n \in \{1000, 2000, 5000, 10000\}$
- Correlation levels between relevant and redundant variables: $\rho \in \{0, 0.2, 0.5\}$
- 10 repetitions per $(n, \rho)$ configuration
- Data split: 60% train / 20% validation / 20% test
- Alpha selected via LASSO path on the validation set

### Key Results

| $n$ | $\rho$ | RMSE (EBM) | RMSE (EBM+LASSO) | Terms kept $k$ | True main effects recovered | True interactions recovered | Importance on true effects |
|---|---|---|---|---|---|---|---|
| 1000 | 0 | 0.632 | 0.492 | 17.8 | 8.2 / 10 | 4.0 / 5 | 93.2% |
| 2000 | 0 | 0.533 | 0.447 | 20.7 | 9.8 / 10 | 4.6 / 5 | 94.9% |
| 5000 | 0 | 0.470 | 0.425 | 18.6 | 10 / 10 | 5.0 / 5 | 98.4% |
| 10000 | 0 | 0.445 | 0.413 | 18.4 | 10 / 10 | 5.0 / 5 | 99.1% |

LASSO post-processing consistently improves generalisation over the original EBM across all $(n, \rho)$ settings. With $n \geq 5000$, all 10 true main effects and all 5 true interactions are reliably recovered. Even under the strongest correlation ($\rho = 0.5$), structural recovery is not substantially harmed. Non-true terms that are occasionally selected contribute only marginally — roughly 90–99% of total model importance is concentrated on the true effects.

---

## Benchmark Results Summary

| Dataset | Metric | EBM | EBM+LASSO | Best Baseline | Terms retained |
|---|---|---|---|---|---|
| ALS | MSE | 0.265 | 0.271 | RF/GBM 0.255 | 19 / 379 |
| House Price | RMSE | 0.135 | **0.125** | LightGBM 0.132 | 44 / 282 |
| Mercedes-Benz | MSE | 8.475 | 8.362 | RF 8.359 | 34 / 387 |
| Crime | RMSE | 0.131 | 0.135 | LightGBM 0.133 | 36 / 136 |
| QSAR | MSE | 0.550 | 0.574 | XGBoost 0.531 | 351 / 1034 |
| PC4 | AUC | 0.941 | **0.941** | LightGBM 0.948 | 15 / 47 |

EBM+LASSO achieves 41–97% term reduction across all datasets. On House Price it outperforms all baselines; on PC4 it matches the original EBM at 68% compression. The main challenging case is QSAR (sparse binary fingerprints), where limited feature redundancy constrains how aggressively the model can be pruned without accuracy loss.

**Inference speedup**: 6–10× over the full EBM, scaling with the pruning ratio.

---

## Installation

```bash
pip install interpret scikit-learn xgboost lightgbm openml numpy pandas matplotlib scipy
```

Or with pinned versions:

```bash
pip install interpret>=0.6.0 scikit-learn>=1.3.0 xgboost>=2.0.0 lightgbm>=4.0.0 \
            openml>=0.14.0 numpy>=1.24.0 pandas>=2.0.0 matplotlib>=3.7.0 scipy>=1.11.0
```

---

## Key Functions

Both benchmark notebooks are self-contained. The main building blocks:

#### `run_benchmark(X, y, n_splits=5, metric='rmse')`
K-fold CV for XGBoost, LightGBM, RandomForest, and LASSO. Returns a summary DataFrame with mean metric, std, and mean R². Supported metrics: `'mae'`, `'mse'`, `'rmse'`.

#### `ebm_lasso_cross_validation(X, y, inner_bags, outer_bags, interactions, n_splits=5, metric='mae')`
Fits EBM + LASSO post-processing with k-fold CV. Per fold: fit EBM → extract term contributions via `eval_terms()` → sweep LASSO path → select alpha minimising *metric* on the validation fold → record metric, R², and number of retained terms.

#### `run_lasso_poly_optimized(X, y, metric='mse', top_n_features=30, n_splits=5)`
LASSO baseline augmented with pairwise interaction features. Pre-selects the top-*k* features by |corr| with *y* before expanding to avoid feature explosion.

---

## Design Notes

**`positive=True` in LASSO** — EBM term contributions are already signed; constraining LASSO coefficients to be non-negative ensures post-processing only scales terms down, never flips their direction, preserving semantic consistency.

**Path search for alpha selection** — The full `lasso_path` is swept on each training fold and alpha is selected by directly minimising the validation metric, keeping alpha selection consistent with the outer CV loop and avoiding test-set leakage.

**Sparse matrix handling** — `run_benchmark` automatically detects scipy sparse and pandas sparse dtypes and disables `StandardScaler(with_mean=True)` to avoid densification errors.

---

## References

- Greenwell, B., Dahlmann, A., & Dhoble, S. (2023). *Explainable boosting machines with sparsity.* [arXiv:2311.07452](https://arxiv.org/abs/2311.07452)
- Nori, H., Jenkins, S., Koch, P., & Caruana, R. (2019). *InterpretML: A unified framework for machine learning interpretability.* [arXiv:1909.09223](https://arxiv.org/abs/1909.09223)
- Lou, Y., Caruana, R., Hooker, G., & Gehrke, J. (2013). *Accurate intelligible models with pairwise interactions.* KDD.
- Hastie, T. & Efron, B. (2016). *Computer Age Statistical Inference.* Cambridge University Press.
