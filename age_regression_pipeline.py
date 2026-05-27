#!/usr/bin/env python3
"""
Age regression from PPG features.
Models: Ridge, Random Forest, Gradient Boosting, FT-Transformer (Gorishniy et al., NeurIPS 2021).
Subgroup sensitivity analysis with forest plots (MAE and Pearson r, 95% bootstrap CI).

FT-Transformer note (thesis context):
  Each of the 131 PPG features is treated as a "token" (like a word in NLP).
  The Feature Tokenizer learns a per-feature embedding (analogous to word embeddings),
  and the Transformer Encoder learns cross-feature contextual relationships -
  an alternative representation of tabular numerical data.
"""
import os, math, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')

# -- Configuration --------------------------------------------------------------
DATA_PATH = r'C:\Users\adamo\OneDrive - Politechnika Śląska\Downloads\P1_data_age_regression.xlsx'
OUT_DIR   = r'C:\Users\adamo\OneDrive - Politechnika Śląska\Downloads\age_regression_results'
SEED      = 42
N_BOOT    = 1000
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'

os.makedirs(OUT_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"Device : {DEVICE}")
print(f"Output : {OUT_DIR}\n")

# ==============================================================================
# 1  Load & clean
# ==============================================================================
print("[1] Loading data ...")
df = pd.read_excel(DATA_PATH)
print(f"  Raw            : {df.shape}")

df = df[df['age'] <= 90].copy()
print(f"  After age <= 90 : {df.shape}")

ppg_cols = df.columns[78:209].tolist()   # 1_nni_mean ... 1_IPAD  (131 features)
print(f"  PPG features   : {len(ppg_cols)}  ({ppg_cols[0]} ... {ppg_cols[-1]})")

df = df.dropna(subset=ppg_cols).copy().reset_index(drop=True)
print(f"  After PPG dropna: {df.shape}  |  unique subjects: {df['subject_id'].nunique()}")

# QC: remove rows with any PPG feature > 100x IQR beyond quartiles (signal artifacts)
X_raw = df[ppg_cols].values.astype(np.float64)
artifact_mask = np.zeros(len(X_raw), dtype=bool)
for i in range(X_raw.shape[1]):
    col = X_raw[:, i]
    q1, q3 = np.percentile(col, 25), np.percentile(col, 75)
    iqr    = max(q3 - q1, 1e-10)
    artifact_mask |= (col < q1 - 100 * iqr) | (col > q3 + 100 * iqr)
df = df[~artifact_mask].copy().reset_index(drop=True)
print(f"  After QC (>100xIQR): {df.shape}  |  removed {artifact_mask.sum()} artifact rows\n")

X      = df[ppg_cols].values.astype(np.float64)
y      = df['age'].values.astype(np.float64)
groups = df['subject_id'].values

# ==============================================================================
# 2  Group-aware train / test split  (no subject leakage)
# ==============================================================================
print("[2] Splitting data ...")
gss = GroupShuffleSplit(1, test_size=0.2, random_state=SEED)
train_idx, test_idx = next(gss.split(X, y, groups=groups))
assert not (set(groups[train_idx]) & set(groups[test_idx])), "DATA LEAKAGE DETECTED!"

X_tr, X_te = X[train_idx], X[test_idx]
y_tr, y_te = y[train_idx], y[test_idx]
g_tr       = groups[train_idx]
df_te      = df.iloc[test_idx].copy()

print(f"  Train : {len(train_idx):4d} rows  |  {len(np.unique(g_tr))} subjects")
print(f"  Test  : {len(test_idx):4d} rows  |  {len(np.unique(groups[test_idx]))} subjects\n")

# ==============================================================================
# 3  Sklearn models  (GroupKFold-5 CV)
# ==============================================================================
print("[3] Training sklearn models ...")
sk_models = {
    'Ridge (PCA)': Pipeline([
        ('sc',  StandardScaler()),
        ('pca', PCA(n_components=0.95)),
        ('m',   Ridge(alpha=1.0))
    ]),
    'RandomForest': Pipeline([
        ('sc', StandardScaler()),
        ('m',  RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1))
    ]),
    'GradBoost': Pipeline([
        ('sc', StandardScaler()),
        ('m',  GradientBoostingRegressor(n_estimators=300, max_depth=4, random_state=SEED))
    ]),
}
gkf    = GroupKFold(5)
cv_maes = {}
for name, pipe in sk_models.items():
    # Ridge (PCA): n_jobs=1 to avoid Windows multiprocessing issues with linear models
    n_jobs_cv = 1 if 'Ridge' in name else -1
    sc = -cross_val_score(pipe, X_tr, y_tr, groups=g_tr,
                          cv=gkf, scoring='neg_mean_absolute_error', n_jobs=n_jobs_cv)
    cv_maes[name] = sc
    print(f"  {name:15s}: CV MAE = {sc.mean():.2f} +/- {sc.std():.2f}")
    pipe.fit(X_tr, y_tr)
print()

# ==============================================================================
# 4  FT-Transformer  (PyTorch)
# ==============================================================================
print("[4] Training FT-Transformer ...")

# Split training set -> subtrain (75%) + early-stop val (25%), still group-safe
gss2 = GroupShuffleSplit(1, test_size=0.25, random_state=SEED)
s_idx, v_idx = next(gss2.split(X_tr, y_tr, groups=g_tr))
X_s, X_v = X_tr[s_idx], X_tr[v_idx]
y_s, y_v = y_tr[s_idx], y_tr[v_idx]

scaler_ft = StandardScaler().fit(X_s)
Xs_n  = scaler_ft.transform(X_s).astype(np.float32)
Xv_n  = scaler_ft.transform(X_v).astype(np.float32)
Xte_n = scaler_ft.transform(X_te).astype(np.float32)

y_mu, y_sig = float(y_s.mean()), float(y_s.std())
ys_n = ((y_s - y_mu) / y_sig).astype(np.float32)
yv_n = ((y_v - y_mu) / y_sig).astype(np.float32)

BATCH  = 256
tr_dl  = DataLoader(TensorDataset(torch.tensor(Xs_n), torch.tensor(ys_n)),
                    batch_size=BATCH, shuffle=True, drop_last=True)
va_dl  = DataLoader(TensorDataset(torch.tensor(Xv_n), torch.tensor(yv_n)),
                    batch_size=BATCH, shuffle=False)


class FeatureTokenizer(nn.Module):
    """Maps each numerical feature x_i to a d-dimensional token: x_i * W_i + b_i.
    Analogous to word-embedding lookup in NLP, but for continuous features."""
    def __init__(self, n_features: int, d: int):
        super().__init__()
        self.W = nn.Parameter(torch.empty(n_features, d))
        self.b = nn.Parameter(torch.zeros(n_features, d))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, n]  ->  [B, n, 1] * [n, d] + [n, d]  ->  [B, n, d]
        return x.unsqueeze(-1) * self.W + self.b


class FTTransformer(nn.Module):
    """Feature Tokenizer + Transformer (Gorishniy et al., NeurIPS 2021).
    Each PPG feature becomes a token; the Transformer learns cross-feature
    contextual relationships. A [CLS] token aggregates the representation."""
    def __init__(self, n_features: int, d: int = 192, heads: int = 8,
                 layers: int = 3, dropout: float = 0.1):
        super().__init__()
        ffn_d = max(int(d * 4 / 3), d)
        self.tok  = FeatureTokenizer(n_features, d)
        self.cls  = nn.Parameter(torch.zeros(1, 1, d))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=ffn_d,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.enc  = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = torch.cat([self.cls.expand(x.size(0), -1, -1),
                            self.tok(x)], dim=1)          # [B, n+1, d]
        out = self.norm(self.enc(tokens)[:, 0])           # [CLS] output
        return self.head(out).squeeze(-1)                 # [B]


net     = FTTransformer(n_features=X_s.shape[1]).to(DEVICE)
opt     = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-5)
sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
loss_fn = nn.MSELoss()

best_val_mae = np.inf
best_state   = None
patience, no_imp = 20, 0

for epoch in range(1, 201):
    net.train()
    for Xb, yb in tr_dl:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        loss_fn(net(Xb), yb).backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    sched.step()

    net.eval()
    with torch.no_grad():
        pv = torch.cat([net(Xb.to(DEVICE)).cpu()
                        for Xb, _ in va_dl]).numpy()
    vm = mean_absolute_error(y_v, pv * y_sig + y_mu)

    if vm < best_val_mae:
        best_val_mae = vm
        best_state   = {k: v.clone() for k, v in net.state_dict().items()}
        no_imp = 0
    else:
        no_imp += 1

    if epoch % 20 == 0:
        print(f"  Epoch {epoch:3d}  val_MAE={vm:.2f}")
    if no_imp >= patience:
        print(f"  Early stop at epoch {epoch}")
        break

net.load_state_dict(best_state)
print(f"  Best val MAE: {best_val_mae:.2f}\n")

net.eval()
with torch.no_grad():
    chunks = [net(torch.tensor(Xte_n[i:i+BATCH]).to(DEVICE)).cpu().numpy()
              for i in range(0, len(Xte_n), BATCH)]
pred_ft = np.concatenate(chunks) * y_sig + y_mu
cv_maes['FT-Transformer'] = np.array([best_val_mae])

# ==============================================================================
# 5  Test-set evaluation
# ==============================================================================
print("[5] Test-set performance:")
all_preds = {**{n: p.predict(X_te) for n, p in sk_models.items()},
             'FT-Transformer': pred_ft}
results = {}
for name, pred in all_preds.items():
    mae  = mean_absolute_error(y_te, pred)
    rmse = np.sqrt(mean_squared_error(y_te, pred))
    r2   = r2_score(y_te, pred)
    r, p = stats.pearsonr(y_te, pred)
    results[name] = dict(MAE=mae, RMSE=rmse, R2=r2, r=r, p=p)
    print(f"  {name:20s}  MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.3f}  r={r:.3f}")

best_name = min(results, key=lambda n: results[n]['MAE'])
y_pred    = all_preds[best_name]
print(f"\n  -> Best model: {best_name}  (MAE={results[best_name]['MAE']:.2f})\n")

# ==============================================================================
# 6  Hyperparameter Tuning  (RandomizedSearchCV + GroupKFold)
# ==============================================================================
print("\n[6] Hyperparameter tuning ...")
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor

print("  Tuning RandomForest  (n_iter=15, cv=3) ...")
rf_search = RandomizedSearchCV(
    Pipeline([('sc', StandardScaler()),
              ('m',  RandomForestRegressor(random_state=SEED, n_jobs=-1))]),
    {'m__n_estimators':     [200, 300, 400],
     'm__max_features':     ['sqrt', 'log2', 0.25, 0.35],
     'm__min_samples_leaf': [1, 2, 3, 5],
     'm__max_depth':        [None, 20, 30]},
    n_iter=15, scoring='neg_mean_absolute_error',
    cv=GroupKFold(3), random_state=SEED, refit=True, n_jobs=1,
)
rf_search.fit(X_tr, y_tr, groups=g_tr)
print(f"  RF tuned  CV MAE: {-rf_search.best_score_:.2f}")
print(f"  Best params: { {k[3:]: v for k, v in rf_search.best_params_.items()} }")

print("  Tuning HistGradientBoosting  (n_iter=20, cv=3) ...")
hgb_search = RandomizedSearchCV(
    Pipeline([('sc', StandardScaler()),
              ('m',  HistGradientBoostingRegressor(random_state=SEED))]),
    {'m__max_iter':          [200, 300, 500],
     'm__max_depth':         [None, 5, 8, 12],
     'm__learning_rate':     [0.03, 0.05, 0.1, 0.15],
     'm__l2_regularization': [0.0, 0.1, 1.0, 5.0],
     'm__min_samples_leaf':  [10, 20, 30, 50],
     'm__max_leaf_nodes':    [31, 63, 127]},
    n_iter=20, scoring='neg_mean_absolute_error',
    cv=GroupKFold(3), random_state=SEED, refit=True, n_jobs=1,
)
hgb_search.fit(X_tr, y_tr, groups=g_tr)
print(f"  HGB tuned CV MAE: {-hgb_search.best_score_:.2f}")
print(f"  Best params: { {k[3:]: v for k, v in hgb_search.best_params_.items()} }")

# ==============================================================================
# 7  Stacking Ensemble
# ==============================================================================
print("\n[7] Stacking ensemble ...")

# FT-Transformer OOF on val set (va_dl still in scope from section 4)
net.eval()
with torch.no_grad():
    oof_ft = torch.cat([net(Xb.to(DEVICE)).cpu()
                        for Xb, _ in va_dl]).numpy() * y_sig + y_mu

stack_pipes = {
    'Ridge (PCA)':    sk_models['Ridge (PCA)'],
    'RF (tuned)':     rf_search.best_estimator_,
    'HistGB (tuned)': hgb_search.best_estimator_,
}
oof_meta = {}
for name, pipe in stack_pipes.items():
    oof_meta[name] = cross_val_predict(
        pipe, X_tr, y_tr, groups=g_tr, cv=GroupKFold(5), n_jobs=1)
    print(f"  OOF {name:20s}: MAE={mean_absolute_error(y_tr, oof_meta[name]):.2f}")

# Stack-3: Ridge + RF(tuned) + HGB(tuned) — meta trained on full train OOF
meta_Xtr_3  = np.column_stack([oof_meta[n] for n in stack_pipes])
meta_Xte_3  = np.column_stack([stack_pipes[n].predict(X_te) for n in stack_pipes])
meta_sc3    = StandardScaler().fit(meta_Xtr_3)
stack_lr3   = Ridge(alpha=0.5)
stack_lr3.fit(meta_sc3.transform(meta_Xtr_3), y_tr)
stack_pred3 = np.clip(stack_lr3.predict(meta_sc3.transform(meta_Xte_3)),
                      float(y.min()), float(y.max()))

# Stack-4: +FT-Transformer (OOF only on val subset, ~25% of train)
ft_oof_full        = np.full(len(y_tr), np.nan)
ft_oof_full[v_idx] = oof_ft
val4_mask          = ~np.isnan(ft_oof_full)
meta_Xtr_4 = np.column_stack([oof_meta[n][val4_mask] for n in stack_pipes]
                              + [ft_oof_full[val4_mask]])
y_tr_4     = y_tr[val4_mask]
meta_Xte_4 = np.column_stack([stack_pipes[n].predict(X_te) for n in stack_pipes]
                              + [pred_ft])
meta_sc4   = StandardScaler().fit(meta_Xtr_4)
stack_lr4  = Ridge(alpha=0.5)
stack_lr4.fit(meta_sc4.transform(meta_Xtr_4), y_tr_4)
stack_pred4 = np.clip(stack_lr4.predict(meta_sc4.transform(meta_Xte_4)),
                      float(y.min()), float(y.max()))

mae_s3, r_s3 = mean_absolute_error(y_te, stack_pred3), stats.pearsonr(y_te, stack_pred3)[0]
mae_s4, r_s4 = mean_absolute_error(y_te, stack_pred4), stats.pearsonr(y_te, stack_pred4)[0]
print(f"  Stack-3 (Ridge+RF+HGB):    MAE={mae_s3:.2f}  r={r_s3:.3f}")
print(f"  Stack-4 (+FT-Transformer): MAE={mae_s4:.2f}  r={r_s4:.3f}")
print(f"  Meta-coefs (Stack-3): {list(zip(stack_pipes.keys(), stack_lr3.coef_.round(3)))}")

# ==============================================================================
# 8  Collect all 8 models + compute unified metrics
# ==============================================================================
ALL_MODELS = {
    'Ridge (PCA)':         all_preds['Ridge (PCA)'],
    'RandomForest':        all_preds['RandomForest'],
    'GradBoost':           all_preds['GradBoost'],
    'FT-Transformer':      pred_ft,
    'RF (tuned)':          rf_search.best_estimator_.predict(X_te),
    'HistGB (tuned)':      hgb_search.best_estimator_.predict(X_te),
    'Stack-3 (R+RF+HGB)':  stack_pred3,
    'Stack-4 (+FT)':       stack_pred4,
}
MODEL_COLORS = {
    'Ridge (PCA)':         '#4c72b0',
    'RandomForest':        '#dd8452',
    'GradBoost':           '#55a868',
    'FT-Transformer':      '#c44e52',
    'RF (tuned)':          '#9467bd',
    'HistGB (tuned)':      '#8c564b',
    'Stack-3 (R+RF+HGB)':  '#e377c2',
    'Stack-4 (+FT)':       '#17becf',
}

ALL_RESULTS = {}
print("\n[8] All models – test set:")
print(f"  {'Model':28s}  {'MAE':>6}  {'RMSE':>7}  {'R2':>7}  {'r':>7}")
print("  " + "-" * 58)
for name, pred in ALL_MODELS.items():
    mae    = mean_absolute_error(y_te, pred)
    rmse   = float(np.sqrt(mean_squared_error(y_te, pred)))
    r2     = r2_score(y_te, pred)
    r_v, _ = stats.pearsonr(y_te, pred)
    ALL_RESULTS[name] = dict(MAE=mae, RMSE=rmse, R2=r2, r=r_v)
    print(f"  {name:28s}  {mae:6.2f}  {rmse:7.2f}  {r2:7.3f}  {r_v:7.3f}")

BEST         = min(ALL_RESULTS, key=lambda n: ALL_RESULTS[n]['MAE'])
lim_global   = [float(y_te.min()) - 1, float(y_te.max()) + 1]
xs_global    = np.linspace(*lim_global, 200)
print(f"\n  -> Best: {BEST}  (MAE={ALL_RESULTS[BEST]['MAE']:.2f})\n")

# ==============================================================================
# 9  Feature importance  (best tree-based model)
# ==============================================================================
_pipe_map = {
    'GradBoost':   sk_models['GradBoost'],
    'RandomForest': sk_models['RandomForest'],
    'RF (tuned)':  rf_search.best_estimator_,
}
tree_name = min(_pipe_map, key=lambda n: ALL_RESULTS[n]['MAE'])
tree_pipe = _pipe_map[tree_name]
imps = tree_pipe.named_steps['m'].feature_importances_
top  = np.argsort(imps)[::-1][:30]
fig, ax = plt.subplots(figsize=(10, 9))
ax.barh(range(30), imps[top[::-1]], color='steelblue', alpha=0.8)
ax.set_yticks(range(30))
ax.set_yticklabels([ppg_cols[i] for i in top[::-1]], fontsize=9)
ax.set_xlabel('Feature importance')
ax.set_title(f'Top 30 PPG Features – {tree_name}', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved feature_importance.png")

# ==============================================================================
# Helper functions
# ==============================================================================
rng = np.random.default_rng(SEED)


def bootstrap_metrics(yt, yp, n_boot=N_BOOT):
    n = len(yt)
    if n < 10:
        return None
    maes, rs, mes = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt_b, yp_b = yt[idx], yp[idx]
        maes.append(mean_absolute_error(yt_b, yp_b))
        mes.append(float(np.mean(yp_b - yt_b)))
        if yt_b.std() > 0:
            rs.append(stats.pearsonr(yt_b, yp_b)[0])
    r_val = stats.pearsonr(yt, yp)[0] if yt.std() > 0 else np.nan
    return dict(
        n=n, mean_age=float(yt.mean()),
        mae=mean_absolute_error(yt, yp),
        mae_lo=float(np.percentile(maes, 2.5)),
        mae_hi=float(np.percentile(maes, 97.5)),
        r=r_val,
        r_lo=float(np.percentile(rs, 2.5))  if rs else np.nan,
        r_hi=float(np.percentile(rs, 97.5)) if rs else np.nan,
        me=float(np.mean(yp - yt)),
        me_lo=float(np.percentile(mes, 2.5)),
        me_hi=float(np.percentile(mes, 97.5)),
    )


IF_NAMES = {
    'IF_myocardial_infarction':                'Myocardial infarction',
    'IF_chronic_ischemic_heart_disease':        'Chronic ischemic heart disease',
    'IF_hypertension':                          'Hypertension',
    'IF_congestive_heart_failure':              'Congestive heart failure',
    'IF_peripheral_vascular_disease':           'Peripheral vascular disease',
    'IF_cerebrovascular_disease':               'Cerebrovascular disease',
    'IF_dementia':                              'Dementia',
    'IF_chronic_pulmonary_disease':             'Chronic pulmonary disease',
    'IF_connective_tissue_disease':             'Connective tissue disease',
    'IF_peptic_ulcer_disease':                  'Peptic ulcer disease',
    'IF_mild_liver_disease':                    'Mild liver disease',
    'IF_diabetes_without_chronic_complication': 'Diabetes (no comp.)',
    'IF_diabetes_with_chronic_complication':    'Diabetes (with comp.)',
    'IF_diabetes_any':                          'Diabetes (any)',
    'IF_hemiplegia_or_paraplegia':              'Hemiplegia / paraplegia',
    'IF_renal_disease':                         'Renal disease',
    'IF_any_malignancy_excluding_skin':         'Malignancy (excl. skin)',
    'IF_moderate_or_severe_liver_disease':      'Mod./severe liver disease',
    'IF_metastatic_solid_tumor':                'Metastatic solid tumor',
    'IF_aids_hiv':                              'AIDS / HIV',
}


def build_rows(df_t, yt, yp):
    rows = []

    def H(label):
        rows.append({'label': label, 'is_hdr': True})

    def add(mask, label):
        m = np.asarray(mask, dtype=bool)
        if m.sum() < 10:
            return
        res = bootstrap_metrics(yt[m], yp[m])
        if res:
            rows.append({'label': label, 'is_hdr': False, **res})

    H('Sex')
    add(df_t['GENDER'] == 'M', '  Male')
    add(df_t['GENDER'] == 'F', '  Female')
    H('Ethnicity')
    for eth in ['White', 'Black', 'Hispanic/Latino', 'Asian', 'Other']:
        if eth in df_t['ethnicity_group'].values:
            add(df_t['ethnicity_group'] == eth, f'  {eth}')
    H('Heart rate (bpm)')
    hr = df_t['HR_bpm']
    for lo, hi, lab in [(0, 60, '<60'), (60, 80, '60-80'),
                        (80, 100, '80-100'), (100, 9999, '>=100')]:
        msk = (hr >= lo) & (hr < hi) if hi < 9999 else (hr >= lo)
        add(msk.fillna(False), f'  {lab}')
    H('BMI (kg/m2)')
    bmi = df_t['BMI'].where(df_t['BMI'].between(10, 80))
    for lo, hi, lab in [(0, 25, '<25'), (25, 30, '25-30'), (30, 9999, '>=30')]:
        msk = (bmi >= lo) & (bmi < hi) if hi < 9999 else (bmi >= lo)
        add(msk.fillna(False), f'  {lab}')
    for col, hdr, lab0, lab1 in [
        ('has_AF',            'Atrial fibrillation',   '  No AF',         '  AF'),
        ('has_shock',         'Shock',                 '  No shock',      '  Shock'),
        ('died_in_hospital',  'In-hospital mortality', '  Survived',      '  Died'),
        ('drugs_exclusion',   'Drug exclusion',        '  No exclusion',  '  Excluded'),
        ('has_any_arrhythmia','Any arrhythmia',        '  No arrhythmia', '  Arrhythmia'),
    ]:
        H(hdr)
        cv = df_t[col]
        v0 = False if cv.dtype == bool else 0
        v1 = True  if cv.dtype == bool else 1
        add(cv == v0, lab0)
        add(cv == v1, lab1)
    H('Risk score (Charlson)')
    rs = df_t['RISK_SCORE']
    for lo, hi, lab in [(0, 2, '0-1'), (2, 4, '2-3'), (4, 6, '4-5'), (6, 999, '>=6')]:
        msk = (rs >= lo) & (rs < hi) if hi < 999 else (rs >= lo)
        add(msk, f'  {lab}')
    H('ICD comorbidities (Charlson index)')
    for col, cname in IF_NAMES.items():
        if col not in df_t.columns:
            continue
        H(f'  ICD {cname}')
        cv = df_t[col]
        add(cv == 0, f'    No {cname}')
        add(cv == 1, f'    {cname}')
    return rows


def forest_plot(rows, overall, mk, lo_k, hi_k, xlabel, fname, model_label=''):
    data_rows = [r for r in rows if not r['is_hdr'] and mk in r
                 and r.get(mk) is not None and not np.isnan(r.get(mk, np.nan))]
    if not data_rows:
        print(f"  Skipping {fname} (no valid data)")
        return
    vals = ([r[mk]   for r in data_rows]
          + [r[lo_k] for r in data_rows]
          + [r[hi_k] for r in data_rows])
    vals = [v for v in vals if v is not None and not np.isnan(v)]
    ov   = overall[mk]
    pad  = max((max(vals) - min(vals)) * 0.08, 0.1)
    if mk == 'mae':
        xlo = max(0,    min(vals) - pad)
        xhi = min(999,  max(vals) + pad)
        xlo, xhi = min(xlo, ov * 0.92), max(xhi, ov * 1.08)
    elif mk == 'me':
        xlo = min(vals) - pad
        xhi = max(vals) + pad
        xlo = min(xlo, ov - pad, -pad)   # always include 0 on the left
        xhi = max(xhi, ov + pad,  pad)   # always include 0 on the right
    else:   # pearson r
        xlo = max(-0.3, min(vals) - pad)
        xhi = min(1.05, max(vals) + pad)
        xlo, xhi = min(xlo, ov * 0.92), max(xhi, ov * 1.08)
    nr    = len(rows) + 3
    fig_h = max(14, nr * 0.21)
    fig   = plt.figure(figsize=(15, fig_h))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.03, wspace=0.01)
    ax_l = fig.add_subplot(1, 3, 1)
    ax_c = fig.add_subplot(1, 3, 2)
    ax_r = fig.add_subplot(1, 3, 3)
    for ax in (ax_l, ax_r):
        ax.set_xlim(0, 1); ax.set_ylim(-0.5, nr - 0.5); ax.invert_yaxis()
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])
    ax_c.set_xlim(xlo, xhi); ax_c.set_ylim(-0.5, nr - 0.5); ax_c.invert_yaxis()
    ax_c.spines['top'].set_visible(False); ax_c.spines['right'].set_visible(False)
    ax_c.spines['left'].set_visible(False); ax_c.set_yticks([])
    ax_c.tick_params(axis='x', labelsize=8)
    ax_c.set_xlabel(xlabel, fontsize=9)
    ax_c.axvline(ov, color='#cc0000', ls='--', lw=1.2, alpha=0.75, zorder=1)
    if mk == 'me':   # extra zero line: no-bias reference
        ax_c.axvline(0, color='#444444', ls=':', lw=1.0, alpha=0.55, zorder=1)
    ax_l.text(0.01, 0, 'Subgroup',  va='center', fontsize=9, fontweight='bold', zorder=5)
    ax_l.text(0.77, 0, 'N',         va='center', ha='center', fontsize=9, fontweight='bold', zorder=5)
    ax_l.text(0.93, 0, 'Mean age',  va='center', ha='center', fontsize=8.5, fontweight='bold', zorder=5)
    ax_r.text(0.5,  0, f'{xlabel} [95% CI]',
              va='center', ha='center', fontsize=8.5, fontweight='bold', zorder=5)
    ov_lo, ov_hi = overall[lo_k], overall[hi_k]
    ax_l.text(0.01, 1, 'Overall', va='center', fontsize=9, fontweight='bold', zorder=5)
    ax_l.text(0.77, 1, str(overall['n']),        va='center', ha='center', fontsize=8.5, zorder=5)
    ax_l.text(0.93, 1, f"{overall['mean_age']:.1f}", va='center', ha='center', fontsize=8.5, zorder=5)
    ax_c.plot([ov_lo, ov_hi], [1, 1], color='#1a1a1a', lw=2.5, solid_capstyle='butt', zorder=2)
    ax_c.plot(ov, 1, 's', color='#1a1a1a', ms=8, zorder=3)
    ax_r.text(0.5, 1, f'{ov:.3f} [{ov_lo:.3f}-{ov_hi:.3f}]',
              va='center', ha='center', fontsize=8, zorder=5)
    DOT_COLOR = '#2c7bb6'
    for ri, r in enumerate(rows):
        y_row = ri + 2
        bg    = '#f4f4f4' if ri % 2 == 0 else 'white'
        for ax in (ax_l, ax_c, ax_r):
            ax.axhspan(y_row - 0.47, y_row + 0.47, color=bg, zorder=0, alpha=1.0)
        lbl = r['label']
        if r['is_hdr']:
            ax_l.text(0.01, y_row, lbl, va='center', fontsize=8.5, fontweight='bold', zorder=5)
            continue
        ax_l.text(0.01, y_row, lbl, va='center', fontsize=8, zorder=5)
        if r.get('n'):
            ax_l.text(0.77, y_row, str(r['n']),   va='center', ha='center', fontsize=8, zorder=5)
        if r.get('mean_age'):
            ax_l.text(0.93, y_row, f"{r['mean_age']:.1f}", va='center', ha='center', fontsize=8, zorder=5)
        mv, lv, hv = r.get(mk), r.get(lo_k), r.get(hi_k)
        if mv is None or np.isnan(mv):
            continue
        lo_p = max(lv, xlo) if lv is not None and not np.isnan(lv) else mv
        hi_p = min(hv, xhi) if hv is not None and not np.isnan(hv) else mv
        ax_c.plot([lo_p, hi_p], [y_row, y_row], color=DOT_COLOR, lw=1.5,
                  solid_capstyle='butt', zorder=2)
        ax_c.plot(mv, y_row, 'o', color=DOT_COLOR, ms=5, zorder=3)
        ci_str = (f'{mv:.3f} [{lv:.3f}-{hv:.3f}]'
                  if lv is not None and not np.isnan(lv) else f'{mv:.3f}')
        ax_r.text(0.5, y_row, ci_str, va='center', ha='center', fontsize=7.5, zorder=5)
    fig.suptitle(
        f'Age Regression – {xlabel} by Subgroup\n'
        f'Model: {model_label}  |  Test set N={overall["n"]}',
        fontsize=10, fontweight='bold',
    )
    plt.savefig(f'{OUT_DIR}/{fname}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {fname}")


# ==============================================================================
# 10  Per-model: scatter + residuals  (one figure per model)
# ==============================================================================
print("[10] Per-model scatter + residuals ...")


def _slug(name):
    return (name.replace(' ', '_').replace('(', '').replace(')', '')
                .replace('/', '_').replace('+', 'p'))


for name, pred in ALL_MODELS.items():
    clr = MODEL_COLORS[name]
    rv  = ALL_RESULTS[name]
    res = pred - y_te

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(f'Age Prediction – {name}', fontsize=12, fontweight='bold')

    # Left: scatter
    ax1.scatter(y_te, pred, alpha=0.28, s=12, color=clr, zorder=2)
    ax1.plot(lim_global, lim_global, 'r--', lw=1.5, label='Identity (perfect)', zorder=3)
    sl, ic, *_ = stats.linregress(y_te, pred)
    ax1.plot(xs_global, sl * xs_global + ic, 'k-', lw=1.5, label='Regression fit', zorder=4)
    ax1.set_xlim(lim_global); ax1.set_ylim(lim_global)
    ax1.set_xlabel('Actual age [years]'); ax1.set_ylabel('Predicted age [years]')
    ax1.set_title('Predicted vs Actual')
    ax1.legend(fontsize=9, loc='lower right')
    ax1.text(0.04, 0.97,
             f"r = {rv['r']:.3f}\nMAE = {rv['MAE']:.2f} yr"
             f"\nRMSE = {rv['RMSE']:.2f} yr\nR² = {rv['R2']:.3f}",
             transform=ax1.transAxes, va='top', fontsize=9,
             bbox=dict(boxstyle='round', fc='white', alpha=0.88))

    # Right: residuals
    ax2.scatter(y_te, res, alpha=0.28, s=12, color=clr, zorder=2)
    ax2.axhline(0, color='red', ls='--', lw=1.5, label='Zero error', zorder=3)
    sl_r, ic_r, *_ = stats.linregress(y_te, res)
    ax2.plot(xs_global, sl_r * xs_global + ic_r, 'k-', lw=1.5, label='Trend', zorder=4)
    ax2.set_xlim(lim_global)
    ax2.set_xlabel('Actual age [years]'); ax2.set_ylabel('Residual (predicted – actual) [years]')
    ax2.set_title('Residuals vs Actual Age')
    ax2.legend(fontsize=9, loc='upper right')
    ax2.text(0.04, 0.97,
             f"Mean error: {res.mean():+.2f} yr\nStd: {res.std():.2f} yr"
             f"\nTrend slope: {sl_r:+.3f}",
             transform=ax2.transAxes, va='top', fontsize=9,
             bbox=dict(boxstyle='round', fc='white', alpha=0.88))

    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/{_slug(name)}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {_slug(name)}.png")

# ==============================================================================
# 11  Per-model: forest plots  (MAE and Pearson r)
# ==============================================================================
print("\n[11] Per-model forest plots ...")
for name, pred in ALL_MODELS.items():
    slug    = _slug(name)
    overall = bootstrap_metrics(y_te, pred)
    rows    = build_rows(df_te, y_te, pred)
    print(f"  {name}: MAE={overall['mae']:.2f} [{overall['mae_lo']:.2f}-{overall['mae_hi']:.2f}]"
          f"  ME={overall['me']:+.2f} [{overall['me_lo']:+.2f}-{overall['me_hi']:+.2f}]"
          f"  r={overall['r']:.3f} [{overall['r_lo']:.3f}-{overall['r_hi']:.3f}]")
    forest_plot(rows, overall, 'mae', 'mae_lo', 'mae_hi',
                'MAE [years]', f'forest_MAE_{slug}.png', model_label=name)
    forest_plot(rows, overall, 'me',  'me_lo',  'me_hi',
                'Mean Error [years]  (+ = predicted too old)',
                f'forest_ME_{slug}.png', model_label=name)
    forest_plot(rows, overall, 'r',   'r_lo',   'r_hi',
                'Pearson r',   f'forest_r_{slug}.png',   model_label=name)

# ==============================================================================
# 12  Summary figures  (all 8 models)
# ==============================================================================
print("\n[12] Summary figures ...")

# 12a  Metrics bar chart – left sorted by MAE asc, right sorted by Pearson r desc
mae_sorted = sorted(ALL_RESULTS, key=lambda n: ALL_RESULTS[n]['MAE'])
r_sorted   = sorted(ALL_RESULTS, key=lambda n: ALL_RESULTS[n]['r'], reverse=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('All Models – Test Set Comparison',
             fontsize=12, fontweight='bold')

for ax, panel_names, vals_key, ylabel, best_is_min, subtitle in zip(
    axes,
    [mae_sorted, r_sorted],
    ['MAE', 'r'],
    ['MAE [years]', 'Pearson r'],
    [True, False],
    ['sorted by MAE  (lower is better)', 'sorted by Pearson r  (higher is better)'],
):
    panel_vals = [ALL_RESULTS[n][vals_key] for n in panel_names]
    panel_clrs = [MODEL_COLORS[n]          for n in panel_names]
    best_val   = min(panel_vals) if best_is_min else max(panel_vals)
    best_idx   = panel_vals.index(best_val)
    bars = ax.bar(range(len(panel_names)), panel_vals, color=panel_clrs, alpha=0.88,
                  edgecolor='black', linewidth=0.5)
    # Highlight best model with thick red border (keeps its actual color)
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(2.5)
    ax.set_xticks(range(len(panel_names)))
    ax.set_xticklabels(panel_names, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(subtitle)
    yrange = max(panel_vals) - min(panel_vals) if max(panel_vals) != min(panel_vals) else 0.01
    for i, (bar, v) in enumerate(zip(bars, panel_vals)):
        label = ('★ ' if i == best_idx else '') + (f'{v:.3f}' if 'Pearson' in ylabel else f'{v:.2f}')
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + yrange * 0.015,
                label, ha='center', va='bottom', fontsize=8,
                fontweight='bold' if i == best_idx else 'normal')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/summary_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved summary_comparison.png")

# 12b  Error distribution violin (sorted by MAE)
from matplotlib.lines import Line2D
fig, ax = plt.subplots(figsize=(14, 5))
fig.suptitle('Prediction Error Distribution by Model  (sorted by MAE)',
             fontsize=12, fontweight='bold')
err_data = [ALL_MODELS[n] - y_te for n in mae_sorted]
vp = ax.violinplot(err_data, positions=range(len(mae_sorted)),
                   showmeans=True, showmedians=True)
for pc, n in zip(vp['bodies'], mae_sorted):
    pc.set_facecolor(MODEL_COLORS[n]); pc.set_alpha(0.6)
vp['cmeans'].set_color('black'); vp['cmedians'].set_color('red')
ax.axhline(0, color='gray', ls='--', lw=1, alpha=0.6)
ax.set_xticks(range(len(mae_sorted)))
ax.set_xticklabels(mae_sorted, rotation=25, ha='right', fontsize=9)
ax.set_ylabel('Prediction error [years]')
ax.legend(handles=[Line2D([0],[0], color='black', lw=1.5, label='Mean'),
                   Line2D([0],[0], color='red',   lw=1.5, label='Median')], fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/summary_errors.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved summary_errors.png")

# 12c  Age-calibration plot – ME by actual age bin (all models)
age_bins   = [18, 30, 40, 50, 60, 70, 80, 90]
age_labels = ['18–29', '30–39', '40–49', '50–59', '60–69', '70–79', '80–89']
# age_bins defines left edges; bin b covers [age_bins[b], age_bins[b+1])
bin_idx = np.searchsorted(age_bins, y_te, side='right') - 1
bin_idx = np.clip(bin_idx, 0, len(age_labels) - 1)
bin_counts = [int((bin_idx == b).sum()) for b in range(len(age_labels))]

fig, ax = plt.subplots(figsize=(13, 6))
fig.suptitle('Age-Calibration: Mean Error by True Age Group\n'
             '(positive = model predicts older than reality)',
             fontsize=11, fontweight='bold')

n_models  = len(mae_sorted)
bar_w     = 0.8 / n_models
x_base    = np.arange(len(age_labels))

for mi, mname in enumerate(mae_sorted):
    pred_m = ALL_MODELS[mname]
    mes_bin, lo_bin, hi_bin = [], [], []
    for b in range(len(age_labels)):
        mask = bin_idx == b
        if mask.sum() < 5:
            mes_bin.append(np.nan); lo_bin.append(np.nan); hi_bin.append(np.nan)
            continue
        bt = bootstrap_metrics(y_te[mask], pred_m[mask])
        mes_bin.append(bt['me'] if bt else np.nan)
        lo_bin.append(bt['me_lo'] if bt else np.nan)
        hi_bin.append(bt['me_hi'] if bt else np.nan)
    x_pos = x_base + (mi - n_models / 2 + 0.5) * bar_w
    clr   = MODEL_COLORS[mname]
    bars  = ax.bar(x_pos, mes_bin, width=bar_w * 0.88,
                   color=clr, alpha=0.82, label=mname,
                   edgecolor='black', linewidth=0.4)
    # 95% CI whiskers
    for xp, me_v, lo_v, hi_v in zip(x_pos, mes_bin, lo_bin, hi_bin):
        if np.isnan(me_v): continue
        ax.plot([xp, xp], [lo_v, hi_v], color='black', lw=1.0, zorder=4)
        ax.plot([xp - bar_w * 0.2, xp + bar_w * 0.2], [lo_v, lo_v],
                color='black', lw=0.8, zorder=4)
        ax.plot([xp - bar_w * 0.2, xp + bar_w * 0.2], [hi_v, hi_v],
                color='black', lw=0.8, zorder=4)

ax.axhline(0, color='black', ls='--', lw=1.2, alpha=0.6, label='No bias (ME=0)')
ax.set_xticks(x_base)
ax.set_xticklabels(
    [f'{lab}\n(N={cnt})' for lab, cnt in zip(age_labels, bin_counts)],
    fontsize=10)
ax.set_ylabel('Mean Error [years]', fontsize=10)
ax.set_xlabel('True age group', fontsize=10)
ax.legend(fontsize=8, ncol=2, loc='upper right')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/summary/calibration_by_age.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved calibration_by_age.png")

# ==============================================================================
print("\n" + "=" * 65)
print("FINAL SUMMARY  (sorted by MAE)")
print("=" * 65)
print(f"  {'Model':28s}  {'MAE':>6}  {'RMSE':>7}  {'R2':>7}  {'r':>7}")
print("  " + "-" * 58)
for name in mae_sorted:
    rv   = ALL_RESULTS[name]
    mark = " <-- BEST" if name == BEST else ""
    print(f"  {name:28s}  {rv['MAE']:6.2f}  {rv['RMSE']:7.2f}  {rv['R2']:7.3f}  {rv['r']:7.3f}{mark}")
print("=" * 65)
print(f"\nPer-model files:  {{slug}}.png | forest_MAE_{{slug}}.png | forest_r_{{slug}}.png")
print(f"Summary files:    summary_comparison.png | summary_errors.png | feature_importance.png")

# ==============================================================================
# 13  Permutation Feature Importance  (model-agnostic, all 6 base models)
# ==============================================================================
# Why permutation importance?
#   MDI (feature_importances_) only exists for tree ensembles and is biased
#   toward high-cardinality features. Permutation importance is model-agnostic:
#   shuffle one feature at a time on the TEST set, measure mean MAE increase.
#   Works for Ridge, FT-Transformer, HistGB, RF, GradBoost, RF(tuned).
#   Stacked models are excluded: they take base model predictions as input,
#   not raw features, so permutation on original features is ill-defined.
print("\n[13] Permutation feature importance (all 6 base models) ...")
from sklearn.inspection import permutation_importance


class _FTWrapper:
    """Thin sklearn-compatible wrapper around the trained FT-Transformer."""
    def __init__(self, trained_net, sc, mu, sig, dev, batch=256):
        self._net = trained_net
        self._sc  = sc
        self._mu  = mu
        self._sig = sig
        self._dev = dev
        self._bs  = batch

    def fit(self, X, y=None):          # required by sklearn's check_scoring
        return self

    def predict(self, X):
        self._net.eval()
        Xf = self._sc.transform(X).astype(np.float32)
        chunks = [self._net(torch.tensor(Xf[i:i + self._bs]).to(self._dev)).detach().cpu().numpy()
                  for i in range(0, len(Xf), self._bs)]
        return np.concatenate(chunks) * self._sig + self._mu


ft_wrapper = _FTWrapper(net, scaler_ft, y_mu, y_sig, DEVICE)

_perm_pool = {
    'Ridge (PCA)':    sk_models['Ridge (PCA)'],
    'RandomForest':   sk_models['RandomForest'],
    'GradBoost':      sk_models['GradBoost'],
    'FT-Transformer': ft_wrapper,
    'RF (tuned)':     rf_search.best_estimator_,
    'HistGB (tuned)': hgb_search.best_estimator_,
}
# Sort by MAE ascending (best first) — consistent with all other charts
perm_candidates = {n: _perm_pool[n]
                   for n in sorted(_perm_pool, key=lambda n: ALL_RESULTS[n]['MAE'])}

N_REPEATS_PERM = 10
perm_imps = {}
for name, model in perm_candidates.items():
    print(f"  {name} ...", flush=True)
    r = permutation_importance(
        model, X_te, y_te,
        scoring='neg_mean_absolute_error',
        n_repeats=N_REPEATS_PERM,
        random_state=SEED,
        n_jobs=1,
    )
    perm_imps[name] = r.importances_mean   # mean MAE increase per feature

# --- Figure A: 2x3 grid, top-15 per model ---
fig, axes = plt.subplots(2, 3, figsize=(20, 13))
fig.suptitle(
    'Permutation Feature Importance – MAE increase when feature shuffled\n'
    '(test set, 10 repeats; higher = more important)',
    fontsize=12, fontweight='bold',
)
for ax, (name, imps) in zip(axes.flat, perm_imps.items()):
    top15 = np.argsort(imps)[::-1][:15]
    clr   = MODEL_COLORS[name]
    colors_bar = ['#d62728' if v < 0 else clr for v in imps[top15[::-1]]]
    ax.barh(range(15), imps[top15[::-1]], color=colors_bar, alpha=0.82)
    ax.set_yticks(range(15))
    ax.set_yticklabels([ppg_cols[i] for i in top15[::-1]], fontsize=8)
    ax.axvline(0, color='black', lw=0.8, ls='--')
    ax.set_xlabel('Mean MAE increase [years]', fontsize=8)
    ax.set_title(name, fontweight='bold', fontsize=10)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/feature_importance_permutation.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved feature_importance_permutation.png")

# --- Figure B: Top-10 features shared across models (heatmap-style) ---
# Union of top-10 from each model, then show importance across all models
top10_sets = [set(np.argsort(perm_imps[n])[::-1][:10]) for n in perm_candidates]
shared_feats = sorted(set.union(*top10_sets),
                      key=lambda i: np.mean([perm_imps[n][i] for n in perm_candidates]),
                      reverse=True)[:25]   # cap at 25 rows

# Columns already in MAE order because perm_candidates is sorted above
mat   = np.array([[perm_imps[n][i] for n in perm_candidates] for i in shared_feats])
model_names_short = list(perm_candidates.keys())   # already MAE-sorted

fig, ax = plt.subplots(figsize=(13, max(6, len(shared_feats) * 0.38)))
im = ax.imshow(mat, aspect='auto', cmap='RdYlGn')
ax.set_xticks(range(len(model_names_short)))
ax.set_xticklabels(model_names_short, rotation=30, ha='right', fontsize=9)
ax.set_yticks(range(len(shared_feats)))
ax.set_yticklabels([ppg_cols[i] for i in shared_feats], fontsize=8)
plt.colorbar(im, ax=ax, label='Mean MAE increase [years]')
ax.set_title('Feature Importance Across Models – Top-25 Shared Features\n'
             '(permutation importance, test set)', fontweight='bold', fontsize=11)
# Annotate cells
for r in range(len(shared_feats)):
    for c in range(len(model_names_short)):
        v = mat[r, c]
        ax.text(c, r, f'{v:.3f}', ha='center', va='center',
                fontsize=6.5, color='black')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/feature_importance_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved feature_importance_heatmap.png")

print(f"\nAll outputs saved to:\n  {OUT_DIR}")

