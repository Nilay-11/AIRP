"""Shared data loading, model definitions and metrics."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # all models run on CPU; hide the GPU so parallel workers never contend for it
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score, recall_score,
                             f1_score, brier_score_loss, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "nhanes_cvd_2011_2018.csv"
RES = ROOT / "results"
FIG = ROOT / "figures"
MODELS_DIR = ROOT / "models"
for p in (RES, FIG, MODELS_DIR):
    p.mkdir(exist_ok=True)

SEED = 42
FEATURES = ["age", "male", "eth_mexican_american", "eth_other_hispanic", "eth_nh_black", "eth_other_multi",
            "income_poverty_ratio", "education", "systolic_bp", "diastolic_bp", "pulse_rate", "bmi",
            "waist_cm", "total_chol", "hdl_chol", "hba1c", "creatinine", "uric_acid", "smoking", "diabetes_dx"]
FEATURE_GROUPS = {
    "Demographic": ["age", "male", "eth_mexican_american", "eth_other_hispanic", "eth_nh_black", "eth_other_multi"],
    "Socioeconomic": ["income_poverty_ratio", "education"],
    "Blood pressure & pulse": ["systolic_bp", "diastolic_bp", "pulse_rate"],
    "Anthropometric": ["bmi", "waist_cm"],
    "Lipids": ["total_chol", "hdl_chol"],
    "Glycaemic": ["hba1c", "diabetes_dx"],
    "Renal/metabolic": ["creatinine", "uric_acid"],
    "Smoking": ["smoking"],
}
PRETTY = {"age": "Age", "male": "Male sex", "eth_mexican_american": "Mexican American",
          "eth_other_hispanic": "Other Hispanic", "eth_nh_black": "Non-Hispanic Black",
          "eth_other_multi": "Other/multiracial", "income_poverty_ratio": "Income-to-poverty ratio",
          "education": "Education", "systolic_bp": "Systolic BP", "diastolic_bp": "Diastolic BP",
          "pulse_rate": "Pulse rate", "bmi": "BMI", "waist_cm": "Waist circumference",
          "total_chol": "Total cholesterol", "hdl_chol": "HDL cholesterol", "hba1c": "HbA1c",
          "creatinine": "Serum creatinine", "uric_acid": "Uric acid", "smoking": "Smoking status",
          "diabetes_dx": "Diagnosed diabetes"}
MODEL_ORDER = ["LR", "RF", "SVM", "XGB", "MLP", "ResNet", "FTT"]
CLASSICAL, DEEP = ["LR", "RF", "SVM", "XGB"], ["MLP", "ResNet", "FTT"]
MODEL_NAMES = {"LR": "Logistic Regression", "RF": "Random Forest", "SVM": "SVM (RBF)",
               "XGB": "XGBoost", "MLP": "MLP", "ResNet": "ResNet", "FTT": "FT-Transformer"}


def load():
    return pd.read_csv(DATA)


def split(df, seed=SEED):
    return train_test_split(df, test_size=0.2, stratify=df.cvd, random_state=seed)


# ---------------------------------------------------------------- MLP
class TorchMLP(BaseEstimator, ClassifierMixin):
    """Feed-forward network (Linear-BatchNorm-ReLU-Dropout blocks), AdamW, early stopping on validation loss."""

    def __init__(self, hidden=(128, 64), dropout=0.2, lr=1e-3, weight_decay=1e-4, batch_size=256,
                 max_epochs=200, patience=15, val_frac=0.15, random_state=SEED):
        self.hidden, self.dropout, self.lr, self.weight_decay = hidden, dropout, lr, weight_decay
        self.batch_size, self.max_epochs, self.patience = batch_size, max_epochs, patience
        self.val_frac, self.random_state = val_frac, random_state

    def _net(self, d):
        layers = []
        for h in self.hidden:
            layers += [torch.nn.Linear(d, h), torch.nn.BatchNorm1d(h), torch.nn.ReLU(), torch.nn.Dropout(self.dropout)]
            d = h
        layers.append(torch.nn.Linear(d, 1))
        return torch.nn.Sequential(*layers)

    def fit(self, X, y):
        torch.set_num_threads(1)
        torch.manual_seed(self.random_state)
        # TORCH_DEVICE=cuda trains on the GPU (used only for FT-Transformer ablation/bootstrap refits); default CPU
        dev = torch.device(os.environ.get("TORCH_DEVICE", "cpu"))
        if dev.type == "cuda":
            torch.cuda.manual_seed_all(self.random_state)
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
        X, y = np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)
        self.classes_ = np.array([0, 1])
        Xt, Xv, yt, yv = train_test_split(X, y, test_size=self.val_frac, stratify=y, random_state=self.random_state)
        Xt, yt, Xv, yv = (torch.from_numpy(a).to(dev) for a in (Xt, yt, Xv, yv))
        self.net_ = self._net(X.shape[1]).to(dev)
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        g = torch.Generator().manual_seed(self.random_state)
        best, best_state, wait = np.inf, None, 0
        for epoch in range(self.max_epochs):
            self.net_.train()
            perm = torch.randperm(len(Xt), generator=g).to(dev)
            for i in range(0, len(Xt), self.batch_size):
                idx = perm[i:i + self.batch_size]
                if len(idx) < 2:
                    continue
                opt.zero_grad()
                loss_fn(self.net_(Xt[idx]).squeeze(1), yt[idx]).backward()
                opt.step()
            self.net_.eval()
            with torch.no_grad():
                vl = loss_fn(self.net_(Xv).squeeze(1), yv).item()
            if vl < best - 1e-5:
                best, wait = vl, 0
                best_state = {k: v.detach().clone() for k, v in self.net_.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.net_.load_state_dict(best_state)
        self.n_epochs_ = epoch + 1
        return self

    def decision_function(self, X):
        self.net_.eval()
        dev = next(self.net_.parameters()).device
        X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        with torch.no_grad():  # batched so large SHAP evaluations fit in memory
            return torch.cat([self.net_(X[i:i + 8192].to(dev)).squeeze(1).cpu()
                              for i in range(0, len(X), 8192)]).numpy()

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-self.decision_function(X)))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class _ResBlock(torch.nn.Module):
    def __init__(self, d, d_hidden, dropout):
        super().__init__()
        self.f = torch.nn.Sequential(torch.nn.BatchNorm1d(d), torch.nn.Linear(d, d_hidden), torch.nn.ReLU(),
                                     torch.nn.Dropout(dropout), torch.nn.Linear(d_hidden, d), torch.nn.Dropout(dropout))

    def forward(self, x):
        return x + self.f(x)


class TorchResNet(TorchMLP):
    """Tabular ResNet (Gorishniy et al., 2021): Linear -> residual blocks [BN-Linear-ReLU-Dropout-Linear-Dropout]
    -> BN-ReLU-Linear head. Same training loop as TorchMLP."""

    def __init__(self, d=128, n_blocks=2, hidden_factor=2.0, dropout=0.2, lr=1e-3, weight_decay=1e-4,
                 batch_size=256, max_epochs=200, patience=15, val_frac=0.15, random_state=SEED):
        self.d, self.n_blocks, self.hidden_factor, self.dropout = d, n_blocks, hidden_factor, dropout
        self.lr, self.weight_decay, self.batch_size, self.max_epochs = lr, weight_decay, batch_size, max_epochs
        self.patience, self.val_frac, self.random_state = patience, val_frac, random_state

    def _net(self, d_in):
        h = int(self.d * self.hidden_factor)
        return torch.nn.Sequential(torch.nn.Linear(d_in, self.d),
                                   *[_ResBlock(self.d, h, self.dropout) for _ in range(self.n_blocks)],
                                   torch.nn.BatchNorm1d(self.d), torch.nn.ReLU(), torch.nn.Linear(self.d, 1))


class _ReGLU(torch.nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        return a * torch.relu(b)


class _FTBlock(torch.nn.Module):
    def __init__(self, d, n_heads, attn_dropout, ffn_dropout, residual_dropout, first):
        super().__init__()
        self.ln1 = torch.nn.Identity() if first else torch.nn.LayerNorm(d)  # first block: no pre-norm (as in FT-T)
        self.attn = torch.nn.MultiheadAttention(d, n_heads, dropout=attn_dropout, batch_first=True)
        self.ln2 = torch.nn.LayerNorm(d)
        h = int(d * 4 / 3)
        self.ffn = torch.nn.Sequential(torch.nn.Linear(d, 2 * h), _ReGLU(), torch.nn.Dropout(ffn_dropout),
                                       torch.nn.Linear(h, d))
        self.drop = torch.nn.Dropout(residual_dropout)

    def forward(self, x):
        z = self.ln1(x)
        x = x + self.drop(self.attn(z, z, z, need_weights=False)[0])
        return x + self.drop(self.ffn(self.ln2(x)))


class _FTNet(torch.nn.Module):
    def __init__(self, n_features, d, n_blocks, n_heads, attn_dropout, ffn_dropout, residual_dropout):
        super().__init__()
        self.w = torch.nn.Parameter(torch.empty(n_features, d))
        self.b = torch.nn.Parameter(torch.empty(n_features, d))
        self.cls = torch.nn.Parameter(torch.empty(1, 1, d))
        for p in (self.w, self.b, self.cls):
            torch.nn.init.uniform_(p, -d ** -0.5, d ** -0.5)
        self.blocks = torch.nn.ModuleList([_FTBlock(d, n_heads, attn_dropout, ffn_dropout, residual_dropout, i == 0)
                                           for i in range(n_blocks)])
        self.head = torch.nn.Sequential(torch.nn.LayerNorm(d), torch.nn.ReLU(), torch.nn.Linear(d, 1))

    def forward(self, x):
        tok = x.unsqueeze(-1) * self.w + self.b  # feature tokenizer: one d-dim token per feature
        tok = torch.cat([self.cls.expand(len(x), -1, -1), tok], dim=1)
        for blk in self.blocks:
            tok = blk(tok)
        return self.head(tok[:, 0])


class TorchFTTransformer(TorchMLP):
    """FT-Transformer (Gorishniy et al., 2021): feature tokenizer + [CLS] + pre-norm Transformer blocks
    (multi-head self-attention, ReGLU feed-forward). Same training loop as TorchMLP."""

    def __init__(self, d=64, n_blocks=2, n_heads=8, attn_dropout=0.2, ffn_dropout=0.1, residual_dropout=0.0,
                 lr=1e-4, weight_decay=1e-5, batch_size=256, max_epochs=200, patience=15, val_frac=0.15,
                 random_state=SEED):
        self.d, self.n_blocks, self.n_heads = d, n_blocks, n_heads
        self.attn_dropout, self.ffn_dropout, self.residual_dropout = attn_dropout, ffn_dropout, residual_dropout
        self.lr, self.weight_decay, self.batch_size, self.max_epochs = lr, weight_decay, batch_size, max_epochs
        self.patience, self.val_frac, self.random_state = patience, val_frac, random_state

    def _net(self, d_in):
        return _FTNet(d_in, self.d, self.n_blocks, self.n_heads, self.attn_dropout, self.ffn_dropout,
                      self.residual_dropout)


# ---------------------------------------------------------------- models
def make_pipeline(name, **params):
    est = {
        "LR": LogisticRegression(max_iter=5000, random_state=SEED),
        "RF": RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=SEED),
        "SVM": SVC(kernel="rbf", cache_size=2000, random_state=SEED),
        "XGB": XGBClassifier(tree_method="hist", eval_metric="logloss", n_jobs=4, random_state=SEED),
        "MLP": TorchMLP(random_state=SEED),
        "ResNet": TorchResNet(random_state=SEED),
        "FTT": TorchFTTransformer(random_state=SEED),
    }[name]
    pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("clf", est)])
    if params:
        pipe.set_params(**{f"clf__{k}": v for k, v in params.items()})
    return pipe


def scores(model, X):
    """Probability of CVD (SVM must have been fitted with probability=True)."""
    return model.predict_proba(X)[:, 1]


def rank_scores(model, X):
    """Monotone score usable for AUC (avoids Platt scaling cost for SVM)."""
    clf = model.named_steps["clf"]
    return model.decision_function(X) if isinstance(clf, SVC) and not clf.probability else model.predict_proba(X)[:, 1]


# ---------------------------------------------------------------- metrics
def ece(y, p, bins=10):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    return sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any())


def calib_slope_intercept(y, p):
    """Calibration slope (logit(y) ~ a + b*lp) and calibration-in-the-large (intercept with lp as offset)."""
    from scipy.optimize import minimize, minimize_scalar
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    lp = np.log(pc / (1 - pc))

    def nll(eta):
        return np.sum(np.logaddexp(0, eta) - y * eta)

    slope = minimize(lambda w: nll(w[0] + w[1] * lp), x0=[0.0, 1.0], method="BFGS").x[1]
    intercept = minimize_scalar(lambda a: nll(a + lp)).x
    return slope, intercept


def all_metrics(y, p, thr):
    y = np.asarray(y)
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    slope, intercept = calib_slope_intercept(y, p)
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "precision": precision_score(y, yhat, zero_division=0), "recall": recall_score(y, yhat),
            "specificity": tn / (tn + fp), "f1": f1_score(y, yhat), "brier": brier_score_loss(y, p),
            "ece": ece(y, p), "calib_slope": slope, "calib_intercept": intercept, "threshold": thr}


def best_f1_threshold(y, p):
    grid = np.unique(np.quantile(p, np.linspace(0.5, 0.995, 400)))
    f = [f1_score(y, (p >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmax(f))])


def delong_test(y, p1, p2):
    """Two-sided DeLong test for correlated ROC AUCs (Sun & Xu fast implementation)."""
    y = np.asarray(y)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    m, n = len(pos), len(neg)

    def comps(p):
        tx, ty, tz = (stats.rankdata(p[pos]), stats.rankdata(p[neg]), stats.rankdata(np.r_[p[pos], p[neg]]))
        auc = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
        v10 = (tz[:m] - tx) / n
        v01 = 1 - (tz[m:] - ty) / m
        return auc, v10, v01

    a1, v10a, v01a = comps(np.asarray(p1))
    a2, v10b, v01b = comps(np.asarray(p2))
    s10, s01 = np.cov(np.vstack([v10a, v10b])), np.cov(np.vstack([v01a, v01b]))
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / m + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n
    z = (a1 - a2) / np.sqrt(var)
    return a1 - a2, 2 * stats.norm.sf(abs(z))


def bootstrap_ci(y, p, fn, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        vals.append(fn(y[i], p[i]))
    return np.percentile(vals, [2.5, 97.5])
