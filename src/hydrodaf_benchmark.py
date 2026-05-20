import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from itertools import cycle
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import ConcatDataset, DataLoader, Dataset


BASIN_AREAS_KM2 = {}


def basin_area_for(path: Path, basin_name: str) -> float:
    if basin_name in BASIN_AREAS_KM2:
        return BASIN_AREAS_KM2[basin_name]
    area_file = path.parent / "basin_areas.csv"
    if area_file.exists():
        areas = pd.read_csv(area_file, dtype={"basin": str})
        match = areas[areas["basin"].astype(str) == str(basin_name)]
        if not match.empty:
            return float(match.iloc[0]["area_km2"])
    return 1.0


@dataclass
class ExperimentConfig:
    input_len: int = 60
    train_ratio: float = 0.7
    target_fraction: float = 1.0
    batch_size: int = 256
    hidden_dim: int = 64
    epochs: int = 40
    pretrain_epochs: int = 0
    finetune_epochs: int = 20
    lr: float = 1e-3
    lambda_max: float = 0.002
    nse_gamma: float = 0.2
    source_weight: float = 1.0
    target_weight: float = 1.0
    domain_weight: float = 1.0
    seed: int = 42
    device: str = "cpu"
    feature_mode: str = "mean_rain"
    max_rain_cols: int = 12


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_basin_excel(
    path: Path,
    basin_name: str,
    feature_mode: str = "mean_rain",
    max_rain_cols: int = 12,
) -> pd.DataFrame:
    df = pd.read_excel(path)
    df["date"] = pd.to_datetime(df[["Year", "month", "day"]])
    df = df.set_index("date").sort_index()
    df["Data"] = pd.to_numeric(df["Data"], errors="coerce")
    df = df.dropna(subset=["Data"])

    rain_cols = [c for c in df.columns if c not in {"Year", "month", "day", "Data"}]
    if not rain_cols:
        raise ValueError(f"No rainfall columns found in {path}")

    out = pd.DataFrame(index=df.index)
    out["q"] = df["Data"].astype(float)
    out["log_q"] = np.log1p(out["q"])
    rain_values = df[rain_cols].astype(float)
    if feature_mode == "mean_rain":
        out["p_mean"] = rain_values.mean(axis=1)
        api = np.zeros(len(out), dtype=float)
        p_values = out["p_mean"].values
        for i in range(1, len(out)):
            api[i] = 0.9 * api[i - 1] + p_values[i]
        out["api"] = api
    elif feature_mode == "all_rain":
        padded = np.zeros((len(out), max_rain_cols), dtype=float)
        n_cols = min(max_rain_cols, rain_values.shape[1])
        padded[:, :n_cols] = rain_values.iloc[:, :n_cols].values
        for j in range(max_rain_cols):
            out[f"rain_{j:02d}"] = padded[:, j]
            api = np.zeros(len(out), dtype=float)
            for i in range(1, len(out)):
                api[i] = 0.9 * api[i - 1] + padded[i, j]
            out[f"api_{j:02d}"] = api
    elif feature_mode == "rain_stats":
        out["p_mean"] = rain_values.mean(axis=1)
        out["p_max"] = rain_values.max(axis=1)
        out["p_std"] = rain_values.std(axis=1).fillna(0.0)
        out["p_wet_frac"] = (rain_values > 0).mean(axis=1)
        for col in ["p_mean", "p_max"]:
            api = np.zeros(len(out), dtype=float)
            p_values = out[col].values
            for i in range(1, len(out)):
                api[i] = 0.9 * api[i - 1] + p_values[i]
            out[f"api_{col}"] = api
    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")
    day = out.index.dayofyear.astype(float)
    out["doy_sin"] = np.sin(2 * np.pi * day / 366.0)
    out["doy_cos"] = np.cos(2 * np.pi * day / 366.0)
    out["area_log"] = math.log1p(basin_area_for(path, basin_name))
    return out


class SequenceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        input_len: int,
        x_scaler: StandardScaler,
        y_scaler: StandardScaler,
        feature_cols: list[str],
    ) -> None:
        self.input_len = input_len
        self.feature_cols = feature_cols
        self.q_raw = df["q"].values.astype(float)
        self.x = x_scaler.transform(df[feature_cols].values.astype(float)).astype(np.float32)
        self.y = y_scaler.transform(df[["log_q"]].values.astype(float)).astype(np.float32).reshape(-1)

    def __len__(self) -> int:
        return max(0, len(self.x) - self.input_len)

    def __getitem__(self, idx: int):
        x = self.x[idx : idx + self.input_len]
        y = self.y[idx + self.input_len]
        q = self.q_raw[idx + self.input_len]
        return torch.from_numpy(x), torch.tensor([y], dtype=torch.float32), torch.tensor(q, dtype=torch.float32)


@dataclass
class DomainData:
    name: str
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    train_dataset: SequenceDataset
    test_dataset: SequenceDataset


def build_domain(
    name: str,
    df: pd.DataFrame,
    config: ExperimentConfig,
    feature_cols: list[str],
    target_fraction: float | None = None,
    fit_df: pd.DataFrame | None = None,
) -> DomainData:
    split = int(len(df) * config.train_ratio)
    full_train_df = df.iloc[:split].copy()
    test_df = df.iloc[split:].copy()

    frac = config.target_fraction if target_fraction is None else target_fraction
    train_n = max(config.input_len + 2, int(len(full_train_df) * frac))
    train_df = full_train_df.iloc[:train_n].copy()

    scaler_fit = train_df if fit_df is None else fit_df
    x_scaler = StandardScaler().fit(scaler_fit[feature_cols].values.astype(float))
    y_scaler = StandardScaler().fit(scaler_fit[["log_q"]].values.astype(float))

    train_dataset = SequenceDataset(train_df, config.input_len, x_scaler, y_scaler, feature_cols)
    test_dataset = SequenceDataset(test_df, config.input_len, x_scaler, y_scaler, feature_cols)
    return DomainData(name, train_df, test_df, x_scaler, y_scaler, train_dataset, test_dataset)


def build_domain_from_dfs(
    name: str,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    config: ExperimentConfig,
    feature_cols: list[str],
    scaler_fit_df: pd.DataFrame | None = None,
) -> DomainData:
    scaler_fit = train_df if scaler_fit_df is None else scaler_fit_df
    x_scaler = StandardScaler().fit(scaler_fit[feature_cols].values.astype(float))
    y_scaler = StandardScaler().fit(scaler_fit[["log_q"]].values.astype(float))
    train_dataset = SequenceDataset(train_df, config.input_len, x_scaler, y_scaler, feature_cols)
    eval_dataset = SequenceDataset(eval_df, config.input_len, x_scaler, y_scaler, feature_cols)
    return DomainData(name, train_df, eval_df, x_scaler, y_scaler, train_dataset, eval_dataset)


def inverse_logq(y_scaled: np.ndarray, y_scaler: StandardScaler) -> np.ndarray:
    logq = y_scaler.inverse_transform(y_scaled.reshape(-1, 1)).reshape(-1)
    return np.maximum(0.0, np.expm1(logq))


def calc_metrics(obs: np.ndarray, sim: np.ndarray) -> dict[str, float]:
    obs = np.asarray(obs, dtype=float).reshape(-1)
    sim = np.asarray(sim, dtype=float).reshape(-1)
    rmse = float(np.sqrt(mean_squared_error(obs, sim)))
    denom = np.sum((obs - np.mean(obs)) ** 2)
    nse = float(1 - np.sum((obs - sim) ** 2) / denom) if denom > 0 else np.nan
    pbias = float(np.sum(sim - obs) / np.sum(obs) * 100) if np.sum(obs) != 0 else np.nan
    r2 = float(r2_score(obs, sim))
    if np.std(obs) == 0 or np.std(sim) == 0 or np.mean(obs) == 0:
        kge = np.nan
    else:
        r = np.corrcoef(obs, sim)[0, 1]
        alpha = np.std(sim) / np.std(obs)
        beta = np.mean(sim) / np.mean(obs)
        kge = float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))
    return {"NSE": nse, "KGE": kge, "RMSE": rmse, "PBIAS": pbias, "R2": r2}


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_: float = 1.0) -> None:
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.out_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query, key, value, return_weights: bool = False):
        batch_size = query.shape[0]
        q = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = torch.softmax(scores, dim=-1)
        attn = torch.matmul(weights, v).transpose(1, 2).contiguous()
        attn = attn.view(batch_size, -1, self.num_heads * self.head_dim)
        out = self.out_projection(attn)
        if return_weights:
            return out, weights
        return out


class DAFModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        use_domain_specific_v: bool = True,
        use_domain_specific_head: bool = True,
    ) -> None:
        super().__init__()
        self.use_domain_specific_v = use_domain_specific_v
        self.use_domain_specific_head = use_domain_specific_head
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.1)
        self.q_projection = nn.Linear(hidden_dim, hidden_dim)
        self.k_projection = nn.Linear(hidden_dim, hidden_dim)
        if use_domain_specific_v:
            self.v_source = nn.Linear(hidden_dim, hidden_dim)
            self.v_target = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.v_shared = nn.Linear(hidden_dim, hidden_dim)
        self.attention = MultiHeadAttention(hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        def make_predictor():
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )

        if use_domain_specific_head:
            self.predictor_source = make_predictor()
            self.predictor_target = make_predictor()
        else:
            self.predictor = make_predictor()
        self.grl = GradientReversalLayer()
        self.discriminator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode(self, x, domain: str, return_weights: bool = False):
        z = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        enc, _ = self.lstm(z)
        q = self.q_projection(enc)
        k = self.k_projection(enc)
        if self.use_domain_specific_v:
            v = self.v_source(enc) if domain == "source" else self.v_target(enc)
        else:
            v = self.v_shared(enc)
        if return_weights:
            attn, weights = self.attention(q, k, v, return_weights=True)
        else:
            attn = self.attention(q, k, v, return_weights=False)
            weights = None
        feat = self.norm(attn + enc)[:, -1, :]
        return feat, weights

    def forward_pair(self, x_source, x_target):
        fs, _ = self.encode(x_source, "source")
        ft, _ = self.encode(x_target, "target")
        if self.use_domain_specific_head:
            ps = self.predictor_source(fs)
            pt = self.predictor_target(ft)
        else:
            ps = self.predictor(fs)
            pt = self.predictor(ft)
        return (
            ps,
            pt,
            self.discriminator(self.grl(fs)),
            self.discriminator(self.grl(ft)),
        )

    def forward_target(self, x_target, return_weights: bool = False):
        feat, weights = self.encode(x_target, "target", return_weights=return_weights)
        pred = self.predictor_target(feat) if self.use_domain_specific_head else self.predictor(feat)
        if return_weights:
            return pred, weights
        return pred


class HybridLoss(nn.Module):
    def __init__(self, gamma: float = 0.2) -> None:
        super().__init__()
        self.gamma = gamma
        self.mse = nn.MSELoss()

    def forward(self, pred, true):
        mse = self.mse(pred, true)
        denom = torch.sum((true - torch.mean(true)) ** 2)
        if denom <= 1e-6:
            return mse
        nse_loss = torch.sum((pred - true) ** 2) / denom
        return mse + self.gamma * nse_loss


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def train_supervised(model, dataset: Dataset, config: ExperimentConfig) -> None:
    model.to(config.device)
    model.train()
    loader = make_loader(dataset, config.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = HybridLoss(config.nse_gamma)
    for _ in range(config.epochs):
        for x, y, _ in loader:
            x = x.to(config.device)
            y = y.to(config.device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


def finetune_supervised(model, dataset: Dataset, config: ExperimentConfig, epochs: int) -> None:
    old_epochs = config.epochs
    config.epochs = epochs
    train_supervised(model, dataset, config)
    config.epochs = old_epochs


def pretrain_daf_source(model: DAFModel, source_dataset: Dataset, config: ExperimentConfig) -> None:
    if config.pretrain_epochs <= 0:
        return
    model.to(config.device)
    loader = make_loader(source_dataset, config.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = HybridLoss(config.nse_gamma)
    for _ in range(config.pretrain_epochs):
        model.train()
        for x, y, _ in loader:
            x = x.to(config.device)
            y = y.to(config.device)
            optimizer.zero_grad()
            feat, _ = model.encode(x, "source")
            pred = model.predictor_source(feat) if model.use_domain_specific_head else model.predictor(feat)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


def train_daf(model: DAFModel, source_dataset: Dataset, target_dataset: Dataset, config: ExperimentConfig) -> None:
    model.to(config.device)
    pretrain_daf_source(model, source_dataset, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    forecast_loss = HybridLoss(config.nse_gamma)
    domain_loss = nn.BCEWithLogitsLoss()
    source_loader = make_loader(source_dataset, config.batch_size, shuffle=True)
    target_loader = make_loader(target_dataset, config.batch_size, shuffle=True)
    if len(source_loader) >= len(target_loader):
        main_loader = source_loader
        other_loader = cycle(target_loader)
        source_first = True
    else:
        main_loader = target_loader
        other_loader = cycle(source_loader)
        source_first = False
    for epoch in range(config.epochs):
        p = epoch / max(1, config.epochs - 1)
        model.grl.lambda_ = config.lambda_max * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)
        model.train()
        for batch_main, batch_other in zip(main_loader, other_loader):
            if source_first:
                xs, ys, _ = batch_main
                xt, yt, _ = batch_other
            else:
                xt, yt, _ = batch_main
                xs, ys, _ = batch_other
            xs = xs.to(config.device)
            ys = ys.to(config.device)
            xt = xt.to(config.device)
            yt = yt.to(config.device)
            optimizer.zero_grad()
            ps, pt, ds, dt = model.forward_pair(xs, xt)
            labels_s = torch.ones_like(ds)
            labels_t = torch.zeros_like(dt)
            loss = (
                config.source_weight * forecast_loss(ps, ys)
                + config.target_weight * forecast_loss(pt, yt)
                + config.domain_weight * (domain_loss(ds, labels_s) + domain_loss(dt, labels_t))
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


def predict_lstm(model, dataset: Dataset, y_scaler: StandardScaler, config: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    model.to(config.device)
    model.eval()
    loader = make_loader(dataset, config.batch_size, shuffle=False)
    preds, obs = [], []
    with torch.no_grad():
        for x, _, q in loader:
            yhat = model(x.to(config.device)).cpu().numpy().reshape(-1)
            preds.append(yhat)
            obs.append(q.numpy().reshape(-1))
    pred_q = inverse_logq(np.concatenate(preds), y_scaler)
    obs_q = np.concatenate(obs)
    return obs_q, pred_q


def predict_daf(model: DAFModel, dataset: Dataset, y_scaler: StandardScaler, config: ExperimentConfig):
    model.to(config.device)
    model.eval()
    loader = make_loader(dataset, config.batch_size, shuffle=False)
    preds, obs = [], []
    with torch.no_grad():
        for x, _, q in loader:
            yhat = model.forward_target(x.to(config.device)).cpu().numpy().reshape(-1)
            preds.append(yhat)
            obs.append(q.numpy().reshape(-1))
    pred_q = inverse_logq(np.concatenate(preds), y_scaler)
    obs_q = np.concatenate(obs)
    return obs_q, pred_q


def align_source_to_target(source_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    start = max(source_df.index.min(), target_df.index.min())
    end = min(source_df.index.max(), target_df.index.max())
    aligned = source_df.loc[start:end].copy()
    if len(aligned) < 200:
        return source_df.copy()
    return aligned


def run_one_model(
    model_name: str,
    source: DomainData,
    target: DomainData,
    input_dim: int,
    config: ExperimentConfig,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    if model_name == "local_lstm":
        model = LSTMRegressor(input_dim, config.hidden_dim)
        train_supervised(model, target.train_dataset, config)
        obs, pred = predict_lstm(model, target.test_dataset, target.y_scaler, config)
    elif model_name == "source_only":
        model = LSTMRegressor(input_dim, config.hidden_dim)
        train_supervised(model, source.train_dataset, config)
        source_scaled_target_test = SequenceDataset(
            target.test_df, config.input_len, source.x_scaler, source.y_scaler, target.train_dataset.feature_cols
        )
        obs, pred = predict_lstm(model, source_scaled_target_test, source.y_scaler, config)
    elif model_name == "finetune_lstm":
        model = LSTMRegressor(input_dim, config.hidden_dim)
        train_supervised(model, source.train_dataset, config)
        finetune_supervised(model, target.train_dataset, config, config.finetune_epochs)
        obs, pred = predict_lstm(model, target.test_dataset, target.y_scaler, config)
    elif model_name == "regional_lstm":
        model = LSTMRegressor(input_dim, config.hidden_dim)
        train_supervised(model, ConcatDataset([source.train_dataset, target.train_dataset]), config)
        obs, pred = predict_lstm(model, target.test_dataset, target.y_scaler, config)
    elif model_name == "daf_shared_v":
        model = DAFModel(input_dim, config.hidden_dim, use_domain_specific_v=False, use_domain_specific_head=True)
        train_daf(model, source.train_dataset, target.train_dataset, config)
        obs, pred = predict_daf(model, target.test_dataset, target.y_scaler, config)
    elif model_name == "daf":
        model = DAFModel(input_dim, config.hidden_dim, use_domain_specific_v=True, use_domain_specific_head=True)
        train_daf(model, source.train_dataset, target.train_dataset, config)
        obs, pred = predict_daf(model, target.test_dataset, target.y_scaler, config)
    elif model_name == "daf_single_head":
        model = DAFModel(input_dim, config.hidden_dim, use_domain_specific_v=True, use_domain_specific_head=False)
        train_daf(model, source.train_dataset, target.train_dataset, config)
        obs, pred = predict_daf(model, target.test_dataset, target.y_scaler, config)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return calc_metrics(obs, pred), obs, pred


def run_local(args) -> None:
    config = ExperimentConfig(
        input_len=args.input_len,
        target_fraction=args.target_fraction,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        lr=args.lr,
        lambda_max=args.lambda_max,
        nse_gamma=args.nse_gamma,
        source_weight=args.source_weight,
        target_weight=args.target_weight,
        domain_weight=args.domain_weight,
        seed=args.seeds[0],
        device=args.device,
        feature_mode=args.feature_mode,
        max_rain_cols=args.max_rain_cols,
    )
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.feature_mode == "mean_rain":
        feature_cols = ["log_q", "p_mean", "api", "doy_sin", "doy_cos", "area_log"]
    elif args.feature_mode == "all_rain":
        feature_cols = (
            ["log_q"]
            + [f"rain_{j:02d}" for j in range(args.max_rain_cols)]
            + [f"api_{j:02d}" for j in range(args.max_rain_cols)]
            + ["doy_sin", "doy_cos", "area_log"]
        )
    elif args.feature_mode == "rain_stats":
        feature_cols = [
            "log_q",
            "p_mean",
            "p_max",
            "p_std",
            "p_wet_frac",
            "api_p_mean",
            "api_p_max",
            "doy_sin",
            "doy_cos",
            "area_log",
        ]
    else:
        raise ValueError(f"Unknown feature_mode: {args.feature_mode}")
    input_dim = len(feature_cols)

    raw = {}
    basin_names = sorted(set([args.source, *args.targets]))
    for basin_name in basin_names:
        path = data_dir / f"{basin_name}.xlsx"
        if not path.exists():
            raise FileNotFoundError(f"Expected basin file not found: {path}")
        raw[basin_name] = read_basin_excel(path, basin_name, config.feature_mode, config.max_rain_cols)

    rows = []
    for seed in args.seeds:
        config.seed = seed
        set_seed(seed)
        for target_name in args.targets:
            source_df = align_source_to_target(raw[args.source], raw[target_name])
            target_df = raw[target_name]
            source = build_domain(args.source, source_df, config, feature_cols, target_fraction=1.0)
            target = build_domain(target_name, target_df, config, feature_cols, target_fraction=args.target_fraction)
            for model_name in args.models:
                print(f"[run] seed={seed} target={target_name} model={model_name}")
                set_seed(seed)
                metrics, obs, pred = run_one_model(model_name, source, target, input_dim, config)
                row = {
                    "seed": seed,
                    "source": args.source,
                    "target": target_name,
                    "model": model_name,
                    "target_fraction": args.target_fraction,
                    **metrics,
                }
                rows.append(row)
                pred_dir = out_dir / "predictions"
                pred_dir.mkdir(exist_ok=True)
                pd.DataFrame({"obs": obs, "pred": pred}).to_csv(
                    pred_dir / f"{target_name}_{model_name}_seed{seed}_frac{args.target_fraction}.csv",
                    index=False,
                )
                pd.DataFrame(rows).to_csv(out_dir / "metrics_raw.csv", index=False)

    raw_metrics = pd.DataFrame(rows)
    summary = (
        raw_metrics.groupby(["target", "model", "target_fraction"], as_index=False)
        .agg(
            NSE_mean=("NSE", "mean"),
            NSE_sd=("NSE", "std"),
            KGE_mean=("KGE", "mean"),
            KGE_sd=("KGE", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_sd=("RMSE", "std"),
            PBIAS_mean=("PBIAS", "mean"),
            PBIAS_sd=("PBIAS", "std"),
            R2_mean=("R2", "mean"),
            R2_sd=("R2", "std"),
        )
        .sort_values(["target", "NSE_mean"], ascending=[True, False])
    )
    raw_metrics.to_csv(out_dir / "metrics_raw.csv", index=False)
    summary.to_csv(out_dir / "metrics_summary.csv", index=False)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)
    print(f"Saved metrics to {out_dir}")


def feature_columns(feature_mode: str, max_rain_cols: int) -> list[str]:
    if feature_mode == "mean_rain":
        return ["log_q", "p_mean", "api", "doy_sin", "doy_cos", "area_log"]
    if feature_mode == "all_rain":
        return (
            ["log_q"]
            + [f"rain_{j:02d}" for j in range(max_rain_cols)]
            + [f"api_{j:02d}" for j in range(max_rain_cols)]
            + ["doy_sin", "doy_cos", "area_log"]
        )
    if feature_mode == "rain_stats":
        return [
            "log_q",
            "p_mean",
            "p_max",
            "p_std",
            "p_wet_frac",
            "api_p_mean",
            "api_p_max",
            "doy_sin",
            "doy_cos",
            "area_log",
        ]
    raise ValueError(f"Unknown feature_mode: {feature_mode}")


def read_local_basins(data_dir: Path, basin_names: list[str], config: ExperimentConfig) -> dict[str, pd.DataFrame]:
    raw = {}
    for basin_name in sorted(set(basin_names)):
        path = data_dir / f"{basin_name}.xlsx"
        if not path.exists():
            raise FileNotFoundError(f"Expected basin file not found: {path}")
        raw[basin_name] = read_basin_excel(path, basin_name, config.feature_mode, config.max_rain_cols)
    return raw


def run_sweep_daf(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    rows = []

    for feature_mode in args.feature_modes:
        for input_len in args.input_lens:
            for hidden_dim in args.hidden_dims:
                for lr in args.lrs:
                    for lambda_max in args.lambda_maxs:
                        for target_weight in args.target_weights:
                            for source_weight in args.source_weights:
                                for domain_weight in args.domain_weights:
                                    for nse_gamma in args.nse_gammas:
                                        for pretrain_epochs in args.pretrain_epochs_grid:
                                            for seed in args.seeds:
                                                config = ExperimentConfig(
                                                    input_len=input_len,
                                                    train_ratio=args.train_ratio,
                                                    target_fraction=args.target_fraction,
                                                    batch_size=args.batch_size,
                                                    hidden_dim=hidden_dim,
                                                    epochs=args.epochs,
                                                    pretrain_epochs=pretrain_epochs,
                                                    lr=lr,
                                                    lambda_max=lambda_max,
                                                    nse_gamma=nse_gamma,
                                                    source_weight=source_weight,
                                                    target_weight=target_weight,
                                                    domain_weight=domain_weight,
                                                    seed=seed,
                                                    device=args.device,
                                                    feature_mode=feature_mode,
                                                    max_rain_cols=args.max_rain_cols,
                                                )
                                                set_seed(seed)
                                                feature_cols = feature_columns(feature_mode, args.max_rain_cols)
                                                raw = read_local_basins(data_dir, [args.source, *args.targets], config)
                                                target_scores = []
                                                for target_name in args.targets:
                                                    target_df = raw[target_name]
                                                    split = int(len(target_df) * args.train_ratio)
                                                    train_pool = target_df.iloc[:split].copy()
                                                    tv_split = int(len(train_pool) * args.inner_train_ratio)
                                                    target_train_df = train_pool.iloc[:tv_split].copy()
                                                    target_val_df = train_pool.iloc[tv_split:].copy()
                                                    if args.target_fraction < 1.0:
                                                        n = max(input_len + 2, int(len(target_train_df) * args.target_fraction))
                                                        target_train_df = target_train_df.iloc[:n].copy()

                                                    source_aligned = align_source_to_target(raw[args.source], target_train_df)
                                                    source_train_df = source_aligned.loc[: target_train_df.index.max()].copy()
                                                    if len(source_train_df) < input_len + 2:
                                                        source_train_df = source_aligned.copy()

                                                    source = build_domain_from_dfs(
                                                        args.source,
                                                        source_train_df,
                                                        source_train_df.tail(max(input_len + 2, 100)),
                                                        config,
                                                        feature_cols,
                                                    )
                                                    target = build_domain_from_dfs(
                                                        target_name, target_train_df, target_val_df, config, feature_cols
                                                    )
                                                    set_seed(seed)
                                                    model = DAFModel(
                                                        len(feature_cols),
                                                        hidden_dim,
                                                        use_domain_specific_v=not args.shared_v,
                                                        use_domain_specific_head=not args.single_head,
                                                    )
                                                    train_daf(model, source.train_dataset, target.train_dataset, config)
                                                    obs, pred = predict_daf(model, target.test_dataset, target.y_scaler, config)
                                                    metrics = calc_metrics(obs, pred)
                                                    target_scores.append(metrics["NSE"])
                                                    row = {
                                                        "seed": seed,
                                                        "source": args.source,
                                                        "target": target_name,
                                                        "feature_mode": feature_mode,
                                                        "input_len": input_len,
                                                        "hidden_dim": hidden_dim,
                                                        "lr": lr,
                                                        "lambda_max": lambda_max,
                                                        "source_weight": source_weight,
                                                        "target_weight": target_weight,
                                                        "domain_weight": domain_weight,
                                                        "nse_gamma": nse_gamma,
                                                        "pretrain_epochs": pretrain_epochs,
                                                        "shared_v": args.shared_v,
                                                        "single_head": args.single_head,
                                                        **metrics,
                                                    }
                                                    rows.append(row)
                                                    pd.DataFrame(rows).to_csv(out_dir / "sweep_raw.csv", index=False)
                                                print(
                                                    "[sweep] "
                                                    f"feature={feature_mode} input={input_len} hidden={hidden_dim} "
                                                    f"lr={lr} lambda={lambda_max} sw={source_weight} tw={target_weight} "
                                                    f"dw={domain_weight} gamma={nse_gamma} "
                                                    f"pre={pretrain_epochs} seed={seed} avg_val_nse={np.nanmean(target_scores):.4f}"
                                                )

    raw_df = pd.DataFrame(rows)
    grouped_cols = [
        "feature_mode",
        "input_len",
        "hidden_dim",
        "lr",
        "lambda_max",
        "source_weight",
        "target_weight",
        "domain_weight",
        "nse_gamma",
        "pretrain_epochs",
        "shared_v",
        "single_head",
    ]
    summary = (
        raw_df.groupby(grouped_cols, as_index=False)
        .agg(NSE_mean=("NSE", "mean"), NSE_sd=("NSE", "std"), KGE_mean=("KGE", "mean"), RMSE_mean=("RMSE", "mean"))
        .sort_values("NSE_mean", ascending=False)
    )
    summary.to_csv(out_dir / "sweep_summary.csv", index=False)
    print(f"Saved DAF sweep to {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HydroDAF public CAMELS-US benchmark runner")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run-group", help="Run one public CAMELS-US source-target group")
    p.add_argument("--data-dir", default="data/camels_us/selected/huc01")
    p.add_argument("--out-dir", default="outputs/huc01")
    p.add_argument("--source", default="01013500")
    p.add_argument("--targets", nargs="+", default=["01022500", "01144000", "01169000"])
    p.add_argument(
        "--models",
        nargs="+",
        default=["local_lstm", "source_only", "finetune_lstm", "regional_lstm", "daf_shared_v", "daf"],
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--target-fraction", type=float, default=0.3)
    p.add_argument("--input-len", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--pretrain-epochs", type=int, default=5)
    p.add_argument("--finetune-epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda-max", type=float, default=0.0001)
    p.add_argument("--nse-gamma", type=float, default=0.1)
    p.add_argument("--source-weight", type=float, default=1.0)
    p.add_argument("--target-weight", type=float, default=4.0)
    p.add_argument("--domain-weight", type=float, default=1.0)
    p.add_argument("--feature-mode", choices=["mean_rain", "all_rain", "rain_stats"], default="mean_rain")
    p.add_argument("--max-rain-cols", type=int, default=12)
    p.add_argument("--device", default="cpu")
    p.set_defaults(func=run_local)

    s = sub.add_parser("sweep-daf", help="Tune DAF hyperparameters on an inner validation split")
    s.add_argument("--data-dir", default="data/camels_us/selected/huc01")
    s.add_argument("--out-dir", default="outputs/daf_sweep")
    s.add_argument("--source", default="01013500")
    s.add_argument("--targets", nargs="+", default=["01022500"])
    s.add_argument("--seeds", nargs="+", type=int, default=[42])
    s.add_argument("--target-fraction", type=float, default=0.3)
    s.add_argument("--train-ratio", type=float, default=0.7)
    s.add_argument("--inner-train-ratio", type=float, default=0.8)
    s.add_argument("--feature-modes", nargs="+", choices=["mean_rain", "all_rain", "rain_stats"], default=["mean_rain"])
    s.add_argument("--input-lens", nargs="+", type=int, default=[60])
    s.add_argument("--hidden-dims", nargs="+", type=int, default=[64])
    s.add_argument("--lrs", nargs="+", type=float, default=[1e-3])
    s.add_argument("--lambda-maxs", nargs="+", type=float, default=[0.002])
    s.add_argument("--target-weights", nargs="+", type=float, default=[1.0])
    s.add_argument("--nse-gammas", nargs="+", type=float, default=[0.2])
    s.add_argument("--pretrain-epochs-grid", nargs="+", type=int, default=[0])
    s.add_argument("--source-weights", nargs="+", type=float, default=[1.0])
    s.add_argument("--domain-weights", nargs="+", type=float, default=[1.0])
    s.add_argument("--batch-size", type=int, default=512)
    s.add_argument("--epochs", type=int, default=20)
    s.add_argument("--max-rain-cols", type=int, default=12)
    s.add_argument("--shared-v", action="store_true")
    s.add_argument("--single-head", action="store_true")
    s.add_argument("--device", default="cpu")
    s.set_defaults(func=run_sweep_daf)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
