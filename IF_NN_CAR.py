#!/usr/bin/env python3
"""
Individual-fitting rolling forecast with NN
-----------------------------------------------

python IF_NN_CAR.py \
  --data_path /home/JunpingZhu/input/com_1m_bar.parquet \
  --model NN2 --lag 3 --train_month 1
"""

import argparse, os, random, datetime, warnings, numpy as np, pandas as pd, torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler

os.environ["OMP_NUM_THREADS"] = os.getenv("OMP_NUM_THREADS", "32")
os.environ["MKL_NUM_THREADS"] = os.getenv("MKL_NUM_THREADS", "32")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
num_threads = int(os.environ["OMP_NUM_THREADS"])
torch.set_num_threads(num_threads)        # math kernel threads
torch.set_num_interop_threads(1)          # dispatcher threads

param_grid = {
    "epochs" : [50, 100],
    "dropout": [0.0, 0.1, 0.2],
    "max_w2" : [10., 100., 1000., np.finfo(np.float32).max],
    "l1"     : [0.0, 1e-5, 1e-4],
    "l2"     : [0.0, 1e-5, 1e-4],
    "rho"    : [0.90, 0.95, 0.99, 0.999],
    "eps"    : [1e-10, 1e-8, 1e-6, 1e-4],
}

depth_cfg = {
    "NN1": [32],
    "NN2": [32, 16],
    "NN3": [32, 16, 8],
    "NN4": [32, 16, 8, 4],
    "NN5": [32, 16, 8, 4, 2],
}
AMP_ENABLE = True

# ───────────────────────────────── argparse ─────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str,
                    default="/home/JunpingZhu/input/com_1m_bar.parquet")
parser.add_argument("--model", type=str, default="NN2",
                    choices=list(depth_cfg.keys()), help="选择网络深度结构")
parser.add_argument("--lag", type=int, default=3, help="滞后阶数")
parser.add_argument("--train_month", type=int, default=1,
                    help="训练窗口（月），验证+测试各固定1个月")
parser.add_argument("--start_date", type=str, default="2018-01-01")
parser.add_argument("--end_date", type=str, default="2024-12-31")
parser.add_argument("--n_iter", type=int, default=5,
                    help="随机搜索迭代次数")
parser.add_argument("--n_restart", type=int, default=3,
                    help="最佳配置下的重训练次数(ensemble size)")
opt = parser.parse_args()

# ─────────────────────────── NN helper functions ───────────────────────────────

def _block(sizes, drop):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1]),
                   nn.BatchNorm1d(sizes[i + 1]),
                   nn.ReLU(inplace=True)]
        if drop > 0:
            layers.append(nn.Dropout(drop))
    return nn.Sequential(*layers)

def make_net(name: str, in_dim: int, drop: float) -> nn.Module:
    hidden = depth_cfg[name]
    return nn.Sequential(_block([in_dim] + hidden, drop), nn.Linear(hidden[-1], 1))

def train_one(model, X, y, Xv, yv, cfg, batch=2048):
    crit = nn.SmoothL1Loss(beta=0.999)
    opti = torch.optim.Adadelta(model.parameters(), rho=cfg["rho"], eps=cfg["eps"],
                                weight_decay=cfg["l2"])
    scaler = torch.cuda.amp.GradScaler(enabled=AMP_ENABLE)
    best_state, best, wait = model.state_dict(), np.inf, 0

    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch, shuffle=False)

    Xv = torch.tensor(Xv, dtype=torch.float32, device=DEVICE)
    yv = torch.tensor(yv, dtype=torch.float32, device=DEVICE)

    for _ in range(cfg["epochs"]):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opti.zero_grad()
            with torch.cuda.amp.autocast(enabled=AMP_ENABLE):
                pred = model(xb).squeeze(dim=1)
                loss = crit(pred, yb)
                if cfg["l1"] > 0:
                    loss += cfg["l1"] * sum(p.abs().sum() for p in model.parameters())
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["max_w2"])
            scaler.step(opti)
            scaler.update()

        # ─ validation
        model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=AMP_ENABLE):
            vloss = crit(model(Xv).squeeze(dim=1), yv).item()
        if vloss < best - 1e-6:
            best, best_state, wait = vloss, model.state_dict(), 0
        else:
            wait += 1
            if wait >= 10:
                break

    model.load_state_dict(best_state)
    return best

def fit_best(name, Xtr, ytr, Xva, yva, n_iter=5, n_restart=3):
    """Random search + ensemble retraining (individual target)."""
    best, best_cfg = np.inf, None
    for _ in range(n_iter):
        cfg = {k: random.choice(v) for k, v in param_grid.items()}
        net = make_net(name, Xtr.shape[1], cfg["dropout"]).to(DEVICE)
        vloss = train_one(net, Xtr, ytr, Xva, yva, cfg)
        if vloss < best:
            best, best_cfg = vloss, cfg

    XY = np.vstack([Xtr, Xva])
    y_full = np.hstack([ytr, yva])
    preds, models = [], []
    for seed in range(n_restart):
        torch.manual_seed(seed)
        net = make_net(name, Xtr.shape[1], best_cfg["dropout"]).to(DEVICE)
        train_one(net, XY, y_full, Xva, yva, best_cfg)
        models.append(net)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=AMP_ENABLE):
            preds.append(net(torch.tensor(XY, dtype=torch.float32, device=DEVICE))
                         .squeeze(dim=1).cpu().numpy())
    return models, np.mean(preds, axis=0)


def build_super_matrix(df: pd.DataFrame, tgt: str, lag: int = 3):
    """Create feature matrix X(t)=[returns_{all symbols, lags 0..lag-1}], target y(t)=R_{t+1}(tgt)."""
    lagged = [df.shift(i).add_suffix(f"_lag{i}") for i in range(lag)]
    X = pd.concat(lagged, axis=1)
    y = df[tgt].shift(-1)
    m = X.notna().all(1) & y.notna()
    return X.loc[m], y.loc[m], X.index[m]

def month_end(d: datetime.date):
    y, m = d.year, d.month
    return datetime.datetime(y + (m == 12), (m % 12) + 1, 1) - datetime.timedelta(days=1)

# ─────────────────────────────── main rolling ─────────────────────────────────


def rolling_indiv(df: pd.DataFrame, symbol: str, lag: int, model_name: str,
                  train_months: int, start_date: datetime.datetime,
                  end_date: datetime.datetime, n_iter: int, n_restart: int):
    """One-symbol rolling forecast returning a DataFrame."""
    X, y, idx = build_super_matrix(df, symbol, lag)

    # Pre-allocate scalers outside the loop (re-fit each step)
    preds_list, cur = [], start_date
    while True:
        tr_end = month_end(cur + relativedelta(months=train_months))
        va_end = month_end(tr_end + relativedelta(months=1))
        te_end = month_end(va_end + relativedelta(months=1))
        if te_end > end_date:
            break

        m_tr = (idx >= cur) & (idx <= tr_end)
        m_va = (idx > tr_end) & (idx <= va_end)
        m_te = (idx > va_end) & (idx <= te_end)
        if min(m_tr.sum(), m_va.sum(), m_te.sum()) < 100:
            cur += relativedelta(months=1)
            continue

        scaler_X = StandardScaler().fit(X[m_tr])
        Xtr = scaler_X.transform(X[m_tr])
        Xva = scaler_X.transform(X[m_va])
        Xte = scaler_X.transform(X[m_te])

        y_mean, y_std = y[m_tr].mean(), y[m_tr].std()
        y_std = 1.0 if y_std == 0 else y_std
        ytr = ((y[m_tr] - y_mean) / y_std).values
        yva = ((y[m_va] - y_mean) / y_std).values

        models, _ = fit_best(model_name, Xtr, ytr, Xva, yva,
                             n_iter=n_iter, n_restart=n_restart)

        Xte_t = torch.tensor(Xte, dtype=torch.float32, device=DEVICE)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=AMP_ENABLE):
            preds = np.mean([m(Xte_t).squeeze(dim=1).cpu().numpy() for m in models], axis=0)
        y_pred = preds * y_std + y_mean  # inverse standardisation

        preds_df = pd.DataFrame({"datetime": idx[m_te],
                                 "y_true": y[m_te].values,
                                 "y_pred": y_pred})
        preds_list.append(preds_df)
        print(f"[{cur:%Y-%m}] {symbol}  test {(va_end + datetime.timedelta(days=1)):%Y-%m-%d} ~ {te_end:%Y-%m-%d}")
        cur += relativedelta(months=1)

    return pd.concat(preds_list, ignore_index=True)


def r2_os_zero(df: pd.DataFrame):
    """Out-of-sample R² relative to zero-return benchmark."""
    sse_m = ((df.y_true - df.y_pred) ** 2).sum()
    sse_0 = (df.y_true ** 2).sum()
    return 1 - sse_m / sse_0


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)

    print("Loading data ...")
    import fastparquet
    raw = pd.read_parquet(opt.data_path, engine="fastparquet").reset_index()

    # ─ symbol filtering and intraday time window ─
    invalid = {"BC", "CJ", "EB", "EG", "FU", "I", "JM", "LG", "LH", "LU", "NR", "P",
               "PF", "PG", "PK", "PX", "RR", "SA", "SC", "SH", "SP", "SS", "UR", "WR",
               "PM", "BB", "RI", "JR", "LR", "RS", "WH"}
    raw = raw[~raw["underlying_symbol"].isin(invalid)]
    raw = raw[(raw["datetime"].dt.time >= datetime.time(9, 0)) &
              (raw["datetime"].dt.time <= datetime.time(15, 0))]
    raw = raw[raw.trading_date >= "2018-01-01"]

    raw["log_close"] = np.log(raw["close"])
    raw["log_ret"] = raw.groupby("underlying_symbol")["log_close"].diff()

    pivot = raw.pivot(index="datetime", columns="underlying_symbol", values="log_ret").iloc[1:]

    # remove first/last minute each day
    df = (pivot.reset_index()
              .assign(date=lambda x: x.datetime.dt.date)
              .groupby("date", group_keys=False).apply(lambda g: g.iloc[1:-1])
              .drop(columns="date")
              .set_index("datetime"))

    symbols = df.columns.tolist()
    start_dt = datetime.datetime.fromisoformat(opt.start_date)
    end_dt = datetime.datetime.fromisoformat(opt.end_date)

    all_preds = []
    print(f"Start rolling forecast  |  model={opt.model}  lag={opt.lag}")
    for sym in symbols:
        print(f"\n===>  {sym}  <===")
        preds_sym = rolling_indiv(df, sym, lag=opt.lag, model_name=opt.model,
                                  train_months=opt.train_month, start_date=start_dt,
                                  end_date=end_dt, n_iter=opt.n_iter,
                                  n_restart=opt.n_restart)
        preds_sym["symbol"] = sym
        print("R²_OS:", r2_os_zero(preds_sym))
        all_preds.append(preds_sym)

    preds_all = pd.concat(all_preds, ignore_index=True)
    out_name = f"preds_indiv_{opt.model}_lag{opt.lag}_tm{opt.train_month}.parquet"
    preds_all.to_parquet(f"/home/JunpingZhu/output/{out_name}", engine="fastparquet")
    print(f"\nAll done, saved to /home/JunpingZhu/output/{out_name}")

    print("\n-------- Overall R²_OS vs 0 --------")
    print(r2_os_zero(preds_all))