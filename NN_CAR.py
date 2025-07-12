import argparse
import os, random, datetime, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler

os.environ["OMP_NUM_THREADS"]  = "32"  
os.environ["MKL_NUM_THREADS"]  = "32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
num = int(os.getenv("OMP_NUM_THREADS", "32"))
torch.set_num_threads(num)          # 运算线程
torch.set_num_interop_threads(1)    # 算子调度线程


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

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="NN2",
                    choices=["NN1","NN2","NN3","NN4","NN5"],
                    help="选择网络深度结构")
opt = parser.parse_args()

def build_multi_matrix(df: pd.DataFrame, lag: int = 3):
    """将 (T, N) price/return 表 -> X=(T, N*lag), Y=(T, N)"""
    lagged = [df.shift(i).add_suffix(f"_lag{i}") for i in range(lag)]
    X = pd.concat(lagged, axis=1)
    Y = df.shift(-1)                          # t+1 return
    m = X.notna().all(1) & Y.notna().all(1)
    return X.loc[m].values, Y.loc[m].values, X.index[m]


def _block(sizes, drop):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i+1]),
                   nn.BatchNorm1d(sizes[i+1]),
                   nn.ReLU(inplace=True)]
        if drop > 0:
            layers.append(nn.Dropout(drop))
    return nn.Sequential(*layers)

def make_net(name, in_dim, out_dim, drop):
    hidden = depth_cfg[name]           
    return nn.Sequential(
        _block([in_dim] + hidden, drop),
        nn.Linear(hidden[-1], out_dim)
    )


def train_one(model, X, Y, Xv, Yv, cfg, batch_size=2048):
    crit = nn.SmoothL1Loss(beta=0.999)
    opt  = torch.optim.Adadelta(model.parameters(),
                                rho=cfg["rho"],
                                eps=cfg["eps"],
                                weight_decay=cfg["l2"])
    scaler = torch.cuda.amp.GradScaler(enabled=AMP_ENABLE)
    best_state, best, wait = model.state_dict(), np.inf, 0

    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(Y, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    Xv = torch.tensor(Xv, dtype=torch.float32, device=DEVICE)
    Yv = torch.tensor(Yv, dtype=torch.float32, device=DEVICE)

    for epoch in range(cfg["epochs"]):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=AMP_ENABLE):
                pred = model(xb)
                loss = crit(pred, yb)
                if cfg["l1"] > 0:
                    loss += cfg["l1"] * sum(p.abs().sum()
                                            for p in model.parameters())
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["max_w2"])
            scaler.step(opt)
            scaler.update()

        # ─ validation
        model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=AMP_ENABLE):
            vloss = crit(model(Xv), Yv).item()
        if vloss < best - 1e-6:
            best, best_state, wait = vloss, model.state_dict(), 0
        else:
            wait += 1
            if wait >= 10: break

    model.load_state_dict(best_state)
    return best

def fit_best(name, Xtr, Ytr, Xva, Yva, n_iter=5, n_restart=3):
    best, best_cfg = np.inf, None
    for _ in range(n_iter):
        cfg = {k: random.choice(v) for k, v in param_grid.items()}
        net = make_net(name, Xtr.shape[1], Ytr.shape[1], cfg["dropout"]).to(DEVICE)
        vloss = train_one(net, Xtr, Ytr, Xva, Yva, cfg)
        if vloss < best:
            best, best_cfg = vloss, cfg

    XY, Y = np.vstack([Xtr, Xva]), np.vstack([Ytr, Yva])
    preds, models = [], []
    for seed in range(n_restart):
        torch.manual_seed(seed)
        net = make_net(name, Xtr.shape[1], Y.shape[1], best_cfg["dropout"]).to(DEVICE)
        train_one(net, XY, Y, Xva, Yva, best_cfg)
        models.append(net)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=AMP_ENABLE):
            preds.append(net(torch.tensor(XY, dtype=torch.float32,
                                          device=DEVICE)).cpu().numpy())
    ensemble_pred = np.mean(preds, axis=0)
    return models, ensemble_pred


def month_end(d):
    y, m = d.year, d.month
    return datetime.datetime(y + (m == 12), (m % 12) + 1, 1) - datetime.timedelta(days=1)

def rolling_multi(df, lag=3,
                  start_date=datetime.datetime(2018, 1, 1),
                  end_date=datetime.datetime(2024, 12, 31),
                  model_name="NN2"):

    X_raw, Y_raw, idx = build_multi_matrix(df, lag)
    data_X = pd.DataFrame(X_raw, index=idx, columns=[f"{c}" for c in range(X_raw.shape[1])])
    data_Y = pd.DataFrame(Y_raw, index=idx, columns=df.columns)

    preds_list, cur = [], start_date
    while True:
        tr_end  = month_end(cur + relativedelta(months=2))
        va_end  = month_end(tr_end + relativedelta(months=1))
        te_end  = month_end(va_end + relativedelta(months=1))
        if te_end > end_date: break

        mask_tr = (idx >= cur) & (idx <= tr_end)
        mask_va = (idx >  tr_end) & (idx <= va_end)
        mask_te = (idx >  va_end) & (idx <= te_end)
        if min(mask_tr.sum(), mask_va.sum(), mask_te.sum()) < 100:
            cur += relativedelta(months=1); continue

        # 标准化
        x_scaler = StandardScaler().fit(data_X[mask_tr])
        X_tr, X_va, X_te = (x_scaler.transform(data_X[m]) for m in (mask_tr, mask_va, mask_te))
        y_mean, y_std = data_Y[mask_tr].mean(), data_Y[mask_tr].std().replace(0, 1.0)
        Y_tr = (data_Y[mask_tr] - y_mean) / y_std
        Y_va = (data_Y[mask_va] - y_mean) / y_std

        # 训练
        models, _ = fit_best(model_name, X_tr, Y_tr.values, X_va, Y_va.values)

        # 预测集成
        Xte_t = torch.tensor(X_te, dtype=torch.float32, device=DEVICE)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=AMP_ENABLE):
            preds = np.mean([m(Xte_t).cpu().numpy() for m in models], axis=0)
        y_pred = preds * y_std.values + y_mean.values  # 反标准化

        preds_df = pd.DataFrame(y_pred, index=idx[mask_te], columns=df.columns)
        preds_list.append(preds_df)
        print(f"[{cur:%Y-%m}] test {(va_end+datetime.timedelta(days=1)):%Y-%m-%d}"
              f" ~ {te_end:%Y-%m-%d}")
        cur += relativedelta(months=1)

    return pd.concat(preds_list)


if __name__ == "__main__":
    import fastparquet, warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    path = "/home/JunpingZhu/input/com_1m_bar.parquet"
    com = pd.read_parquet(path, engine="fastparquet")
    com.reset_index(inplace=True)

    invalid = {'BC','CJ','EB','EG','FU','I','JM','LG','LH','LU','NR','P','PF','PG',
               'PK','PX','RR','SA','SC','SH','SP','SS','UR','WR','PM','BB','RI',
               'JR','LR','RS','WH'}
    com = com[~com['underlying_symbol'].isin(invalid)]
    com = com[(com['datetime'].dt.time >= datetime.time(9,0)) &
              (com['datetime'].dt.time <= datetime.time(15,0))]
    com = com[com.trading_date >= '2018-01-01']

    com['log_close'] = np.log(com['close'])
    com['log_ret']   = com.groupby('underlying_symbol')['log_close'].diff()
    pivot = com.pivot(index="datetime", columns="underlying_symbol",
                      values="log_ret").iloc[1:,:]

    # （可选 Polars 加速）
    # import polars as pl
    # pivot = (pl.from_pandas(com)
    #            .with_columns(pl.col('close').log().diff()
    #                          .over('underlying_symbol').alias('log_ret'))
    #            .pivot(values='log_ret', index='datetime',
    #                   columns='underlying_symbol')
    #            .to_pandas())

    # 切掉日内首尾 1 分钟
    df = pivot.copy()
    df.reset_index(inplace=True)
    df['date'] = df['datetime'].dt.date
    df = df.groupby('date', group_keys=False).apply(lambda g: g.iloc[1:-1])
    df.drop(columns='date', inplace=True)
    df.set_index('datetime', inplace=True)

    print("Symbols:", list(df.columns))
    preds = rolling_multi(df, lag=3,
                          start_date=datetime.datetime(2018,1,1),
                          end_date  =datetime.datetime(2024,12,31),
                          model_name=opt.model)
    preds.to_parquet(
        f"/home/JunpingZhu/output/all_preds_multi{opt.model}.parquet",
        engine="fastparquet"
    )
    print("Done! 预测结果已保存。")