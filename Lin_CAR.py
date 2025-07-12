#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
多输出滚动预测
-------------------------------------------------
- 支持模型：ridge / lasso / enet / pls / pcr
- 超参搜索：GridSearchCV（穷举网格）
"""

import argparse, warnings, datetime, random, numpy as np, pandas as pd, fastparquet
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from math import floor
warnings.filterwarnings("ignore", category=FutureWarning)


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str,
                    default="/home/JunpingZhu/input/com_1m_bar.parquet")
parser.add_argument("--model", type=str, default="ridge",
                    choices=["ridge", "lasso", "enet", "pls", "pcr"])
parser.add_argument("--lag", type=int, default=3)
parser.add_argument("--train_month", type=int, default=1)
parser.add_argument("--start_date", type=str, default="2018-01-01")
parser.add_argument("--end_date", type=str,   default="2024-12-31")
opt = parser.parse_args()

# ────────────────────────── 超参数网格 ───────────────────────────
alpha_grid  = np.logspace(-2, 3, 10)


def make_param_spaces(n_feat: int):
    return {
        "ridge": {"est": MultiOutputRegressor(Ridge()),      "grid": {"estimator__alpha": alpha_grid}},
        "lasso": {"est": MultiOutputRegressor(Lasso(max_iter=5000)), "grid": {"estimator__alpha": alpha_grid}},
        "enet" : {"est": MultiOutputRegressor(ElasticNet(max_iter=5000)),
                  "grid": {"estimator__alpha": alpha_grid,
                           "estimator__l1_ratio": [0.1,0.3,0.5,0.7,0.9]}},
        "pls"  : {"est": PLSRegression(),        "grid": {"n_components": list(range(1, n_feat + 1, 2))}},
        "pcr"  : {"est": None,                   "grid": {"n_components": list(range(1, n_feat + 1, 2))}}, 
    }


def build_multi_matrix(df: pd.DataFrame, lag: int = 3):
    lags = [df.shift(i).add_suffix(f"_lag{i}") for i in range(lag)]
    X = pd.concat(lags, axis=1)
    Y = df.shift(-1)
    m = X.notna().all(1) & Y.notna().all(1)
    return X.loc[m], Y.loc[m]


def train_and_select_model(Xtr, ytr, Xva, yva, model_name: str):
    cfg = param_spaces[model_name]
    if model_name == "pcr":
        def make_pcr(n_comp):
            return MultiOutputRegressor(
                pipe := Pipeline([("pca", PCA(n_components=n_comp)),
                                  ("lr",  LinearRegression())])
            )

        best_est, best_mse = None, np.inf
        for n in cfg["grid"]["n_components"]:
            est = make_pcr(n)
            est.fit(Xtr, ytr)
            mse = np.mean((est.predict(Xva) - yva)**2)
            if mse < best_mse:
                best_mse, best_est = mse, est
        return best_est

    base_est, grid = cfg["est"], cfg["grid"]
    psplit = PredefinedSplit(test_fold=np.r_[np.zeros(len(Xtr)), np.ones(len(Xva))])
    X_all, y_all = pd.concat([Xtr, Xva]), pd.concat([ytr, yva])

    gsearch = GridSearchCV(base_est, param_grid=grid,
                           scoring="neg_mean_squared_error",
                           cv=psplit, refit=True, n_jobs=-1)
    gsearch.fit(X_all, y_all)
    return gsearch.best_estimator_


def month_end(d):
    y, m = d.year, d.month
    return datetime.datetime(y+(m==12), (m%12)+1, 1) - datetime.timedelta(days=1)


def rolling_forecast_multi(X, Y, start_date, end_date,
                           train_month=1, target_model="ridge"):
    dt_index = X.index
    y_cols   = [f"y_{c}" for c in Y.columns]
    full_df  = pd.concat([Y.add_prefix("y_"), X], axis=1)

    preds_ls, cur = [], start_date
    while True:
        tr_end = month_end(cur + relativedelta(months=train_month))
        va_end = month_end(tr_end + relativedelta(months=1))
        te_end = month_end(va_end + relativedelta(months=1))
        if te_end > end_date: break

        m_tr = (dt_index>=cur) & (dt_index<=tr_end)
        m_va = (dt_index>tr_end) & (dt_index<=va_end)
        m_te = (dt_index>va_end) & (dt_index<=te_end)
        if min(m_tr.sum(), m_va.sum(), m_te.sum()) < 100:
            cur += relativedelta(months=1); continue

        Xtr, Xva, Xte = full_df.loc[m_tr, X.columns], full_df.loc[m_va, X.columns], full_df.loc[m_te, X.columns]
        Ytr, Yva      = full_df.loc[m_tr, y_cols],    full_df.loc[m_va, y_cols]

        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xva_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)

        est = train_and_select_model(
            pd.DataFrame(Xtr_s, index=Xtr.index),
            Ytr, pd.DataFrame(Xva_s, index=Xva.index), Yva,
            model_name=target_model
        )
        y_pred = est.predict(Xte_s)
        preds_ls.append(pd.DataFrame(y_pred, index=Xte.index, columns=Y.columns))

        print(f"[{cur:%Y-%m}] test {(va_end+datetime.timedelta(days=1)):%Y-%m-%d} ~ {te_end:%Y-%m-%d}")
        cur += relativedelta(months=1)

    return pd.concat(preds_ls).sort_index()

if __name__ == "__main__":
    path = opt.data_path
    com  = pd.read_parquet(path, engine="fastparquet").reset_index()

    invalid = {'BC','CJ','EB','EG','FU','I','JM','LG','LH','LU','NR','P','PF','PG',
               'PK','PX','RR','SA','SC','SH','SP','SS','UR','WR','PM','BB','RI',
               'JR','LR','RS','WH'}
    com = com[~com['underlying_symbol'].isin(invalid)]
    com = com[(com['datetime'].dt.time >= datetime.time(9,0)) &
              (com['datetime'].dt.time <= datetime.time(15,0))]
    com = com[com.trading_date >= '2018-01-01']

    com["log_close"] = np.log(com["close"])
    com["log_ret"]   = com.groupby("underlying_symbol")["log_close"].diff()

    pivot = com.pivot(index="datetime", columns="underlying_symbol",
                      values="log_ret").iloc[1:,:]

    df = (pivot.reset_index()
                 .assign(date=lambda x: x['datetime'].dt.date)
                 .groupby('date', group_keys=False).apply(lambda g: g.iloc[1:-1])
                 .drop(columns='date')
                 .set_index('datetime'))

    X, Y         = build_multi_matrix(df, lag=opt.lag)
    param_spaces = make_param_spaces(X.shape[1])
    start_dt     = datetime.datetime.fromisoformat(opt.start_date)
    end_dt       = datetime.datetime.fromisoformat(opt.end_date)

    preds = rolling_forecast_multi(X, Y,
                                   start_date=start_dt,
                                   end_date=end_dt,
                                   train_month=opt.train_month,
                                   target_model=opt.model)

    out_name = f"preds_{opt.model}_lag{opt.lag}_tm{opt.train_month}.parquet"
    preds.to_parquet(f"/home/JunpingZhu/output/{out_name}", engine="fastparquet")
    print(f"Done! 结果保存为 {out_name}")