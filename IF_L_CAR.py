#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Individual-fitting rolling forecast (OLS-family)
-----------------------------------------------
python IF_L_CAR.py \
  --data_path /home/JunpingZhu/input/com_1m_bar.parquet \
  --model lasso --lag 3 --train_month 1
"""

import argparse, warnings, datetime, numpy as np, pandas as pd, fastparquet
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=FutureWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--data_path",   type=str,
                    default="/home/JunpingZhu/input/com_1m_bar.parquet")
parser.add_argument("--model",      type=str, default="ridge",
                    choices=["ridge","lasso","enet","pls","pcr"])
parser.add_argument("--lag",        type=int, default=3)
parser.add_argument("--train_month",type=int, default=1)
parser.add_argument("--start_date", type=str, default="2018-01-01")
parser.add_argument("--end_date",   type=str, default="2024-12-31")
opt = parser.parse_args()

alpha_grid = np.logspace(-5, 3, 20)

def make_param_spaces(n_feat:int):
    return {
        "ridge": {"est": Pipeline(
                    [("scaler", StandardScaler()), ("ridge", Ridge())]),
                  "grid": {"ridge__alpha": alpha_grid}},
        "lasso": {"est": Pipeline(
                    [("scaler", StandardScaler()),
                     ("lasso",  Lasso(max_iter=50_00))]),
                  "grid": {"lasso__alpha": alpha_grid}},
        "enet" : {"est": Pipeline(
                    [("scaler", StandardScaler()),
                     ("enet",   ElasticNet(max_iter=50_00))]),
                  "grid": {"enet__alpha": alpha_grid,
                           "enet__l1_ratio":[0.1,0.3,0.5,0.7,0.9]}},
        "pls"  : {"est": Pipeline(
                    [("scaler", StandardScaler()),
                     ("pls",    PLSRegression())]),
                  "grid": {"pls__n_components": list(range(1, n_feat+1, 2))}},
        "pcr"  : {"est": Pipeline(
                    [("scaler", StandardScaler()),
                     ("pca",    PCA()),
                     ("lr",     LinearRegression())]),
                  "grid": {"pca__n_components": list(range(1, n_feat+1, 2))}},
    }


def build_super_matrix(df:pd.DataFrame, target_symbol:str, lag:int=3):
    """df:(T,N) → X:(T,N*lag) ，y:(T, )"""
    lagged = [df.shift(i).add_suffix(f"_lag{i}") for i in range(lag)]
    X = pd.concat(lagged, axis=1)
    y = df[target_symbol].shift(-1)
    m = X.notna().all(1) & y.notna()
    return X.loc[m], y.loc[m], X.index[m]

def month_end(d:datetime.date):
    y,m = d.year, d.month
    return datetime.datetime(y+(m==12),(m%12)+1,1)-datetime.timedelta(days=1)

def train_and_select_model(Xtr, ytr, Xva, yva, model_name:str, param_spaces):
    cfg = param_spaces[model_name]
    estimator, grid = cfg["est"], cfg["grid"]

    X_all = pd.concat([Xtr, Xva])
    y_all = pd.concat([ytr, yva])
    test_fold = np.r_[np.zeros(len(Xtr),dtype=int),
                      np.ones (len(Xva),dtype=int)]
    psplit = PredefinedSplit(test_fold)

    gcv = GridSearchCV(estimator,
                       param_grid=grid,
                       scoring="neg_mean_squared_error",
                       cv=psplit, n_jobs=-1, refit=True)
    gcv.fit(X_all, y_all)
    return gcv.best_estimator_

def rolling_forecast_one_symbol(X, y, dt_index,
                                start_date, end_date,
                                train_month:int, model_name:str,
                                param_spaces):
    full = X.assign(y=y.values)
    full.index = dt_index
    preds = []
    cur = start_date

    while True:
        tr_end = month_end(cur + relativedelta(months=train_month))
        va_end = month_end(tr_end + relativedelta(months=1))
        te_end = month_end(va_end + relativedelta(months=1))
        if te_end > end_date: break

        m_tr = (dt_index>=cur)&(dt_index<=tr_end)
        m_va = (dt_index> tr_end)&(dt_index<=va_end)
        m_te = (dt_index> va_end)&(dt_index<=te_end)
        if min(m_tr.sum(),m_va.sum(),m_te.sum())<100:
            cur += relativedelta(months=1); continue

        Xtr,Xva,Xte = full.loc[m_tr].drop(columns="y"),full.loc[m_va].drop(columns="y"),full.loc[m_te].drop(columns="y")
        ytr,yva     = full.loc[m_tr,"y"], full.loc[m_va,"y"]

        scaler = StandardScaler().fit(Xtr)
        Xtr_s,Xva_s,Xte_s = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)

        est = train_and_select_model(pd.DataFrame(Xtr_s,index=Xtr.index),
                                     ytr,
                                     pd.DataFrame(Xva_s,index=Xva.index),
                                     yva,
                                     model_name, param_spaces)

        y_pred = est.predict(Xte_s).ravel()
        preds.append(pd.DataFrame({"datetime": Xte.index,
                                   "y_true":  full.loc[m_te,"y"].values,
                                   "y_pred":  y_pred}))
        print(f"[{cur:%Y-%m}] {model_name} test {(va_end+datetime.timedelta(days=1)):%Y-%m-%d}"
              f"~{te_end:%Y-%m-%d}")
        cur += relativedelta(months=1)

    return pd.concat(preds, ignore_index=True)

def r2_os_zero(df):
    sse_m = ((df.y_true-df.y_pred)**2).sum()
    sse_0 = (df.y_true**2).sum()
    return 1 - sse_m/sse_0


print("Loading data ...")
data = pd.read_parquet(opt.data_path, engine="fastparquet").reset_index()

invalid = {'BC','CJ','EB','EG','FU','I','JM','LG','LH','LU','NR','P','PF','PG',
           'PK','PX','RR','SA','SC','SH','SP','SS','UR','WR','PM','BB','RI',
           'JR','LR','RS','WH'}
data = data[~data['underlying_symbol'].isin(invalid)]
data = data[(data['datetime'].dt.time >= datetime.time(9,0)) &
            (data['datetime'].dt.time <= datetime.time(15,0))]
data = data[data.trading_date >= '2018-01-01']

data["log_close"] = np.log(data["close"])
data["log_ret"]   = data.groupby("underlying_symbol")["log_close"].diff()

pivot = data.pivot(index="datetime", columns="underlying_symbol",
                   values="log_ret").iloc[1:,:]

df = (pivot.reset_index()
             .assign(date=lambda x: x['datetime'].dt.date)
             .groupby('date', group_keys=False).apply(lambda g: g.iloc[1:-1])
             .drop(columns='date')
             .set_index('datetime'))

symbols   = df.columns.tolist()
start_dt  = datetime.datetime.fromisoformat(opt.start_date)
end_dt    = datetime.datetime.fromisoformat(opt.end_date)

all_preds = []
print(f"Start rolling forecast  ‖  model={opt.model}  lag={opt.lag}")
for sym in symbols:
    print(f"\n===>  {sym}  <===")
    X_sym, y_sym, idx = build_super_matrix(df, sym, lag=opt.lag)
    param_spaces = make_param_spaces(X_sym.shape[1])
    preds_sym = rolling_forecast_one_symbol(
        pd.DataFrame(X_sym, index=idx),
        pd.Series(y_sym.values, index=idx),
        idx, start_dt, end_dt,
        opt.train_month, opt.model, param_spaces)
    preds_sym["symbol"] = sym
    print(r2_os_zero(preds_sym))
    all_preds.append(preds_sym)



preds_all = pd.concat(all_preds, ignore_index=True)
out_name  = f"preds_indiv_{opt.model}_lag{opt.lag}_tm{opt.train_month}.parquet"
preds_all.to_parquet(f"/home/JunpingZhu/output/{out_name}", engine="fastparquet")
print(f"\nAll done, saved to  /home/JunpingZhu/output/{out_name}")

print("\n-------- Overall R²_OS vs 0 --------")
print(r2_os_zero(preds_all))