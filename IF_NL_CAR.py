#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Individual fitting with RandomizedSearchCV
-----------------------------------------
python IF_L_CAR.py \
  --data_path /home/JunpingZhu/input/com_1m_bar.parquet \
  --model rf --lag 3 --train_month 1
"""

import argparse, warnings, datetime, numpy as np, pandas as pd, fastparquet
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
from sklearn.multioutput     import MultiOutputRegressor
from sklearn.linear_model    import Ridge, Lasso, ElasticNet
from sklearn.ensemble        import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from math import floor
warnings.filterwarnings("ignore", category=FutureWarning)


parser = argparse.ArgumentParser()
parser.add_argument("--data_path",   type=str, default="/home/JunpingZhu/com_1m_bar.parquet")
parser.add_argument("--model",       type=str, default="rf",
                    choices=["rf","xrf","gbt","ridge","lasso","enet"])
parser.add_argument("--lag",         type=int, default=3)
parser.add_argument("--train_month", type=int, default=1)
parser.add_argument("--start_date",  type=str, default="2018-01-01")
parser.add_argument("--end_date",    type=str, default="2024-12-31")
opt = parser.parse_args()


max_depth = list(range(1, 20))
def col_frac_grid(n_feat): return [max(1, floor(n_feat*f)) for f in [0.05,0.15,0.25,0.333,0.4]]
def make_param_spaces(n_feat:int):
    trees  = list(range(100, 701, 100))
    depths = list(range(1, 41, 5))
    leaves = [1,25,50]
    maxfea = col_frac_grid(n_feat)
    return {
        "rf" : {"est":RandomForestRegressor(n_jobs=-1,bootstrap=True,max_samples=0.5,random_state=42),
                "grid":{"n_estimators":trees,"max_depth":depths,"min_samples_leaf":leaves,"max_features":maxfea}},
        "xrf": {"est":ExtraTreesRegressor(n_jobs=-1,bootstrap=False,random_state=42),
                "grid":{"n_estimators":trees,"max_depth":depths,"min_samples_leaf":leaves,"max_features":maxfea}},
        "gbt": {"est":GradientBoostingRegressor(loss="huber",alpha=0.999,subsample=0.5,
                                                random_state=42,n_iter_no_change=50),
                "grid":{"n_estimators":trees,"max_depth":[1,2],"learning_rate":[0.001,0.01,0.1]}}
    }

def build_super_matrix(df:pd.DataFrame, target:str, lag:int=3):
    lags = [df.shift(i).add_suffix(f"_lag{i}") for i in range(lag)]
    X = pd.concat(lags, axis=1)
    y = df[target].shift(-1)
    m = X.notna().all(1) & y.notna()
    return X.loc[m], y.loc[m], X.index[m]

def month_end(d):
    y,m = d.year,d.month
    return datetime.datetime(y+(m==12),(m%12)+1,1)-datetime.timedelta(days=1)

def train_and_select_model(Xtr,ytr,Xva,yva, model_name:str, n_iter:int=10):
    cfg = param_spaces[model_name]
    est0, grid = cfg["est"], cfg["grid"]

    ps  = PredefinedSplit(test_fold=np.r_[np.zeros(len(Xtr)), np.ones(len(Xva))])
    X_all, y_all = pd.concat([Xtr,Xva]), pd.concat([ytr,yva])

    search = RandomizedSearchCV(est0, param_distributions=grid, n_iter=n_iter,
                                scoring="neg_mean_squared_error", cv=ps,
                                n_jobs=-1, refit=False)
    search.fit(X_all, y_all)
    best = est0.set_params(**search.best_params_)
    best.fit(X_all, y_all)
    return best

def rolling_forecast(X,y,idx,start_dt,end_dt,train_m,model):
    full = pd.concat([X, y.rename("y")], axis=1)
    preds, cur = [], start_dt
    while True:
        tr_end = month_end(cur + relativedelta(months=train_m))
        va_end = month_end(tr_end + relativedelta(months=1))
        te_end = month_end(va_end + relativedelta(months=1))
        if te_end > end_dt: break

        m_tr = (idx>=cur)&(idx<=tr_end)
        m_va = (idx>tr_end)&(idx<=va_end)
        m_te = (idx>va_end)&(idx<=te_end)
        if min(m_tr.sum(),m_va.sum(),m_te.sum())<100:
            cur += relativedelta(months=1); continue

        Xtr,Xva,Xte = full.loc[m_tr, X.columns], full.loc[m_va,X.columns], full.loc[m_te,X.columns]
        ytr,yva,yte = full.loc[m_tr,"y"],        full.loc[m_va,"y"],        full.loc[m_te,"y"]

        scaler = StandardScaler().fit(Xtr)
        Xtr_s,Xva_s,Xte_s = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)

        est = train_and_select_model(pd.DataFrame(Xtr_s,index=Xtr.index),ytr,
                                     pd.DataFrame(Xva_s,index=Xva.index),yva,
                                     model_name=model)
        y_pred = est.predict(Xte_s)
        preds.append(pd.DataFrame({"datetime":Xte.index,
                                   "y_true":yte.values,
                                   "y_pred":y_pred}))
        print(f"[{cur:%Y-%m}]  {te_end:%Y-%m} done")
        cur += relativedelta(months=1)

    return pd.concat(preds, ignore_index=True)


def r2_os_zero(df):
    sse_m = ((df.y_true-df.y_pred)**2).sum()
    sse_0 = (df.y_true**2).sum()
    return 1 - sse_m/sse_0

if __name__ == "__main__":
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
        X,y,idx = build_super_matrix(df, sym, lag=opt.lag)
        param_spaces = make_param_spaces(X.shape[1])
        preds   = rolling_forecast(X,y,idx,start_dt,end_dt,
                                   train_m=opt.train_month,
                                   model=opt.model)
        preds["symbol"] = sym
        print(r2_os_zero(preds))
        all_preds.append(preds)

    preds_all = pd.concat(all_preds, ignore_index=True)
    out_name  = f"preds_indiv_{opt.model}_lag{opt.lag}_tm{opt.train_month}.parquet"
    preds_all.to_parquet(f"/home/JunpingZhu/output/{out_name}", engine="fastparquet")
    print(f"\nAll done, saved to  /home/JunpingZhu/output/{out_name}")

    print("\n-------- Overall R²_OS vs 0 --------")
    print(r2_os_zero(preds_all))