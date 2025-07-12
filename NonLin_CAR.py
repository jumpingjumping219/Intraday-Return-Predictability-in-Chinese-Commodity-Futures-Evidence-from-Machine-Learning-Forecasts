import argparse
import numpy as np, pandas as pd, datetime, random, warnings, fastparquet
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from math import floor
warnings.filterwarnings("ignore", category=FutureWarning)


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str,
                    default="/home/JunpingZhu/com_1m_bar.parquet",
                    help="Parquet 文件路径")
parser.add_argument("--model", type=str, default="rf",
                    choices=["rf","xrf","gbt","ridge","lasso","enet"],
                    help="选择拟合模型")
parser.add_argument("--lag", type=int, default=3,
                    help="滞后阶数 (lag)")
parser.add_argument("--train_month", type=int, default=1,
                    help="每个滚动窗口的训练月份数")
parser.add_argument("--start_date", type=str, default='2018-01-01',
                    help="开始训练时间")
parser.add_argument("--end_date", type=str, default='2024-12-31',
                    help="结束时间")
opt = parser.parse_args()


max_depth   = list(range(1, 20))
def col_frac_grid(n_feat):
    fracs = [0.05, 0.15, 0.25, 0.333, 0.4]
    return [max(1, floor(n_feat * f)) for f in fracs]


def make_param_spaces(n_feat: int):
    trees_grid  = list(range(100, 701, 100))   
    depth_grid  = list(range(1, 41, 5))        
    min_rows    = [1, 25, 50]                  
    max_feat_int= col_frac_grid(n_feat)

    spaces = {
        # ── RF ──
        "rf": {
            "est": RandomForestRegressor(
                n_jobs=-1, bootstrap=True, max_samples=0.5, random_state=42
            ),
            "grid": {
                "n_estimators"     : trees_grid,
                "max_depth"        : depth_grid,
                "min_samples_leaf" : min_rows,
                "max_features"     : max_feat_int,
            },
        },

        # ── XRF ──
        "xrf": {
            "est": ExtraTreesRegressor(
                n_jobs=-1, bootstrap=False, random_state=42
            ),
            "grid": {
                "n_estimators"     : trees_grid,
                "max_depth"        : depth_grid,
                "min_samples_leaf" : min_rows,
                "max_features"     : max_feat_int,
            },
        },

        # ── GBT ──（
        "gbt": {
            "est": GradientBoostingRegressor(
                loss="huber",             
                alpha=0.999,             
                subsample=0.5,
                random_state=42,
                n_iter_no_change=50
            ),
            "grid": {
                "n_estimators"   : trees_grid,
                "max_depth"      : [1, 2],
                "learning_rate"  : [0.001, 0.01, 0.1],
            },
        },
    }
    return spaces

def build_multi_matrix(df: pd.DataFrame, lag:int=3):
    """df:(T,N) →  X:(T, N*lag) , Y:(T,N)"""
    lags = [df.shift(i).add_suffix(f"_lag{i}") for i in range(lag)]
    X = pd.concat(lags, axis=1)
    Y = df.shift(-1)                       
    m = X.notna().all(1) & Y.notna().all(1)
    return X.loc[m], Y.loc[m]              

def train_and_select_model(Xtr, ytr, Xva, yva, model_name="ridge", n_iter=10):
    cfg = param_spaces[model_name]
    base_est = cfg["est"]

    if model_name in {"ridge", "lasso", "enet"}:
        base_est_final = MultiOutputRegressor(base_est, n_jobs=-1)
        grid_final = {f'estimator__{k}': v for k, v in cfg["grid"].items()}
    else:                  # rf / xrf / gbt 直接用
        base_est_final = base_est
        grid_final = cfg["grid"]

    psplit = PredefinedSplit(
        test_fold=np.r_[np.zeros(len(Xtr)), np.ones(len(Xva))]
    )
    X_all = pd.concat([Xtr, Xva])
    y_all = pd.concat([ytr, yva])


    search = RandomizedSearchCV(
        base_est_final,
        param_distributions=grid_final,
        n_iter=n_iter,
        scoring="neg_mean_squared_error",
        cv=psplit,
        n_jobs=-1,
        refit=False
    )
    search.fit(X_all, y_all)
    best_params = search.best_params_

    best_est = base_est_final.set_params(**best_params)
    best_est.fit(X_all, y_all)
    return best_est

def month_end(d):                   
    y, m = d.year, d.month
    return datetime.datetime(y+(m==12), (m%12)+1,1) - datetime.timedelta(days=1)

def rolling_forecast_multi(X, Y, start_date, end_date,
                           train_month=1,              
                           target_model="ridge"):
    dt_index = X.index        
    y_cols = [f"y_{c}" for c in Y.columns]            
    full_df  = pd.concat([Y.add_prefix("y_"), X], axis=1)   

    preds_ls, cur = [], start_date
    while True:
        tr_end = month_end(cur + relativedelta(months=train_month))         
        va_end = month_end(tr_end + relativedelta(months=1))      
        te_end = month_end(va_end + relativedelta(months=1))      
        if te_end > end_date: break

        mask_tr = (dt_index>=cur)      & (dt_index<=tr_end)
        mask_va = (dt_index>tr_end)    & (dt_index<=va_end)
        mask_te = (dt_index>va_end)    & (dt_index<=te_end)

        if min(mask_tr.sum(),mask_va.sum(),mask_te.sum())<100:
            cur += relativedelta(months=1); continue

        # ——— 划分
        Xtr = full_df.loc[mask_tr,  X.columns]
        Xva = full_df.loc[mask_va,  X.columns]
        Xte = full_df.loc[mask_te,  X.columns]

        Ytr = full_df.loc[mask_tr,  y_cols]           
        Yva = full_df.loc[mask_va,  y_cols]           
        Yte = full_df.loc[mask_te,  y_cols]  

        # ——— 标准化 (fit on train)
        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xva_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)

        # ——— 训练 + valid 选参
        est = train_and_select_model(
            pd.DataFrame(Xtr_s, index=Xtr.index),
            Ytr,
            pd.DataFrame(Xva_s, index=Xva.index),
            Yva,
            model_name=target_model, n_iter=10
        )

        # ——— 测试集预测
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

    com['log_close'] = np.log(com['close'])
    com['log_ret']   = com.groupby('underlying_symbol')['log_close'].diff()
    pivot = com.pivot(index="datetime", columns="underlying_symbol",
                      values="log_ret").iloc[1:,:]

    df = pivot.reset_index()
    df['date'] = df['datetime'].dt.date
    df = df.groupby('date', group_keys=False).apply(lambda g: g.iloc[1:-1])
    df.drop(columns='date', inplace=True)
    df.set_index('datetime', inplace=True)

    X, Y = build_multi_matrix(df, lag=opt.lag)
    param_spaces = make_param_spaces(X.shape[1])
    start_dt = datetime.datetime(2018,1,1)
    end_dt   = datetime.datetime(2024,12,31)

    preds = rolling_forecast_multi(X, Y,
                                   start_date=start_dt,
                                   end_date  =end_dt,
                                    train_month  = opt.train_month,                 
                                    target_model = opt.model)

    out_name = f"all_preds_{opt.model}_lag{opt.lag}_tm{opt.train_month}.parquet"
    preds.to_parquet(f"/home/JunpingZhu/output/{out_name}", engine="fastparquet")
    print(f"Done! 结果保存为 {out_name}")