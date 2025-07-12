import pandas as pd, numpy as np, datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str,
                    default="/home/JunpingZhu/input/com_1m_bar.parquet")
opt = parser.parse_args()


path_pred = opt.data_path
pred = pd.read_parquet(path_pred)

com  = pd.read_parquet('/home/JunpingZhu/input/com_1m_bar.parquet', engine="fastparquet").reset_index()

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



y_true = df.loc[pred.index]      


def r2_os_zero(y, yhat):
    sse_model = np.sum((y - yhat)**2)
    sse_bench = np.sum(y**2)               
    return 1 - sse_model / sse_bench

r2_dict = {sym: r2_os_zero(y_true[sym].values,
                           pred[sym].values)
           for sym in pred.columns}

r2_df = pd.Series(r2_dict, name="R2_OS")
print("各品种 R²_OS:")
print(r2_df.sort_values(ascending=False).round(4))

print("\n平均 R²_OS:", r2_df.mean().round(4))