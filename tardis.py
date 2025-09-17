# (250811) Tardisデータ関連ツール

import pandas as pd


def download_data(
    bgn="2025-01-01",
    end="2025-02-01",
    symbols=["BTCUSDT"],
    data_types=["incremental_book_L2", "book_snapshot_25", "trades", "quotes"],
    download_dir="./data/tardis/daily"
) -> None:
    import nest_asyncio
    from tardis_dev import datasets
    nest_asyncio.apply()
    api_key = "TD.NKvpz3yjO4G7YVD9.LFTR03L2LuWuwK7.mNNxWr5wYNc5ONf.4YM7qlckprXbone.GIRtM6cuqrlgBCw.ayy-"
    datasets.download("bybit", data_types, symbols, bgn, end, "csv", api_key, download_dir)



def load_snapshots(bgn="2025-01-01", end="2025-01-31", gz_dir="./data/tardis/daily", freq="", simple_colname=True) -> pd.DataFrame:
    # freq引数でダウンサンプリング可能 (freq="" なら生tick，"1min" なら1分足)
    df_ss = pd.DataFrame()
    for day in pd.date_range(bgn, end, freq="1d").date:         # 約5秒/日
        print(f"loading {day} snapshot...", end="\r")
        df_day = pd.read_csv(f"{gz_dir}/bybit_book_snapshot_25_{day}_BTCUSDT.csv.gz")       # ⚠️最初はlocal_timestampでソートされていることに注意
        df_day.index = pd.to_datetime(df_day["local_timestamp"], unit="us")     # Indexを時刻に変更
        if freq:
            df_day = df_day.resample(freq).last()               # 1️⃣終値でダウンサンプリング (firstだと厳密には未来の値になるので少し工夫)
            df_day.index += df_day.index[1] - df_day.index[0]   # 2️⃣始値に修正．これでたとえば1分足で12時34分の値は厳密にそれ以前の最終値になる
        df_ss = pd.concat([df_ss, df_day.iloc[:, 4:]])          # 1日分追加，不要な列を削除
    print()
    # if freq:    # 時刻重複を削除 ⚠️tickの場合，1時刻に複数のsnapshot更新が観測される場合があり，情報が減るため行わない
    #     df_ss = df_ss[~df_ss.index.duplicated(keep="first")]
    if simple_colname:  # snapshotの列名簡略化
        cols = (
            {f"bids[{i}].price":  f"b{i+1}"  for i in range(25)} |
            {f"bids[{i}].amount": f"bq{i+1}" for i in range(25)} |
            {f"asks[{i}].price":  f"a{i+1}"  for i in range(25)} |
            {f"asks[{i}].amount": f"aq{i+1}" for i in range(25)}
        )
        df_ss = df_ss[cols.keys()].rename(columns=cols)
    return df_ss

def load_incrementals(bgn="2025-01-01", end="2025-01-31", gz_dir="./data/tardis/daily") -> pd.DataFrame:
    df_incr = pd.DataFrame()
    for day in pd.date_range(bgn, end, freq="1d").date:         # 約10秒/日
        print(f"loading {day} incremental...", end="\r")
        df_day = pd.read_csv(f"{gz_dir}/bybit_incremental_book_L2_{day}_BTCUSDT.csv.gz")    # ⚠️最初はlocal_timestampでソートされていることに注意
        df_day = df_day.set_index(pd.to_datetime(df_day["local_timestamp"], unit="us"))     # Indexを時刻に変更
        df_incr = pd.concat([df_incr, df_day.iloc[:, 4:]])      # 1日分追加，不要な列を削除
    print()
    return df_incr

def load_trades(bgn="2025-01-01", end="2025-01-31", gz_dir="./data/tardis/daily") -> pd.DataFrame:
    df_trades = pd.DataFrame()
    for day in pd.date_range(bgn, end, freq="1d").date:
        print(f"loading {day} trade...", end="\r")
        df_day = pd.read_csv(f"{gz_dir}/bybit_trades_{day}_BTCUSDT.csv.gz")                 # ⚠️最初はlocal_timestampでソートされていることに注意
        df_day = df_day.set_index(pd.to_datetime(df_day["local_timestamp"], unit="us"))     # Indexを時刻に変更
        df_trades = pd.concat([df_trades, df_day.iloc[:, 5:]])  # 1日分追加，不要な列を削除
    print()
    return df_trades

def load_ohlcv(gz_path="~/data/tardis/ohlcv_1min_200528-250622.csv.gz", fillna=True) -> pd.DataFrame:
    df_ohlcv = pd.read_csv(gz_path, index_col=0, parse_dates=True)
    if fillna:  # 欠損埋め (序盤のNaNは残る)
        idx = df_ohlcv["close"].isna()                  # 欠損時刻
        df_ohlcv["close"] = df_ohlcv["close"].ffill()   # 終値を最終観測価格で埋める (つまりffill)
        df_ohlcv.loc[idx, ["open", "high", "low"]] = df_ohlcv.loc[idx, "close"] # open, high, lowも最終観測価格(=close)で埋める
        df_ohlcv[["volume", "volume_usdt"]] = df_ohlcv[["volume", "volume_usdt"]].fillna(0) # 2021年以前のtradeデータに，priceはあるのにvolumeがnanの場合があるので0埋め
    df_ohlcv = df_ohlcv["2020-05-29":]  # もともと 2020/05/28 は16:14以前がないので，2020/05/29 から始める
    return df_ohlcv



if __name__ == "__main__":
    download_data("2025-06-01", "2025-07-01", download_dir="./data/tardis/daily")  # DL用

    bgn = "2025-06-01"
    end = "2025-06-02"
    df_ss = load_snapshots(bgn, end)
    df_incr = load_incrementals(bgn, end)
    df_trades = load_trades(bgn, end)
    df_ohlcv = load_ohlcv()
    print(df_ss.shape)      # snapshotは 55回/秒 = 3327回/分
    print(df_incr.shape)    # 指値注文は 417回/秒 = 25053回/分
    print(df_trades.shape)  # 成行注文は 26回/秒 = 1585回/分
    print(df_ohlcv.shape)   # OHLCVは1分足
