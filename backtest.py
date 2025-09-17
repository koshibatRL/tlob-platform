# (250715) Bybitの取引を精密に再現するバックテストコード
# 主な引数: 最良価格列b1,a1，そのうちの取引を実行したい期間，freq間隔で実行されるorder関数
#   💡最良価格の代わりにmid_priceやOHLCのclose等で代用しても結果はほぼ同じ．損益は手数料の比率が大きいので．
#   ⚠️order関数は state = (時刻t, (現在資産usdt,現在ポジ数量Q,現在ポジ平均約定価格P,現在の指値リストlimits)) を受け取り，("Limit"|"Market",数量int,価格float) を返すように設計しなければならない．
#   ⚠️計算誤差対策のため，数量は本当の数量をUNIT倍した整数値で指定する
# 返り値: その戦略を実行したときの資産変動
# TODO: 指値対応
# レバレッジは初期資産を増やせば良いので未実装

import pandas as pd

UNIT = 1000 # 計算誤差対策のため，数量は本当の数量をUNIT倍した整数値で考える

def backtest(b1, a1, order, bgn="2025-06-01", end="2025-06-02", freq="1min", fee_market=0.00055, fee_limit=0.0002, init_usdt=1000) -> pd.DataFrame:
    ts = pd.date_range(bgn, end, freq=freq)
    state = [ts[0], None, None, "", 0, None, init_usdt, 0, None, []]    # [t, b1[t], a1[t], 指値/成行, 注文数量q, 指値価格p, 注文後usdt, 注文後ポジ数量Q(実際のUNIT倍), 注文後ポジ平均約定価格P, 注文後指値リスト]
    history = [state]
    for t in ts[1:]:
        state = state.copy()
        state[0] = t
        b1t, a1t = state[1:3] = float(b1[:t].iloc[-1]), float(a1[:t].iloc[-1])
        order_type, q, p = order(t, state[6:])  # 🍴注文
        if order_type == "Limit":
            if (q > 0 and p > b1t) or (q < 0 and p < a1t):  # 現在の最良指値より不利な価格を指定した場合，強制的に成行で取引される
                order_type = "Market"
        if order_type == "Market":
            p = a1t if q > 0 else b1t                       # 成行注文価格 (買いならa1, 売りならb1)
            usdt_old, Q_old, P_old, limits_old = state[-4:] # 更新前の状態値

            # 1️⃣取引後の未決済建玉 Q_new の計算
            Q_new = Q_old + q       # 更新後ポジ数量: 注文分を加算
            # 2️⃣取引後の平均約定価格 P_new，決済数量 q_exit の計算
            if Q_new == 0:  # 更新後ポジションが0の場合
                q_exit = -Q_old                 # 決済数量: 既存ポジの符号反転
                P_new = None                    # 平均約定価格: 未定義
            else:           # 更新後ポジションがある場合
                if q * Q_old < 0:   # 注文が旧ポジと反対方向 ⇔ 決済発生 の場合
                    if abs(Q_old) > abs(q): # 部分決済なら
                        q_exit = q              # 決済数量: 注文分
                        P_new = P_old           # 平均約定価格: そのまま
                    else:                   # 全決済 or ドテンなら
                        q_exit = -Q_old         # 決済数量: 既存ポジの符号反転
                        P_new = p               # 平均約定価格: 注文価格
                else:               # 注文が既存ポジと同方向か新規ポジか注文0なら
                    q_exit = 0                                                  # 決済数量: 0
                    P_new = ((Q_old * P_old if Q_old else 0) + q * p) / Q_new   # 平均約定価格: P[t]=(Q[t-1]*P[t-1]+q[t]*p[t])/Q[t]
            # 3️⃣資産の更新: この取引による実現損益 ＝ 決済差益 - 手数料
            PnL = ((q_exit * (P_old - p) if q_exit else 0) - fee_market * abs(q) * p) / UNIT
            usdt_new = usdt_old + PnL
            # 取引後の状態
            state = [t, b1t, a1t, order_type, q, p, usdt_new, Q_new, P_new, limits_old]
        history.append(state)
    df_bt = pd.DataFrame([x[1:-1] for x in history], index=ts, columns=["b1","a1","OrderType","q","p","USDT","Q","P"])
    return df_bt


# ⚗️使用例
if __name__ == "__main__":
    import tardis
    import numpy as np

    # サンプルボット．ランダムに売買注文し，十分利益が出たらドテン，一定期間取引がなかったら決済する．最終取引時刻を監視するためクラス化する
    class Bot:
        def __init__(self):
            self.dt_from_lastorder = 0

        def order(self, t, state: list) -> tuple[str, int, float]:
            price = df_ohlcv.loc[t, "open"]
            usdt, Q, P, limits = state
            order_type = "Market"
            if Q == 0:  # ポジがなければランダムに注文
                qty_x1000 = np.random.randint(-1, 2)
            elif (Q > 0 and price > 1.01 * P) or (Q < 0 and price < 0.99 * P):  # 利益が出たらドテン
                qty_x1000 = -2 * Q
            elif self.dt_from_lastorder > 1440:  # 一定期間取引がなかったら決済
                qty_x1000 = -1 * Q
            else:       # 静観
                qty_x1000 = 0
            self.dt_from_lastorder = 0 if qty_x1000 else self.dt_from_lastorder + 1
            return order_type, qty_x1000, price

    df_ohlcv = tardis.load_ohlcv("~/data/tardis/ohlcv_1min_200528-250622.csv.gz")   # 取引価格に1分足OHLCを使う．LOBのb1,a1等も可
    df_bt = backtest(df_ohlcv["open"], df_ohlcv["open"], Bot().order, "2025-05-01", "2025-06-01")
    print(df_bt)

    # 可視化
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(16, 6))
    df_bt[["USDT"]].plot(ax=axes[0])
    df_bt[["b1"]].plot(ax=axes[1])
    df_bt.loc[df_bt["q"]>0,"a1"].plot(ax=axes[1], ls="", marker="^", ms=5, color="g") if any(df_bt["q"]>0) else None
    df_bt.loc[df_bt["q"]<0,"b1"].plot(ax=axes[1], ls="", marker="v", ms=5, color="r") if any(df_bt["q"]<0) else None
    df_bt[["q", "Q"]].plot(ax=axes[2])
    plt.tight_layout()
    plt.savefig("backtest.png")
