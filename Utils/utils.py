def format_time(t):
    m_, s = divmod(t, 60)
    h, m = divmod(m_, 60)
    return '{:02.0f}:{:02.0f}:{:02.0f}'.format(h, m, s)


def avg_true_range(df):
    ind = range(0, len(df))
    indexlist = list(ind)
    df.index = indexlist

    for index, row in df.iterrows():
        if index != 0:
            tr1 = row["High"] - row["Low"]
            tr2 = abs(row["High"] - df.iloc[index - 1]["Close"])
            tr3 = abs(row["Low"] - df.iloc[index - 1]["Close"])

            true_range = max(tr1, tr2, tr3)
            df.set_value(index, "True Range", true_range)

    df["Avg TR"] = df["True Range"].rolling(min_periods=14, window=14, center=False).mean()
    return df


def chandelier_exit(df):  # default period is 22
    df_tr = avg_true_range(df)

    rolling_high = df_tr["High"][-22:].max()
    rolling_low = df_tr["Low"][-22:].max()

    df['chandelier_long'] = rolling_high - df_tr.iloc[-1]["Avg TR"] * 3
    df['chandelier_short'] = rolling_low - df_tr.iloc[-1]["Avg TR"] * 3
    return df
