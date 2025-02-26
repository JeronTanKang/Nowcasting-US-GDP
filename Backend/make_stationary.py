def make_stationary(df, max_diff=2): # ensure max amount of differencing applied is 2
        df_stationary = df.copy()
        for col in df_stationary.columns:
            diff_count = 0
            while diff_count < max_diff:
                adf_test = adfuller(df_stationary[col].dropna(), autolag="AIC")
                if adf_test[1] > 0.05:  # non stationary
                    df_stationary[col] = df_stationary[col].diff().dropna()
                    diff_count += 1
                else:
                    break
        return df_stationary  # Do NOT dropna() to keep time alignment
