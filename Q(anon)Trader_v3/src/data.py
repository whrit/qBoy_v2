import yfinance as yf
import pandas as pd
import numpy as np
import talib
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
import os

def fetch_stock_data(symbol, start_date, end_date, output_file=None):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    stock_data['Close'] = stock_data['Adj Close']
    stock_data = stock_data.drop(columns=['Adj Close'])
    stock_data.columns = stock_data.columns.str.lower()
    
    if output_file:
        stock_data.to_csv(output_file)
    
    return stock_data

def add_technical_indicators(df):
    open_data = df['open']
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']

    # Overlap Studies
    df['SMA'] = talib.SMA(close)
    df['EMA'] = talib.EMA(close)
    df['WMA'] = talib.WMA(close)
    df['DEMA'] = talib.DEMA(close)
    df['TEMA'] = talib.TEMA(close)
    df['TRIMA'] = talib.TRIMA(close)
    df['KAMA'] = talib.KAMA(close)
    df['MAMA'], df['FAMA'] = talib.MAMA(close)
    df['T3'] = talib.T3(close)
    df['BBANDS_upper'], df['BBANDS_middle'], df['BBANDS_lower'] = talib.BBANDS(close)
    df['SAREXT'] = talib.SAREXT(high, low)
    df['HT_TRENDLINE'] = talib.HT_TRENDLINE(close)

    # Momentum Indicators
    df['ADX'] = talib.ADX(high, low, close)
    df['ADXR'] = talib.ADXR(high, low, close)
    df['APO'] = talib.APO(close)
    df['AROON_down'], df['AROON_up'] = talib.AROON(high, low)
    df['AROONOSC'] = talib.AROONOSC(high, low)
    df['BOP'] = talib.BOP(open_data, high, low, close)
    df['CCI'] = talib.CCI(high, low, close)
    df['CMO'] = talib.CMO(close)
    df['DX'] = talib.DX(high, low, close)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(close)
    df['MFI'] = talib.MFI(high, low, close, volume)
    df['MINUS_DI'] = talib.MINUS_DI(high, low, close)
    df['MINUS_DM'] = talib.MINUS_DM(high, low)
    df['MOM'] = talib.MOM(close)
    df['PLUS_DI'] = talib.PLUS_DI(high, low, close)
    df['PLUS_DM'] = talib.PLUS_DM(high, low)
    df['PPO'] = talib.PPO(close)
    df['ROC'] = talib.ROC(close)
    df['ROCP'] = talib.ROCP(close)
    df['ROCR'] = talib.ROCR(close)
    df['ROCR100'] = talib.ROCR100(close)
    df['RSI'] = talib.RSI(close)
    df['STOCH_k'], df['STOCH_d'] = talib.STOCH(high, low, close)
    df['STOCHF_k'], df['STOCHF_d'] = talib.STOCHF(high, low, close)
    df['STOCHRSI_k'], df['STOCHRSI_d'] = talib.STOCHRSI(close)
    df['TRIX'] = talib.TRIX(close)
    df['ULTOSC'] = talib.ULTOSC(high, low, close)
    df['WILLR'] = talib.WILLR(high, low, close)

    # Volume Indicators
    df['AD'] = talib.AD(high, low, close, volume)
    df['ADOSC'] = talib.ADOSC(high, low, close, volume)
    df['OBV'] = talib.OBV(close, volume)

    # Volatility Indicators
    df['ATR'] = talib.ATR(high, low, close)
    df['NATR'] = talib.NATR(high, low, close)
    df['TRANGE'] = talib.TRANGE(high, low, close)

    # Price Transform
    df['AVGPRICE'] = talib.AVGPRICE(open_data, high, low, close)
    df['MEDPRICE'] = talib.MEDPRICE(high, low)
    df['TYPPRICE'] = talib.TYPPRICE(high, low, close)
    df['WCLPRICE'] = talib.WCLPRICE(high, low, close)

    # Cycle Indicators
    df['HT_DCPERIOD'] = talib.HT_DCPERIOD(close)
    df['HT_DCPHASE'] = talib.HT_DCPHASE(close)
    df['HT_PHASOR_inphase'], df['HT_PHASOR_quadrature'] = talib.HT_PHASOR(close)
    df['HT_SINE_sine'], df['HT_SINE_leadsine'] = talib.HT_SINE(close)
    df['HT_TRENDMODE'] = talib.HT_TRENDMODE(close)

    return df

def select_optimal_indicators(df, target_col='close', n_select=15):
    df_indicators = add_technical_indicators(df)
    
    # Prepare the feature matrix and target variable
    X = df_indicators.drop(columns=['open', 'high', 'low', 'close', 'volume'])
    y = df_indicators[target_col].pct_change()
    
    # Remove the first row (which will have NaN due to pct_change)
    X = X.iloc[1:]
    y = y.iloc[1:]
    
    # Remove any remaining rows with NaN values in either X or y
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    
    # Ensure X and y have the same number of samples
    assert len(X) == len(y), "Mismatch in number of samples between features and target"
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Calculate mutual information scores
    mi_scores = mutual_info_regression(X_scaled, y)
    
    # Create a dataframe of features and their MI scores
    mi_df = pd.DataFrame({'feature': X.columns, 'mi_score': mi_scores})
    
    # Sort by MI score and select top n_select features
    top_features = mi_df.nlargest(n_select, 'mi_score')['feature'].tolist()
    
    return top_features

def apply_optimal_indicators(df, optimal_features):
    df_indicators = add_technical_indicators(df)
    return df_indicators[['open', 'high', 'low', 'close', 'volume'] + optimal_features]

def preprocess_data(data, n_select=15):
    # Add technical indicators
    data_with_indicators = add_technical_indicators(data)
    
    # Select optimal indicators
    optimal_features = select_optimal_indicators(data_with_indicators, n_select=n_select)
    
    # Apply optimal indicators
    data_optimal = apply_optimal_indicators(data_with_indicators, optimal_features)
    
    # Prepare for scaling
    X = data_optimal.drop(['open', 'high', 'low', 'close', 'volume'], axis=1)
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create a new DataFrame with scaled features and original OHLCV data
    preprocessed_data = pd.DataFrame(X_scaled, columns=optimal_features, index=data.index)
    preprocessed_data[['open', 'high', 'low', 'close', 'volume']] = data[['open', 'high', 'low', 'close', 'volume']]
    
    return preprocessed_data, scaler, optimal_features

def debug_dataframe(df, name="DataFrame", save_csv=False, output_file=None):
    print(f"\nDebugging {name}:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"NaN values:\n{df.isna().sum()}")
    print(f"First few rows:\n{df.head()}\n")

    if save_csv:
        if output_file is None:
            output_file = f"{name.lower().replace(' ', '_')}_debug.csv"
        
        # Check if the output_file contains a directory path
        dir_name = os.path.dirname(output_file)
        if dir_name:  # Only try to create the directory if dir_name is not empty
            os.makedirs(dir_name, exist_ok=True)
        
        df.to_csv(output_file)
        print(f"Saved DataFrame to {output_file}")

def prepare_data(symbol, start_date, end_date, n_select=15, debug=False):
    # Fetch stock data
    stock_data = fetch_stock_data(symbol, start_date, end_date)
    
    if debug:
        debug_dataframe(stock_data, "Raw Stock Data", save_csv=True, output_file="raw_stock_data.csv")
    
    # Preprocess data
    preprocessed_data, scaler, optimal_features = preprocess_data(stock_data, n_select)
    
    if debug:
        debug_dataframe(preprocessed_data, "Preprocessed Data", save_csv=True, output_file="preprocessed_data.csv")
    
    return preprocessed_data, scaler, optimal_features, stock_data