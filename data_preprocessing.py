import pandas as pd

def preprocess_data(input_csv_path='sp500_financial_data.csv',
                    output_csv_path='processed_stock_data.csv'):
    """
    Loads the financial data, preprocesses it, engineers the 'Quality' target variable,
    and saves the processed data.
    """
    try:
        df = pd.read_csv(input_csv_path)
        print(f"Loaded data from {input_csv_path}. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: The file {input_csv_path} was not found. Please run data acquisition first.")
        return None

    # --- 1. Handle Missing Data ---
    # Drop rows where any of the key financial metrics for criteria are missing
    initial_rows = len(df)
    metrics_for_na_check = ['ROE', 'DE_Ratio', 'PE_Ratio']
    df.dropna(subset=metrics_for_na_check, inplace=True)
    rows_after_na_drop = len(df)
    print(f"Dropped {initial_rows - rows_after_na_drop} rows due to missing values in {metrics_for_na_check}.")

    if df.empty:
        print("No data remaining after dropping NA values. Cannot proceed.")
        return None

    # --- 2. Create Target Variable ("Quality") ---
    # Conditions for "High Quality" (Quality = 1)
    # ROE > 15% (0.15)
    # D/E Ratio < 0.5
    # P/E Ratio < 20

    # Ensure PE_Ratio is numeric and handle potential non-numeric placeholders if any (though yfinance usually gives numeric or NaN)
    df['PE_Ratio'] = pd.to_numeric(df['PE_Ratio'], errors='coerce')
    # Re-check NAs for PE_Ratio in case coercion created new NaNs, and drop them
    if df['PE_Ratio'].isnull().any():
        print(f"Found NaNs in PE_Ratio after coercing to numeric. Dropping {df['PE_Ratio'].isnull().sum()} more rows.")
        df.dropna(subset=['PE_Ratio'], inplace=True)
        if df.empty:
            print("No data remaining after dropping PE_Ratio NaNs from coercion. Cannot proceed.")
            return None

    # Apply the conditions
    # Note: ROE is a ratio (e.g., 0.15 for 15%)
    df['Quality'] = 0 # Default to Standard Quality

    quality_conditions = (
        (df['ROE'] > 0.15) &
        (df['DE_Ratio'] < 0.5) &
        (df['PE_Ratio'] < 20) &
        (df['PE_Ratio'] > 0) # Implicitly, P/E should be positive for meaningful interpretation in this context
    )
    df.loc[quality_conditions, 'Quality'] = 1

    print("\nQuality Label Distribution:")
    print(df['Quality'].value_counts(normalize=True) * 100)

    # --- 3. Select Features (X) and Target (y) ---
    # Features for the model
    feature_columns = ['ROE', 'DE_Ratio', 'PE_Ratio']
    # Columns to keep in the output file (including Ticker and Year for context if needed later)
    output_columns = ['Ticker', 'Year', 'ROE', 'DE_Ratio', 'PE_Ratio', 'Quality']

    processed_df = df[output_columns]

    # --- 4. Save Processed Data ---
    try:
        processed_df.to_csv(output_csv_path, index=False)
        print(f"\nProcessed data saved to {output_csv_path}. Shape: {processed_df.shape}")
    except Exception as e:
        print(f"Error saving processed data to {output_csv_path}: {e}")
        return None

    return processed_df

if __name__ == '__main__':
    print("Starting data preprocessing...")
    processed_dataframe = preprocess_data()

    if processed_dataframe is not None:
        print("\n--- Preprocessing Summary ---")
        print("Processed DataFrame head:")
        print(processed_dataframe.head())
        print("\nProcessed DataFrame info:")
        processed_dataframe.info()
        print("\nDescriptive statistics of processed data:")
        print(processed_dataframe[['ROE', 'DE_Ratio', 'PE_Ratio']].describe())
    else:
        print("Data preprocessing failed or resulted in no data.")

    print("\nData preprocessing script finished.")
