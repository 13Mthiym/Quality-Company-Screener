# main.py - Orchestrates the entire stock quality screening workflow.

import data_acquisition
import data_preprocessing
import model_training
import model_evaluation
import pandas as pd # For handling data between steps if needed, and for potential full S&P run

def run_full_workflow(tickers_to_fetch=None, data_years=5):
    """
    Runs the complete workflow: data acquisition, preprocessing, training, and evaluation.
    """
    print("--- Starting Stock Quality Screener Workflow ---")

    # --- Step 1: Data Acquisition ---
    print("\n--- Stage 1: Data Acquisition ---")
    raw_data_path = 'sp500_financial_data.csv'

    if tickers_to_fetch is None:
        # Default to a small sample for quicker execution
        # Note: This fallback list should ideally also provide sectors and companies
        # but for quick testing, a simple ticker list can be sufficient here if get_financial_data handles it.
        # For full run, the get_sp500_tickers_and_sectors will provide the full DataFrame.
        sample_tickers_data = {
            'Ticker': ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'XOM', 'NEE', 'MMM', 'DIS', 'LLY', 'NVDA'],
            'Sector': ['Technology'] * 10, # Placeholder sectors for sample if needed
            'Company': ['Apple Inc.', 'Microsoft Corp.', 'Alphabet Inc. (Class A)', 'JPMorgan Chase & Co.', 'Exxon Mobil Corp.', 'NextEra Energy Inc.', '3M Company', 'The Walt Disney Company', 'Eli Lilly and Company', 'NVIDIA Corp.']
        }
        tickers_to_fetch_df = pd.DataFrame(sample_tickers_data)
        print(f"Proceeding with a sample of {len(tickers_to_fetch_df)} tickers...")
    else:
        # If tickers_to_fetch is provided (e.g., from get_sp500_tickers_and_sectors), it's already a DataFrame
        tickers_to_fetch_df = tickers_to_fetch


    # Call the financial data acquisition function, passing the DataFrame
    financial_df, failed_list = data_acquisition.get_financial_data(tickers_to_fetch_df, years=data_years)

    if financial_df is None or financial_df.empty:
        print("Data acquisition failed or returned no data. Exiting workflow.")
        return
        
    print(f"\nData acquired for {financial_df['Ticker'].nunique()} tickers.")
    if failed_list:
        print(f"Could not retrieve data for {len(failed_list)} tickers: {failed_list}")

    try:
        financial_df.to_csv(raw_data_path, index=False)
        print(f"Raw financial data saved to {raw_data_path}")
    except Exception as e:
        print(f"Error saving raw financial data: {e}. Exiting workflow.")
        return

    # --- Step 2: Data Preprocessing ---
    print("\n--- Stage 2: Data Preprocessing ---")
    processed_data_path = 'processed_stock_data.csv'
    processed_df = data_preprocessing.preprocess_data(input_csv_path=raw_data_path,
                                                      output_csv_path=processed_data_path)
    if processed_df is None or processed_df.empty:
        print("Data preprocessing failed or resulted in no data. Exiting workflow.")
        return

    # --- Step 3: Model Training ---
    print("\n--- Stage 3: Model Training ---")
    model_path = 'random_forest_quality_model.joblib'
    trained_model = model_training.train_model(input_csv_path=processed_data_path,
                                               model_output_path=model_path)
    if trained_model is None:
        print("Model training failed. Exiting workflow.")
        return

    # --- Step 4: Model Evaluation ---
    print("\n--- Stage 4: Model Evaluation and Interpretation ---")
    model_evaluation.evaluate_model(model_path=model_path,
                                    x_test_path='X_test.csv',
                                    y_test_path='y_test.csv')

    # --- Stage 5: Consistency Analysis ---
    print("\n--- Stage 5: Consistency Analysis ---")
    if processed_df is not None and not processed_df.empty:
        print(f"Analyzing processed data from {processed_data_path} to find consistent performers...")
        high_quality_df = processed_df[processed_df['Quality'] == 1]

        if not high_quality_df.empty:
            consistency_counts = high_quality_df['Ticker'].value_counts().reset_index()
            consistency_counts.columns = ['Ticker', 'Years as High Quality']
            print("Companies ranked by the number of years they met the 'High Quality' criteria:")
            print(consistency_counts)
        else:
            print("No companies met the 'High Quality' criteria in the dataset.")
    else:
        print("Skipping consistency analysis because processed data is not available.")


    print("\n--- Stock Quality Screener Workflow Completed ---")

if __name__ == '__main__':
    # To run with a small sample of tickers and 5 years of data:
    # run_full_workflow(data_years=5)  #<-- ADD A '#' HERE

    # Example: To attempt a full S&P 500 run (can be time-consuming):
    print("\n--- Attempting Full S&P 500 Run (may take a long time) ---")
    # Corrected function call: data_acquisition.get_sp500_tickers() changed to
    # data_acquisition.get_sp500_tickers_and_sectors()
    all_sp500_data_df = data_acquisition.get_sp500_tickers_and_sectors() # Renamed variable for clarity
    if all_sp500_data_df is not None and not all_sp500_data_df.empty:
       run_full_workflow(tickers_to_fetch=all_sp500_data_df, data_years=10)
    else:
       print("Could not fetch S&P 500 ticker list for full run. Exiting.")