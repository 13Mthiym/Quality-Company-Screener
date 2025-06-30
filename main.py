# main.py - Orchestrates the entire stock quality screening workflow.

import data_acquisition
import data_preprocessing
import model_training
import model_evaluation
import pandas as pd # For handling data between steps if needed, and for potential full S&P run

def run_full_workflow(tickers_to_fetch=None, data_years=5):
    """
    Runs the complete workflow: data acquisition, preprocessing, training, and evaluation.

    Args:
        tickers_to_fetch (list, optional): A specific list of tickers to process.
                                           If None, will try to fetch all S&P 500 tickers.
                                           Defaults to a small sample list for quick runs.
        data_years (int, optional): Number of years of historical data to fetch. Defaults to 5.
    """
    print("--- Starting Stock Quality Screener Workflow ---")

    # --- Step 1 & 2: Data Acquisition ---
    print("\n--- Stage 1: Data Acquisition ---")
    raw_data_path = 'sp500_financial_data.csv'

    if tickers_to_fetch is None:
        # Default to a small sample for quicker execution of the main script
        # For a full S&P 500 run, this list would be much larger.
        tickers_to_fetch = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'XOM', 'NEE', 'MMM', 'DIS', 'LLY', 'NVDA']
        # To run for all S&P 500, one might uncomment the following:
        # tickers_to_fetch = data_acquisition.get_sp500_tickers()
        # if not tickers_to_fetch:
        #     print("Failed to retrieve S&P 500 tickers. Exiting workflow.")
        #     return
        # print(f"Proceeding with {len(tickers_to_fetch)} tickers (Sample or Full S&P 500).")
        print(f"Proceeding with a sample of {len(tickers_to_fetch)} tickers: {tickers_to_fetch[:5]}...")


    financial_df = data_acquisition.get_financial_data(tickers_to_fetch, years=data_years)

    if financial_df is None or financial_df.empty:
        print("Data acquisition failed or returned no data. Exiting workflow.")
        return
    try:
        financial_df.to_csv(raw_data_path, index=False)
        print(f"Raw financial data saved to {raw_data_path}")
    except Exception as e:
        print(f"Error saving raw financial data: {e}. Exiting workflow.")
        return

    # --- Step 3: Data Preprocessing and Feature Engineering ---
    print("\n--- Stage 2: Data Preprocessing ---")
    processed_data_path = 'processed_stock_data.csv'
    processed_df = data_preprocessing.preprocess_data(input_csv_path=raw_data_path,
                                                      output_csv_path=processed_data_path)
    if processed_df is None or processed_df.empty:
        print("Data preprocessing failed or resulted in no data. Exiting workflow.")
        return

    # --- Step 4: Build and Train the Random Forest Model ---
    print("\n--- Stage 3: Model Training ---")
    model_path = 'random_forest_quality_model.joblib'
    # train_model also saves X_test.csv and y_test.csv
    trained_model = model_training.train_model(input_csv_path=processed_data_path,
                                               model_output_path=model_path)
    if trained_model is None:
        print("Model training failed. Exiting workflow.")
        return

    # --- Step 5 & 6: Evaluate the Model and Interpret Results ---
    print("\n--- Stage 4: Model Evaluation and Interpretation ---")
    # evaluate_model loads the model and test data saved by train_model
    model_evaluation.evaluate_model(model_path=model_path,
                                    x_test_path='X_test.csv',
                                    y_test_path='y_test.csv')

    print("\n--- Stock Quality Screener Workflow Completed ---")

if __name__ == '__main__':
    # To run with a small sample of tickers and 5 years of data:
    run_full_workflow(data_years=5)

    # Example: To run with a specific list of tickers:
    # custom_tickers = ['T', 'VZ', 'PFE']
    # run_full_workflow(tickers_to_fetch=custom_tickers, data_years=7)

    # Example: To attempt a full S&P 500 run (can be time-consuming):
    # print("\n--- Attempting Full S&P 500 Run (may take a long time) ---")
    # all_sp500_tickers = data_acquisition.get_sp500_tickers()
    # if all_sp500_tickers:
    #    run_full_workflow(tickers_to_fetch=all_sp500_tickers, data_years=10)
    # else:
    #    print("Could not fetch S&P 500 ticker list for full run.")
