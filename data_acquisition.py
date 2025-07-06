import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time
import requests 

# It's good practice to set a constant for the assumed current year for reproducibility.
ASSUMED_CURRENT_YEAR = datetime.now().year

def find_financial_statement_key(df_index, preferred_keys):
    """
    Finds the first available key from a list of preferred keys in a DataFrame's index.
    """
    if df_index is None:
        return None
    for key in preferred_keys:
        if key in df_index:
            return key
    return None

def get_sp500_tickers():
    """
    Retrieves the list of S&P 500 tickers from stockanalysis.com.
    Handles potential scraping errors and falls back to a hardcoded list.
    """
    url = 'https://stockanalysis.com/list/sp-500-stocks/'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # This will raise an HTTPError for bad responses (4xx or 5xx)
        
        # pd.read_html returns a list of all tables found on the page
        tables = pd.read_html(response.text)
        if not tables:
            raise ValueError("No tables found on the page.")
            
        sp500_df = tables[0] # The first table is usually the one we want

        if 'Symbol' not in sp500_df.columns:
            raise ValueError("Column 'Symbol' not found in the table.")

        # The tickers from this source are generally clean. 
        # yfinance can handle tickers like 'BRK.B' and 'BF.B' directly.
        # No complex cleaning is needed.
        tickers = sp500_df['Symbol'].astype(str).tolist()
        
        print(f"Successfully retrieved {len(tickers)} S&P 500 tickers from stockanalysis.com.")
        return tickers

    except Exception as e:
        print(f"Error retrieving S&P 500 tickers: {e}. Using a fallback list.")
        # Fallback list in case the website structure changes or is unavailable
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'JPM', 'V', 'JNJ', 'XOM', 'MMM', 'NEE']

def get_financial_data(tickers, years=10):
    """
    Fetches key financial metrics (ROE, D/E, P/E) for a list of tickers.
    """
    all_data = []
    skipped_tickers = [] # Keep track of tickers we couldn't get data for
    
    end_report_year = ASSUMED_CURRENT_YEAR - 1 
    start_report_year = end_report_year - years + 1
    print(f"Targeting financial data from fiscal year {start_report_year} to {end_report_year}")

    for i, ticker_symbol in enumerate(tickers):
        print(f"Processing {i+1}/{len(tickers)}: {ticker_symbol}...")
        try:
            ticker = yf.Ticker(ticker_symbol)
            
            # Fetch data once
            info = ticker.info
            annual_financials = ticker.financials
            annual_balance_sheet = ticker.balance_sheet

            # --- Pre-computation Checks ---
            # 1. Check if the ticker is a valid equity
            if not info or info.get('quoteType') != 'EQUITY':
                print(f"  -> Skipping {ticker_symbol}: Not an equity (Type: {info.get('quoteType', 'N/A')}).")
                skipped_tickers.append(ticker_symbol)
                continue
            
            # 2. Check for empty financial statements
            if annual_financials.empty or annual_balance_sheet.empty:
                print(f"  -> Skipping {ticker_symbol}: Financials or Balance Sheet data is empty.")
                skipped_tickers.append(ticker_symbol)
                continue

            # --- Key Identification ---
            # Find the correct row names for the financial metrics we need.
            ni_key = find_financial_statement_key(annual_financials.index, ['Net Income'])
            equity_key = find_financial_statement_key(annual_balance_sheet.index, ['Stockholders Equity', 'Total Stockholder Equity'])
            total_debt_key = find_financial_statement_key(annual_balance_sheet.index, ['Total Debt'])
            st_debt_key = find_financial_statement_key(annual_balance_sheet.index, ['Current Debt', 'Short Long Term Debt'])
            lt_debt_key = find_financial_statement_key(annual_balance_sheet.index, ['Long Term Debt'])

            # Get the current P/E ratio once per ticker
            current_pe_ratio = info.get('trailingPE')

            # Find common years between income statement and balance sheet
            common_cols = annual_financials.columns.intersection(annual_balance_sheet.columns)
            target_cols = sorted([ts for ts in common_cols if start_report_year <= ts.year <= end_report_year], reverse=True)

            if not target_cols:
                print(f"  -> Skipping {ticker_symbol}: No financial data available in the target date range.")
                skipped_tickers.append(ticker_symbol)
                continue

            for report_date in target_cols:
                fiscal_year = report_date.year
                
                # --- Metric Calculation ---
                net_income = annual_financials.loc[ni_key, report_date] if ni_key else None
                total_equity = annual_balance_sheet.loc[equity_key, report_date] if equity_key else None
                
                # ROE
                roe = (net_income / total_equity) if pd.notna(net_income) and pd.notna(total_equity) and total_equity != 0 else None
                
                # D/E Ratio
                total_debt = annual_balance_sheet.loc[total_debt_key, report_date] if total_debt_key and pd.notna(annual_balance_sheet.loc[total_debt_key, report_date]) else None
                # If 'Total Debt' is not available, try to calculate it from short-term and long-term debt
                if total_debt is None:
                    st_debt = annual_balance_sheet.loc[st_debt_key, report_date] if st_debt_key else 0
                    lt_debt = annual_balance_sheet.loc[lt_debt_key, report_date] if lt_debt_key else 0
                    if pd.notna(st_debt) or pd.notna(lt_debt):
                        total_debt = (st_debt if pd.notna(st_debt) else 0) + (lt_debt if pd.notna(lt_debt) else 0)

                de_ratio = (total_debt / total_equity) if pd.notna(total_debt) and pd.notna(total_equity) and total_equity != 0 else None
                
                all_data.append({
                    'Ticker': ticker_symbol, 
                    'Year': fiscal_year, 
                    'ROE': roe, 
                    'DE_Ratio': de_ratio, 
                    'PE_Ratio': current_pe_ratio, # Use the single current P/E for all years
                    'Report_Date': report_date.strftime('%Y-%m-%d')
                })
            
            # A short delay to be respectful to the yfinance API
            time.sleep(0.2)

        except Exception as e:
            print(f"  -> ERROR for {ticker_symbol}: {e.__class__.__name__} - {e}. Skipping.")
            skipped_tickers.append(ticker_symbol)
            continue
            
    return pd.DataFrame(all_data), skipped_tickers

if __name__ == '__main__':
    print("--- Starting S&P 500 Financial Data Acquisition Script ---")
    
    # --- Step 1: Get Tickers ---
    all_tickers = get_sp500_tickers()
    
    # For testing, you can uncomment the line below to run on a small sample
    # all_tickers = all_tickers[:15] 
    
    # --- Step 2: Fetch Financial Data ---
    print(f"\nFetching financial data for {len(all_tickers)} tickers...")
    financial_df, failed_list = get_financial_data(all_tickers, years=10) 
    
    # --- Step 3: Display Summary ---
    if not financial_df.empty:
        print("\n--- Final Data Summary ---")
        print(f"Successfully processed and gathered data for {financial_df['Ticker'].nunique()} tickers.")
        print(f"Total records created: {len(financial_df)}")
        
        print("\n--- Data Sample ---")
        print(financial_df.head())
        
        print("\n--- Tickers Skipped or Failed ---")
        if failed_list:
            print(f"Could not retrieve data for {len(failed_list)} tickers: {failed_list}")
        else:
            print("All tickers processed successfully!")

        # --- Step 4: Save to CSV ---
        try:
            output_filename = 'sp500_financial_data.csv'
            financial_df.to_csv(output_filename, index=False)
            print(f"\nData successfully saved to {output_filename}")
        except Exception as e:
            print(f"\nError saving data to CSV: {e}")

    else:
        print("\n--- No data was fetched. ---")
        
    print("Script finished.")