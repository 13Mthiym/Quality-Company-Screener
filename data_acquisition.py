import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time
import requests

ASSUMED_CURRENT_YEAR = 2024

def find_financial_statement_key(df_index, preferred_keys, default_key=None):
    if df_index is None:
        return default_key if default_key else (preferred_keys[0] if preferred_keys else None)
    for key in preferred_keys:
        if key in df_index:
            return key
    return default_key if default_key else (preferred_keys[0] if preferred_keys else None)

def get_sp500_tickers():
    """
    Retrieves the list of S&P 500 tickers from stockanalysis.com.
    Falls back to a hardcoded list if the fetch fails.
    """
    try:
        # The new, more reliable URL
        url = 'https://stockanalysis.com/list/sp-500-stocks/'

        # Using headers to mimic a browser visit
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

        # Using pandas to read the HTML table directly from the URL
        response = requests.get(url, headers=headers)
        response.raise_for_status() # Check for HTTP errors
        sp500_df_list = pd.read_html(response.text)

        if not sp500_df_list:
            raise ValueError("No tables found on the page.")
        sp500_df = sp500_df_list[0] # Assuming the first table is the correct one

        # The 'Symbol' column contains the tickers
        if 'Symbol' in sp500_df.columns:
            tickers = sp500_df['Symbol'].astype(str).tolist()
            # Clean up tickers for yfinance compatibility
            # General rule: yfinance often uses '-' where sources use '.' (e.g. BRK-B vs BRK.B)
            # However, for specific ones like BRK.B and BF.B, yfinance expects the '.'

            # Start with a general replacement for other cases if any (though S&P500 usually doesn't have many dots)
            # tickers = [ticker.replace('.', '-') for ticker in tickers] # This might be too broad

            # Specific known adjustments for yfinance:
            # BF.B is often listed as BF-B elsewhere
            # BRK.B is often listed as BRK-B elsewhere

            # Correcting based on common yfinance usage:
            # yfinance needs BRK.B (not BRK-B from some sources)
            # yfinance needs BF.B (not BF-B from some sources)
            # Most other S&P500 tickers are simple, no dots or dashes.

            # A better approach for cleaning:
            cleaned_tickers = []
            for ticker in tickers:
                if ticker == 'BRK-B': # If source gives BRK-B
                    cleaned_tickers.append('BRK.B')
                elif ticker == 'BF-B': # If source gives BF-B
                    cleaned_tickers.append('BF.B')
                # Add other specific known transformations if they arise for stockanalysis.com
                else:
                    # For most S&P500 tickers, no change or simple dot removal is needed.
                    # stockanalysis.com seems to provide clean tickers like "GOOGL", "AAPL"
                    # If a ticker like "ABC.N" appeared, yfinance might need "ABC" or "ABC-N".
                    # For now, assume stockanalysis.com provides symbols yfinance mostly understands directly,
                    # apart from the specific cases above.
                    cleaned_tickers.append(ticker.replace('.', '-')) # General case for other potential dots from source

            # Re-apply specific known yfinance preferences after general cleaning
            final_tickers = []
            for ticker in cleaned_tickers:
                if ticker == 'BRK-B': # Ensure it is BRK.B for yfinance
                    final_tickers.append('BRK.B')
                elif ticker == 'BF-B':  # Ensure it is BF.B for yfinance
                    final_tickers.append('BF.B')
                else:
                    final_tickers.append(ticker)


            print(f"Successfully retrieved {len(final_tickers)} S&P 500 tickers from stockanalysis.com.")
            return final_tickers
        else:
            raise ValueError("The 'Symbol' column was not found on the page.")

    except Exception as e:
        print(f"Error retrieving S&P 500 tickers from stockanalysis.com: {e}. Using fallback.")
        # The fallback list remains a good safety net
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'JPM', 'V', 'JNJ', 'XOM', 'MMM', 'NEE']

def get_financial_data(tickers, years=10):
    all_data = []
    end_report_year = ASSUMED_CURRENT_YEAR - 1
    start_report_year = end_report_year - years + 1
    print(f"Targeting data from fiscal year {start_report_year} to {end_report_year}")

    for ticker_symbol in tickers:
        print(f"Fetching data for {ticker_symbol}...")
        current_ticker_pe = None # Initialize P/E for the current ticker
        try:
            ticker = yf.Ticker(ticker_symbol)
            annual_financials = ticker.financials
            annual_balance_sheet = ticker.balance_sheet

            if annual_financials is None or annual_financials.empty or \
               annual_balance_sheet is None or annual_balance_sheet.empty:
                print(f"Skipping {ticker_symbol}: Financials or Balance Sheet is None or empty.")
                time.sleep(0.01)
                continue

            info = None # Initialize info
            try:
                info = ticker.info
                if not info or info.get('quoteType') not in ['EQUITY']:
                    if info and info.get('quoteType') not in ['EQUITY']:
                         print(f"Skipping {ticker_symbol}: Not an EQUITY (Type: {info.get('quoteType', 'N/A')}).")
                         time.sleep(0.01)
                         continue
                # Get current trailingPE from info for this ticker
                current_ticker_pe = info.get('trailingPE')
            except Exception:
                print(f"Warning: Could not get .info for {ticker_symbol}, P/E from info will be None.")


        except Exception as e:
            print(f"Error Ticker/initial data for {ticker_symbol}: {e}. Skip.")
            continue

        try:
            NI_KEYS = ['Net Income']
            EQUITY_KEYS = ['Stockholders Equity', 'Total Stockholder Equity']
            TOTAL_DEBT_KEYS = ['Total Debt']
            ST_DEBT_KEYS = ['Current Debt', 'Short Long Term Debt']
            LT_DEBT_KEYS = ['Long Term Debt']
            # EPS keys are no longer needed for P/E with this simplified approach

            actual_ni_key = find_financial_statement_key(annual_financials.index, NI_KEYS)
            actual_equity_key = find_financial_statement_key(annual_balance_sheet.index, EQUITY_KEYS)
            actual_total_debt_key = find_financial_statement_key(annual_balance_sheet.index, TOTAL_DEBT_KEYS)
            actual_st_debt_key = find_financial_statement_key(annual_balance_sheet.index, ST_DEBT_KEYS)
            actual_lt_debt_key = find_financial_statement_key(annual_balance_sheet.index, LT_DEBT_KEYS)

            common_cols = annual_financials.columns.intersection(annual_balance_sheet.columns)
            target_cols_ts = sorted([ts for ts in common_cols if start_report_year <= ts.year <= end_report_year], reverse=True)

            if not target_cols_ts: continue

            for report_date_ts in target_cols_ts:
                fiscal_year = report_date_ts.year

                ni_series = annual_financials.loc[actual_ni_key] if actual_ni_key and actual_ni_key in annual_financials.index else pd.Series(dtype=float)
                equity_series = annual_balance_sheet.loc[actual_equity_key] if actual_equity_key and actual_equity_key in annual_balance_sheet.index else pd.Series(dtype=float)
                total_debt_series = annual_balance_sheet.loc[actual_total_debt_key] if actual_total_debt_key and actual_total_debt_key in annual_balance_sheet.index else pd.Series(dtype=float)
                st_debt_series = annual_balance_sheet.loc[actual_st_debt_key] if actual_st_debt_key and actual_st_debt_key in annual_balance_sheet.index else pd.Series(dtype=float)
                lt_debt_series = annual_balance_sheet.loc[actual_lt_debt_key] if actual_lt_debt_key and actual_lt_debt_key in annual_balance_sheet.index else pd.Series(dtype=float)

                net_income_val = ni_series.get(report_date_ts)
                total_equity_val = equity_series.get(report_date_ts)

                current_roe = None
                if pd.notna(net_income_val) and pd.notna(total_equity_val) and total_equity_val != 0:
                    current_roe = net_income_val / total_equity_val

                total_debt_direct_val = total_debt_series.get(report_date_ts)
                calculated_total_debt_val = None
                if pd.isna(total_debt_direct_val):
                    short_term_debt_val = st_debt_series.get(report_date_ts)
                    long_term_debt_val = lt_debt_series.get(report_date_ts)
                    st_debt = short_term_debt_val if pd.notna(short_term_debt_val) else 0
                    lt_debt = long_term_debt_val if pd.notna(long_term_debt_val) else 0
                    if pd.notna(short_term_debt_val) or pd.notna(long_term_debt_val):
                        calculated_total_debt_val = st_debt + lt_debt

                final_total_debt_val = total_debt_direct_val if pd.notna(total_debt_direct_val) else calculated_total_debt_val

                current_de_ratio = None
                if pd.notna(final_total_debt_val) and pd.notna(total_equity_val) and total_equity_val != 0:
                    current_de_ratio = final_total_debt_val / total_equity_val

                # P/E ratio is now the current_ticker_pe fetched from .info
                # This value will be the same for all years of this ticker.

                all_data.append({'Ticker': ticker_symbol, 'Year': fiscal_year, 'ROE': current_roe,
                                 'DE_Ratio': current_de_ratio, 'PE_Ratio': current_ticker_pe, # Use current P/E
                                 'Report_Date': report_date_ts.strftime('%Y-%m-%d')})

            time.sleep(0.3)
        except Exception as e:
            print(f"Major error during data processing for {ticker_symbol}: {e.__class__.__name__} - {e}")
            time.sleep(0.5)
            continue

    return pd.DataFrame(all_data)

if __name__ == '__main__':
    print("Starting data acquisition script...")
    # test_tickers_list = get_sp500_tickers()[:20] # Test with a larger slice
    test_tickers_list = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'XOM', 'NEE', 'MMM', 'DIS'] # Diverse test set

    print(f"Fetching for {len(test_tickers_list)} tickers: {test_tickers_list[:5]}...")
    financial_df = get_financial_data(test_tickers_list, years=10)

    if not financial_df.empty:
        print("\n--- Final Data Summary ---")
        print(f"Records: {len(financial_df)}, Tickers: {financial_df['Ticker'].nunique()}")
        yr_series = financial_df['Year'].dropna().astype(int)
        if not yr_series.empty: print(f"Years: {yr_series.min()}-{yr_series.max()}")
        else: print("Years: N/A")

        print("\nData Sample (Head):")
        print(financial_df.head())
        print("\nData Sample (Tail):")
        print(financial_df.tail())

        print("\nMissing Values (%):")
        if len(financial_df) > 0:
            missing_data = (financial_df.isnull().sum()*100/len(financial_df)).round(2)
            print(missing_data[missing_data > 0].sort_values(ascending=False))

        desc_cols = [c for c in ['ROE','DE_Ratio','PE_Ratio'] if c in financial_df.columns]
        if desc_cols and not financial_df[desc_cols].isnull().all().all(): # Check if not all values are NaN
            print("\nKey Metrics Stats:\n", financial_df[desc_cols].describe(percentiles=[.1,.25,.5,.75,.9]))
        else: print("\nNo numerical data for key metrics stats.")

        try:
            output_filename = 'sp500_financial_data.csv'
            financial_df.to_csv(output_filename, index=False)
            print(f"\nData successfully saved to {output_filename}")
        except Exception as e: print(f"\nError saving CSV: {e}")

    else: print("\nNo data fetched.")
    print("Script finished.")
