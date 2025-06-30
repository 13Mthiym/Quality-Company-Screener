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
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S&P_500_companies'
        headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:100.0) Gecko/20100101 Firefox/100.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        tables = pd.read_html(response.text, flavor='lxml')
        sp500_df = tables[0]

        symbol_col_name = 'Symbol'
        if 'Symbol' not in sp500_df.columns:
            potential_symbol_cols = [col for col in sp500_df.columns if 'symbol' in col.lower() or 'ticker' in col.lower()]
            if not potential_symbol_cols: raise ValueError("Symbol column missing")
            symbol_col_name = potential_symbol_cols[0]

        tickers = sp500_df[symbol_col_name].astype(str).tolist()
        tickers = [ticker.replace('.', '-') for ticker in tickers]
        tickers = [ticker if ticker != 'BF-B' else 'BF.B' for ticker in tickers]
        print(f"Successfully retrieved {len(tickers)} S&P 500 tickers.")
        return tickers
    except Exception as e:
        print(f"Error retrieving S&P 500 tickers: {e}. Using fallback.")
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
