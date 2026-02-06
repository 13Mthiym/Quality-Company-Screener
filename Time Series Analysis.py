import pandas as pd
import numpy as np

def perform_sector_and_company_analysis(input_csv_path='processed_stock_data.csv'):
    """
    Performs two types of analysis:
    1. Sector-level average/median/standard deviation for ROE, Debt/Equity, P/E.
    2. Identifies top 3 companies per sector for highest ROE, lowest Debt/Equity, and highest P/E.
    """
    print("--- Starting Consolidated Sector and Company Analysis ---")

    try:
        df = pd.read_csv(input_csv_path)
        print(f"Loaded data from {input_csv_path}. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: The file {input_csv_path} was not found. Please ensure it exists and data preprocessing has been run.")
        return

    # Ensure financial metrics are numeric
    metrics = ['ROE', 'DE_Ratio', 'PE_Ratio']
    for col in metrics:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows where essential columns for analysis are missing
    initial_rows = len(df)
    df.dropna(subset=['Sector', 'Company'] + metrics, inplace=True)
    if len(df) < initial_rows:
        print(f"Dropped {initial_rows - len(df)} rows due to missing values in essential columns for analysis.")

    if df.empty:
        print("No data remaining after cleaning for analysis. Cannot proceed.")
        return

    # --- PART 1: Sector-Level Averages/Medians/Standard Deviations ---
    print("\n--- Part 1: Sector-Level Financial Characteristics ---")

    sector_summary = df.groupby('Sector')[metrics].agg(['mean', 'median', 'std'])
    print(sector_summary)

    print("\n--- Part 1: Insights ---")

    # Insight 1: Which sectors have, on average, higher profitability (ROE)?
    highest_roe_sectors = sector_summary['ROE']['mean'].sort_values(ascending=False).head(3)
    print("\n1. Sectors with highest average profitability (ROE):")
    print(highest_roe_sectors)

    # Insight 2: Which sectors are more leveraged (higher Debt/Equity)?
    highest_de_sectors = sector_summary['DE_Ratio']['mean'].sort_values(ascending=False).head(3)
    print("\n2. Sectors with highest average leverage (Debt/Equity Ratio):")
    print(highest_de_sectors)

    # Insight 3: Which sectors are valued more highly by the market (higher P/E)?
    pe_mean_filtered = sector_summary['PE_Ratio']['mean'][sector_summary['PE_Ratio']['mean'] > 0]
    highest_pe_sectors = pe_mean_filtered.sort_values(ascending=False).head(3)
    print("\n3. Sectors valued more highly by the market (highest average P/E Ratio):")
    print(highest_pe_sectors)

    # Insight 4: How much variation is there within each sector for these metrics (using standard deviation)?
    print("\n4. Variation within sectors (Standard Deviation):")
    for metric in metrics:
        most_variable_sectors = sector_summary[metric]['std'].sort_values(ascending=False).head(3)
        least_variable_sectors = sector_summary[metric]['std'].sort_values(ascending=True).head(3)
        print(f"\n   - {metric}:")
        print(f"     Most Variable: \n{most_variable_sectors}")
        print(f"     Least Variable: \n{least_variable_sectors}")


    # --- PART 2: Top/Bottom Companies per Sector ---
    print("\n\n--- Part 2: Top Performers and Laggards by Sector ---")

    # Calculate average metrics for each company within each sector
    company_sector_avg = df.groupby(['Sector', 'Company'])[metrics].mean().reset_index()

    for sector in company_sector_avg['Sector'].unique():
        print(f"\nSector: {sector}")
        sector_df = company_sector_avg[company_sector_avg['Sector'] == sector].copy()

        # 1. Top 3 companies per sector with the highest average ROE
        top_roe = sector_df[sector_df['ROE'].notna()].sort_values(by='ROE', ascending=False).head(3)
        if not top_roe.empty:
            print(f"  Highest Average ROE:")
            for idx, row in top_roe.iterrows():
                print(f"    - {row['Company']} (ROE: {row['ROE']:.4f})")
        else:
            print(f"  No data for Highest Average ROE in this sector.")

        # 2. Top 3 companies per sector with the lowest average Debt/Equity
        lowest_de = sector_df[sector_df['DE_Ratio'].notna()].sort_values(by='DE_Ratio', ascending=True).head(3)
        if not lowest_de.empty:
            print(f"  Lowest Average Debt/Equity:")
            for idx, row in lowest_de.iterrows():
                print(f"    - {row['Company']} (D/E: {row['DE_Ratio']:.4f})")
        else:
            print(f"  No data for Lowest Average Debt/Equity in this sector.")

        # 3. Top 3 companies per sector with the highest average P/E
        highest_pe = sector_df[(sector_df['PE_Ratio'] > 0) & (sector_df['PE_Ratio'].notna())].sort_values(by='PE_Ratio', ascending=False).head(3)
        if not highest_pe.empty:
            print(f"  Highest Average P/E:")
            for idx, row in highest_pe.iterrows():
                print(f"    - {row['Company']} (P/E: {row['PE_Ratio']:.2f})")
        else:
            print(f"  No data for Highest Average P/E in this sector.")

# --- Execute the analysis ---
if __name__ == '__main__':
    perform_sector_and_company_analysis()
    print("\nConsolidated sector and company analysis finished.")