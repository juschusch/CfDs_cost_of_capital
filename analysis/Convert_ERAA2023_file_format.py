import pandas as pd
import numpy as np
import sys

def convert_demand_data(input_file="Demand_Timeseries_TY2030.xlsx", sheet="DE00", output_file="electricity_demand_converted.csv"):
    print(f"Reading {input_file}...")
    try:
        # Read Excel, skipping the first row (Title) to get headers from the second row
        # Row 0 in Excel is index 0 (Title), Row 1 in Excel is index 1 (Header)
        # pd.read_excel header parameter is 0-indexed row number. So header=1 uses the second row.
        df = pd.read_excel(input_file, sheet_name=sheet, header=1)
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        return

    # Identify year columns (integers > 1900)
    year_cols = [c for c in df.columns if isinstance(c, int) and c > 1900]
    print(f"Found years: {year_cols}")
    
    dfs = []
    
    for year in year_cols:
        # Select data for this year
        # We assume columns are: Date, Hour, [Years...]
        sub = df[['Date', 'Hour', year]].copy()
        
        # Drop NaNs (e.g. Feb 29 for non-leap years if present as empty)
        sub = sub.dropna(subset=[year])
        
        # Function to create datetime objects
        def make_datetime(row):
            try:
                # Handle Date format "DD.MM."
                d_str = str(row['Date']).strip()
                # Handle potential float conversion of "dd.mm" if excel confuses it
                
                day = 1
                month = 1
                
                parts = d_str.split('.')
                if len(parts) >= 2:
                    day = int(parts[0])
                    month = int(parts[1])
                else:
                    return pd.NaT

                # Handle Hour (1-24 -> 0-23)
                hour = int(row['Hour']) - 1
                
                return pd.Timestamp(year=year, month=month, day=day, hour=hour)
            except (ValueError, IndexError):
                return pd.NaT

        sub['timestamp'] = sub.apply(make_datetime, axis=1)
        sub = sub.dropna(subset=['timestamp'])
        
        # Set index and rename value column
        sub = sub.set_index('timestamp')
        sub = sub[[year]].rename(columns={year: 'DE'})
        
        dfs.append(sub)
    
    if not dfs:
        print("No valid data found.")
        return

    # Concatenate all years
    print("Concatenating data...")
    final_df = pd.concat(dfs)
    final_df = final_df.sort_index()
    
    # Format for CSV: Index name empty, Column 'DE'
    final_df.index.name = None
    
    print(f"Writing to {output_file}...")
    print(final_df.head())
    final_df.to_csv(output_file)
    print("Done.")

# Run the conversion
if __name__ == "__main__":
    convert_demand_data()
