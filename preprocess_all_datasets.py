#!/usr/bin/env python3
"""
Convert all ARFF and other formats to CSV for batch processing
"""

import pandas as pd
from scipy.io import arff
import os
from pathlib import Path

def convert_arff_to_csv(arff_path, csv_path):
    """Convert ARFF file to CSV"""
    try:
        data, meta = arff.loadarff(arff_path)
        df = pd.DataFrame(data)
        
        # Decode byte strings to regular strings
        for col in df.columns:
            if df[col].dtype == object:
                try:
                    df[col] = df[col].str.decode('utf-8')
                except:
                    pass
        
        df.to_csv(csv_path, index=False)
        print(f"✅ Converted: {arff_path.name} -> {csv_path.name}")
        return True
    except Exception as e:
        print(f"❌ Failed to convert {arff_path.name}: {e}")
        return False

def convert_adult_data(data_path, csv_path):
    """Convert adult.data to CSV with proper column names"""
    try:
        # Adult dataset column names
        columns = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num',
            'marital-status', 'occupation', 'relationship', 'race', 'sex',
            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'target'
        ]
        
        df = pd.read_csv(data_path, names=columns, sep=',\s*', engine='python', na_values='?')
        df.to_csv(csv_path, index=False)
        print(f"✅ Converted: {data_path.name} -> {csv_path.name}")
        return True
    except Exception as e:
        print(f"❌ Failed to convert {data_path.name}: {e}")
        return False

def main():
    csv_dir = Path("/project/def-arashmoh/shahab33/Msc/CSV")
    
    print("="*60)
    print("Converting all datasets to CSV format")
    print("="*60)
    
    converted_count = 0
    
    # Convert ARFF files
    arff_files = list(csv_dir.glob("*.arff"))
    print(f"\nFound {len(arff_files)} ARFF files")
    
    for arff_file in arff_files:
        csv_file = arff_file.with_suffix('.csv')
        if csv_file.exists():
            print(f"⏭️  Skipping {arff_file.name} (CSV already exists)")
            continue
        
        if convert_arff_to_csv(arff_file, csv_file):
            converted_count += 1
    
    # Convert adult.data
    adult_data = csv_dir / "adult.data"
    if adult_data.exists():
        adult_csv = csv_dir / "adult.csv"
        if not adult_csv.exists():
            if convert_adult_data(adult_data, adult_csv):
                converted_count += 1
        else:
            print(f"⏭️  Skipping adult.data (CSV already exists)")
    
    # List all CSV files
    csv_files = list(csv_dir.glob("*.csv"))
    print(f"\n{'='*60}")
    print(f"Total CSV files available: {len(csv_files)}")
    print("="*60)
    
    for csv_file in sorted(csv_files):
        print(f"  - {csv_file.name}")
    
    print(f"\n✅ Conversion complete! Converted {converted_count} new files.")

if __name__ == "__main__":
    main()
