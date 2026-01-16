"""Sort CSV files in results directory by specified column(s)."""
from __future__ import annotations
import argparse
import csv
import os
from pathlib import Path
from typing import List


def sort_csv(input_path: str, output_path: str, sort_by: List[str], reverse: bool = False):
    """Sort a CSV file by specified column(s)."""
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            print(f"Warning: {input_path} has no header, skipping")
            return
        
        # Validate sort columns
        for col in sort_by:
            if col not in fieldnames:
                print(f"Warning: column '{col}' not found in {input_path}, available: {fieldnames}")
                return
        
        rows = list(reader)
    
    # Sort rows by specified columns
    def sort_key(row):
        key = []
        for col in sort_by:
            val = row[col]
            # Try to convert to number for numeric sorting
            try:
                key.append(float(val))
            except (ValueError, TypeError):
                key.append(val)
        return key
    
    rows.sort(key=sort_key, reverse=reverse)
    
    # Write sorted data
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Sorted {input_path} → {output_path} (by {', '.join(sort_by)})")


def main():
    p = argparse.ArgumentParser(description="Sort CSV files in results directory")
    p.add_argument("--dir", default="results", help="directory containing CSV files")
    p.add_argument("--sort_by", nargs='+', default=["level_id"], 
                   help="column(s) to sort by (space-separated)")
    p.add_argument("--reverse", action="store_true", help="sort in descending order")
    p.add_argument("--inplace", action="store_true", 
                   help="overwrite original files (default: create .sorted.csv)")
    p.add_argument("--pattern", default="*.csv", help="glob pattern for CSV files")
    args = p.parse_args()
    
    results_dir = Path(args.dir)
    if not results_dir.exists():
        print(f"Error: directory {results_dir} does not exist")
        return
    
    csv_files = list(results_dir.glob(args.pattern))
    if not csv_files:
        print(f"No CSV files found in {results_dir}")
        return
    
    for csv_file in sorted(csv_files):
        if csv_file.name.endswith('.sorted.csv') and not args.inplace:
            # Skip already sorted files to avoid double-processing
            continue
        
        if args.inplace:
            output_path = csv_file
        else:
            # Create .sorted.csv version
            output_path = csv_file.with_suffix('.sorted.csv')
        
        sort_csv(str(csv_file), str(output_path), args.sort_by, args.reverse)
    
    print(f"\nProcessed {len(csv_files)} file(s)")


if __name__ == "__main__":
    main()

