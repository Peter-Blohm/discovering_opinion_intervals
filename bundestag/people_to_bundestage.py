import csv
import re
import pandas as pd
from datetime import datetime
from collections import defaultdict

bundestag_periods = {
    17: ("2009-10-27", "2013-10-21"),
    18: ("2013-10-22", "2017-10-22"),
    19: ("2017-10-23", "2021-10-25"),
    20: ("2021-10-26", "2025-03-24"),
    21: ("2025-03-25", "2100-01-01")
}

for period, (start, end) in bundestag_periods.items():
    bundestag_periods[period] = (
        datetime.strptime(start, "%Y-%m-%d"),
        datetime.strptime(end, "%Y-%m-%d")
    )

def validate_filenames(input_file):
    """Check if all filenames conform to the pattern YYYYMMDD_*"""
    pattern = re.compile(r'^(\d{8}).*$')
    
    non_conforming = set()
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row['filename']
            if not pattern.match(filename):
                non_conforming.add(filename)
    
    return non_conforming

def determine_bundestag_period(date_str):
    """Determine which Bundestag period a date belongs to"""
    date = datetime.strptime(date_str, "%Y%m%d")
    
    for period, (start, end) in bundestag_periods.items():
        if start <= date <= end:
            return period
    
    return None

def map_people_to_bundestag_matrix(input_file):
    """Create a matrix mapping people to their Bundestag periods with 0s and 1s"""
    people_votes = defaultdict(set)  # Person -> set of dates they voted
    people_bundestag = defaultdict(set)  # Person -> set of Bundestag periods
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row['filename']
            person = row['Bezeichnung']
            
            # Extract date from filename (assuming YYYYMMDD_*)
            match = re.match(r'^(\d{8})_.*$', filename)
            if match:
                date_str = match.group(1)
                people_votes[person].add(date_str)
                
                # Determine which Bundestag period this vote belongs to
                period = determine_bundestag_period(date_str)
                if period:
                    people_bundestag[person].add(period)
    
    # Create a DataFrame with the matrix structure
    bundestag_numbers = sorted(bundestag_periods.keys())
    data = []
    
    for person in sorted(people_bundestag.keys()):
        row = [person]
        for period in bundestag_numbers:
            presence = 1 if period in people_bundestag[person] else 0
            row.append(presence)
        data.append(row)
    
    # Create column names
    columns = ['Name'] + [str(period) for period in bundestag_numbers]
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)
    return df

def main():
    input_file = "bundestag/all_votes.csv"
    output_file = "bundestag/person_bundestag_matrix.csv"
    
    non_conforming = validate_filenames(input_file)
    if non_conforming:
        print("The following filenames do not conform to the pattern YYYYMMDD_*:")
        for filename in sorted(non_conforming):
            print(f"  - {filename}")
        return
    
    print("All filenames conform to the required pattern.")
    
    # Create the membership matrix
    membership_df = map_people_to_bundestag_matrix(input_file)
    
    # Display the first few rows
    print("\nBundestag Membership Matrix (first few rows):")
    print(membership_df.head())
    
    # Write the results to a CSV file
    membership_df.to_csv(output_file, index=False)
    print(f"\nResults have been written to {output_file}")
    
    # Print some statistics
    total_people = len(membership_df)
    print(f"\nTotal number of people: {total_people}")
    
    for period in sorted(bundestag_periods.keys()):
        count = membership_df[str(period)].sum()
        print(f"  - {period}. Bundestag: {count} people")

if __name__ == "__main__":
    main()