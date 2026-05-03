import pandas as pd
import glob

# 1. User Inputs
pattern = input("Enter the file pattern (e.g., sales_*.csv): ")
output_name = input("Enter the name for the merged file (e.g., final_data.csv): ")

# 2. Find files based on user input
all_files = glob.glob(pattern)

if not all_files:
    print("No files found matching that pattern!")
else:
# 3. Fast Load & Merge
    df = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)

# 4.Clean headers: lowercase and replace space
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    # Convert dates: Automatically handles multiple formats
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date

# Remove duplicates
    df = df.drop_duplicates()

    # 5. Export
    df.to_csv(output_name, index=False)
    print(f"\n--- Process Complete ---")
    print(f"Files Merged: {all_files}")
    print(f"Final Row Count: {len(df)}")
    print(f"Saved as: {output_name}")

