import pandas as pd

# Define column names for the UCI Adult dataset
columns = [
    "age", "workclass", "fnlwgt", "education", "education-num", 
    "marital-status", "occupation", "relationship", "race", "sex", 
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

# Specify the file path to adult.data
file_path = "adult.data"

try:
    # Read the data into a pandas DataFrame
    df = pd.read_csv(file_path, names=columns, skipinitialspace=True)

    # Count '?' values in each column before imputation
    missing_counts = (df == '?').sum()
    print("Count of '?' (missing values) in each column before imputation:")
    for column, count in missing_counts.items():
        print(f"{column}: {count} missing values")
    print(f"\nTotal rows in dataset: {len(df)}")

    # Columns with missing values ('?')
    columns_with_missing = ['workclass', 'occupation', 'native-country']

    # Impute '?' values with mode based on income label
    for column in columns_with_missing:
        # Get unique income labels
        income_labels = df['income'].unique()
        
        for label in income_labels:
            # Filter rows for the current income label
            mask = df['income'] == label
            # Compute mode for the column within this income group (excluding '?')
            mode_value = df[mask & (df[column] != '?')][column].mode()
            
            if not mode_value.empty:
                # Use the first mode value if multiple modes exist
                mode_value = mode_value[0]
                # Replace '?' with mode for rows with this income label
                df.loc[mask & (df[column] == '?'), column] = mode_value
            else:
                # If no mode (e.g., all values are '?'), use the global mode
                global_mode = df[column][df[column] != '?'].mode()[0]
                df.loc[mask & (df[column] == '?'), column] = global_mode

    # Verify no '?' values remain
    missing_counts_after = (df == '?').sum()
    print("\nCount of '?' (missing values) in each column after imputation:")
    for column, count in missing_counts_after.items():
        print(f"{column}: {count} missing values")

    # Save the imputed DataFrame to a new file
    output_file = "adult_imputed.data"
    df.to_csv(output_file, index=False, header=False)
    print(f"\nImputed dataset saved to '{output_file}'")

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please provide the correct path.")
except Exception as e:
    print(f"An error occurred: {str(e)}")