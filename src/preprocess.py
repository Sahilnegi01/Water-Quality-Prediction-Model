import pandas as pd
import sys
import yaml
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load parameters from params.yaml
params = yaml.safe_load(open("params.yaml"))['preprocess']

def preprocess(input_path, output_path):
    try:
        # Read the input CSV file
        data = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: The file at {input_path} was not found.")
        return

    # Handle missing values
    for col in data.columns:
        if data[col].dtype in ['int64', 'float64']:  
            data[col] = data[col].fillna(data[col].mean())  # Fill numeric columns with mean
        else:  
            data[col] = data[col].fillna(data[col].mode()[0])  # Fill non-numeric columns with mode

    # Encode categorical columns if they exist
    le = LabelEncoder()

    if 'Month' in data.columns:
        data['Month'] = data['Month'].apply(
            lambda x: int(x) if str(x).isdigit() else x
        )  # Ensure Month is numeric before encoding
        data['Month'] = le.fit_transform(data['Month'])  # Encode Month
    
    if 'WaterType' in data.columns:
        data['WaterType'] = le.fit_transform(data['WaterType'])  # Encode WaterType

    if 'Color' in data.columns:
        data['Color'] = le.fit_transform(data['Color'])  # Encode Color

    # Add Season column based on Month
    if 'Month' in data.columns:
        data['Season'] = data['Month'].apply(
            lambda x: 0 if x in [12, 1, 2] else
                      1 if x in [3, 4, 5] else
                      2 if x in [6, 7, 8] else
                      3
        )
    
    data['Source'] = le.fit_transform(data['Source'])


    # Standardize numeric columns
    numeric_columns = ['Sulfate', 'pH', 'Water Temperature', 'Turbidity']
    for col in numeric_columns:
        if col not in data.columns:
            print(f"Warning: '{col}' column not found in the dataset.")
            numeric_columns.remove(col)  # Remove missing columns from standardization

    if numeric_columns:  # Proceed if numeric columns exist
        scaler = StandardScaler()
        data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Drop rows with any remaining missing values
    data.dropna(inplace=True)

    # Convert all integer columns to float for consistency
    for column in data.select_dtypes(include='int64').columns:
        data[column] = data[column].astype('float64')

    # Save the processed data to the output path
    data.to_csv(output_path, header=True, index=False)
    print(f"Preprocessed data saved to: {output_path}")

if __name__ == "__main__":
    preprocess(params["input"], params["output"])