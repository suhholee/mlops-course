import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def load_data(input_path):
    print(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    return df

def clean_data(df):
    print("Cleaning and transforming data")
    
    # Rename columns
    df = df.rename(
        columns={
            'stolen base': 'Stolen base',
            'Double (2B)': 'Double',
            'third baseman': 'Triple',
            'home run': 'HR',
            'run batted in': 'RBI',
            'a walk': 'Walk'
        }
    )
    
    # Drop rows with all NaN values
    df = df.dropna(how='all')
    
    # Calculate On-base Plus Slugging if missing
    df.loc[
        df['On-base Plus Slugging'].isna(),
        'On-base Plus Slugging'
    ] = (
        df['On-base Percentage'] + df['Slugging Percentage']
    )
    
    # Handle missing Strikeouts
    df['Strikeouts'] = df['Strikeouts'].replace('--', np.nan)
    df['Strikeouts'] = pd.to_numeric(df['Strikeouts'], errors='coerce')
    
    # Impute missing Strikeouts based on At-bat ratio
    strikeout_ratio = df[df["Strikeouts"].notna()]['Strikeouts'].sum() / df[df["Strikeouts"].notna()]['At-bat'].sum()
    df.loc[df['Strikeouts'].isna(), 'Strikeouts'] = df.loc[df['Strikeouts'].isna(), 'At-bat'] * strikeout_ratio
    
    # Handle missing Caught stealing
    df['Caught stealing'] = df['Caught stealing'].replace('--', np.nan)
    df['Caught stealing'] = pd.to_numeric(df['Caught stealing'], errors='coerce')
    caught_stealing_mean = df["Caught stealing"].mean()
    df["Caught stealing"].fillna(caught_stealing_mean, inplace=True)
    
    # Filter players with minimum experience
    df = df[(df['Games'] >= 162 * 5) & (df['At-bat'] >= 2500)]
    
    print(f"Data cleaned. Shape: {df.shape}")
    return df

def split_data(df, test_size=0.2, random_state=42):
    print("Splitting data into train and test sets")
    
    # Drop columns not needed for modeling
    X = df.drop(['Player name', 'position', 'On-base Plus Slugging', 'On-base Percentage', 'Slugging Percentage'], axis=1)
    y = df['On-base Plus Slugging']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def save_outputs(X_train, X_test, y_train, y_test, train_path, test_path):
    print(f"Saving processed data to {train_path} and {test_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    
    # Save training data
    train_df = X_train.copy()
    train_df['On-base Plus Slugging'] = y_train.values
    train_df.to_csv(train_path, index=False)
    
    # Save test data
    test_df = X_test.copy()
    test_df['On-base Plus Slugging'] = y_test.values
    test_df.to_csv(test_path, index=False)
    
    print("Data saved successfully")

def process_data(input_path, output_train_path, output_test_path, test_size=0.2, random_state=42):
    # Load data
    df = load_data(input_path)
    
    # Clean data
    df = clean_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df, test_size, random_state)
    
    # Save processed data
    save_outputs(X_train, X_test, y_train, y_test, output_train_path, output_test_path)
    
    return df, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    INPUT_DATA = './data/baseball_hitting.csv'
    OUTPUT_TRAIN = './data/processed_train_data.csv'
    OUTPUT_TEST = './data/processed_test_data.csv'
    
    process_data(INPUT_DATA, OUTPUT_TRAIN, OUTPUT_TEST)