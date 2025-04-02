import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

INPUT_DATA = 'data/baseball_hitting.csv'
OUTPUT_TRAIN = 'data/processed_train_data.csv'
OUTPUT_TEST = 'data/processed_test_data.csv'
OUTPUT_TRAIN_SELECTED = 'data/processed_train_selected.csv'
OUTPUT_TEST_SELECTED = 'data/processed_test_selected.csv'

def load_data(input_path):
    df = pd.read_csv(input_path)
    return df

def clean_data(df):
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
    
    df = df.dropna(how='all')
    
    df.loc[
        df['On-base Plus Slugging'].isna(),
        'On-base Plus Slugging'
    ] = (
        df['On-base Percentage'] + df['Slugging Percentage']
    )
    
    df['Strikeouts'] = df['Strikeouts'].replace('--', np.nan)
    df['Strikeouts'] = pd.to_numeric(df['Strikeouts'], errors='coerce')
    
    strikeout_ratio = df[df["Strikeouts"].notna()]['Strikeouts'].sum() / df[df["Strikeouts"].notna()]['At-bat'].sum()
    df.loc[df['Strikeouts'].isna(), 'Strikeouts'] = df.loc[df['Strikeouts'].isna(), 'At-bat'] * strikeout_ratio
    
    df['Caught stealing'] = df['Caught stealing'].replace('--', np.nan)
    df['Caught stealing'] = pd.to_numeric(df['Caught stealing'], errors='coerce')
    caught_stealing_mean = df["Caught stealing"].mean()
    df["Caught stealing"].fillna(caught_stealing_mean, inplace=True)
    
    df = df[(df['Games'] >= 162 * 5) & (df['At-bat'] >= 2500)]
    
    return df

def split_data(df):
    X = df.drop(['Player name', 'position', 'On-base Plus Slugging', 'On-base Percentage', 'Slugging Percentage'], axis=1)
    y = df['On-base Plus Slugging']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

def save_outputs(X_train, X_test, y_train, y_test):
    train_df = X_train.copy()
    train_df['On-base Plus Slugging'] = y_train.values
    train_df.to_csv(OUTPUT_TRAIN, index=False)
    
    test_df = X_test.copy()
    test_df['On-base Plus Slugging'] = y_test.values
    test_df.to_csv(OUTPUT_TEST, index=False)

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    df = load_data(INPUT_DATA)
    df = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    save_outputs(X_train, X_test, y_train, y_test)