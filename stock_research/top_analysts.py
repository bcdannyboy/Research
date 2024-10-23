import pandas as pd
import argparse

def find_top_analysts(input_csv, top_n=10, min_predictions=5):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_csv)

    # Handle infinite or NaN accuracy values (due to zero avg_error)
    df.replace([float('inf'), float('nan')], 0, inplace=True)

    # Filter out analysts with fewer than min_predictions
    df_filtered = df[df['total_predictions'] >= min_predictions]

    # Calculate the weighted score: accuracy multiplied by total predictions
    df_filtered['weighted_score'] = df_filtered['accuracy'] * df_filtered['total_predictions']

    # Sort the DataFrame based on weighted_score in descending order
    df_sorted = df_filtered.sort_values(by='weighted_score', ascending=False)

    # Select the top N analysts
    top_analysts = df_sorted.head(top_n)

    # Print the top analysts
    print("\nTop Analysts based on accuracy and volume:")
    print(top_analysts[['analystName', 'analystCompany', 'publisher', 'accuracy', 'avg_error', 'total_predictions', 'diversity', 'weighted_score']])
