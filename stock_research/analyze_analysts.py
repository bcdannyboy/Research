import os
import sys
import argparse
import requests
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
import plotly.express as px
from datetime import timedelta
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Load the API key from the .env file
load_dotenv()
API_KEY = os.getenv("FMP_API_KEY")

# Maximum number of requests per minute
MAX_REQUESTS_PER_MINUTE = 500

# Lock for thread-safe operations
lock = Lock()

# Global variables for rate limiting
request_times = []

def rate_limited_request(url):
    global request_times
    current_time = time.time()
    with lock:
        # Remove timestamps older than 60 seconds
        request_times = [t for t in request_times if current_time - t < 60]

        if len(request_times) >= MAX_REQUESTS_PER_MINUTE:
            # Sleep until we can make a new request
            sleep_time = 60 - (current_time - request_times[0]) + 0.1
            time.sleep(sleep_time)
            # Update current time and request_times after sleep
            current_time = time.time()
            request_times = [t for t in request_times if current_time - t < 60]

        request_times.append(current_time)

    # Make the actual API request
    response = requests.get(url)
    return response

# Function to get the price target data
def get_price_target(symbol):
    url = f"https://financialmodelingprep.com/api/v4/price-target?symbol={symbol}&apikey={API_KEY}"
    response = rate_limited_request(url)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        raise Exception(f"Error fetching price target data for {symbol}: {response.status_code}")

# Function to get historical data using yfinance
def get_historical_data(symbol):
    stock = yf.Ticker(symbol)
    hist = stock.history(period="max")
    if not hist.empty:
        return hist
    else:
        raise Exception(f"No historical data available for symbol: {symbol}")

# Function to analyze analyst skill based on predictions
def analyze_analyst_skill(symbols, output_file, time_horizon_days=90):
    all_price_targets = []
    all_hist_data = {}

    # Fetch price target data in parallel
    with ThreadPoolExecutor(max_workers=50) as executor:
        future_to_symbol = {executor.submit(get_price_target, symbol): symbol for symbol in symbols}
        for future in tqdm(as_completed(future_to_symbol), total=len(symbols), desc="Fetching price target data"):
            symbol = future_to_symbol[future]
            try:
                price_targets = future.result()
                if price_targets:
                    price_target_df = pd.DataFrame(price_targets)
                    price_target_df["symbol"] = symbol
                    all_price_targets.append(price_target_df)
            except Exception as e:
                pass  # Ignore errors for individual symbols

    if not all_price_targets:
        print("No price target data to analyze.")
        return

    all_price_target_df = pd.concat(all_price_targets, ignore_index=True)
    all_price_target_df["publishedDate"] = pd.to_datetime(all_price_target_df["publishedDate"]).dt.normalize()
    all_price_target_df.set_index("publishedDate", inplace=True)

    # Fetch historical data in parallel
    with ThreadPoolExecutor(max_workers=50) as executor:
        future_to_symbol = {executor.submit(get_historical_data, symbol): symbol for symbol in symbols}
        for future in tqdm(as_completed(future_to_symbol), total=len(symbols), desc="Fetching historical data"):
            symbol = future_to_symbol[future]
            try:
                hist_data = future.result()
                hist_data = hist_data["Close"]
                hist_data.index = hist_data.index.normalize()
                all_hist_data[symbol] = hist_data
            except Exception as e:
                pass  # Ignore errors for individual symbols

    # Prepare data for analysis
    analyst_skill = {}

    # Analyze each symbol
    for symbol in tqdm(symbols, desc="Analyzing symbols"):
        if symbol not in all_hist_data:
            continue

        hist_data = all_hist_data[symbol]
        price_target_df = all_price_target_df[all_price_target_df["symbol"] == symbol]

        if price_target_df.empty:
            continue

        # For each price target, compute the error after the time horizon
        errors = []
        for idx, row in price_target_df.iterrows():
            published_date = idx
            target_date = published_date + timedelta(days=time_horizon_days)

            # Find the closest available date in historical data
            actual_dates = hist_data.index
            if target_date > actual_dates[-1]:
                # Cannot compute error if target date is beyond available historical data
                continue
            idx_closest = actual_dates.get_indexer([target_date], method='nearest')[0]
            if idx_closest == -1:
                continue  # Skip if no nearest date is found
            actual_date = actual_dates[idx_closest]

            actual_price = hist_data.loc[actual_date]

            # Compute percentage error
            predicted_price = row["priceTarget"]
            percentage_error = abs((actual_price - predicted_price) / predicted_price) * 100
            errors.append({
                "analystName": row["analystName"],
                "analystCompany": row["analystCompany"],
                "publisher": row["newsPublisher"],
                "symbol": symbol,
                "predicted_price": predicted_price,
                "actual_price": actual_price,
                "percentage_error": percentage_error
            })

        if not errors:
            continue

        errors_df = pd.DataFrame(errors)

        # Aggregate errors per analyst
        for name, group in errors_df.groupby("analystName"):
            avg_error = group["percentage_error"].mean()
            count = len(group)
            diversity = group["symbol"].nunique()
            breadth = count * diversity  # Number of predictions times number of unique symbols
            accuracy = 1 / avg_error if avg_error != 0 else float('inf')  # Avoid division by zero
            skill_score = accuracy * breadth  # Skill is breadth x accuracy

            if name in analyst_skill:
                analyst_skill[name]["breadth"] += breadth
                analyst_skill[name]["total_predictions"] += count
                analyst_skill[name]["total_error"] += group["percentage_error"].sum()
                analyst_skill[name]["unique_symbols"].update(group["symbol"].unique())
            else:
                analyst_skill[name] = {
                    "analystCompany": group["analystCompany"].iloc[0],
                    "publisher": group["publisher"].iloc[0],
                    "breadth": breadth,
                    "total_predictions": count,
                    "total_error": group["percentage_error"].sum(),
                    "unique_symbols": set(group["symbol"].unique())
                }

    # Calculate final skill score for each analyst
    for name in analyst_skill:
        total_error = analyst_skill[name]["total_error"]
        total_predictions = analyst_skill[name]["total_predictions"]
        avg_error = total_error / total_predictions if total_predictions != 0 else float('inf')
        accuracy = 1 / avg_error if avg_error != 0 else float('inf')
        breadth = analyst_skill[name]["breadth"]
        skill_score = accuracy * breadth
        analyst_skill[name]["avg_error"] = avg_error
        analyst_skill[name]["accuracy"] = accuracy
        analyst_skill[name]["skill_score"] = skill_score
        analyst_skill[name]["diversity"] = len(analyst_skill[name]["unique_symbols"])

    # Convert analyst_skill to DataFrame
    analyst_skill_df = pd.DataFrame.from_dict(analyst_skill, orient='index')
    analyst_skill_df.reset_index(inplace=True)
    analyst_skill_df.rename(columns={'index': 'analystName'}, inplace=True)

    # Sort analysts by skill score
    analyst_skill_df.sort_values(by='skill_score', ascending=False, inplace=True)

    # Write the results to the output file
    columns_to_output = [
        'analystName', 'analystCompany', 'publisher', 'skill_score',
        'accuracy', 'avg_error', 'breadth', 'total_predictions', 'diversity'
    ]
    analyst_skill_df.to_csv(output_file, columns=columns_to_output, index=False)

    print(f"Analyst skill scores have been written to {output_file}")

    # Plotting the skill score for visualization using Plotly
    if not analyst_skill_df.empty:
        fig = px.scatter(
            analyst_skill_df,
            x='total_predictions',
            y='skill_score',
            text='analystName',
            labels={"total_predictions": "Number of Predictions", "skill_score": "Skill Score"},
            title="Analyst Skill: Number of Predictions vs Skill Score",
            hover_name='analystName',
            hover_data={
                "Avg Error (%)": analyst_skill_df['avg_error'],
                "Company": analyst_skill_df['analystCompany'],
                "Publisher": analyst_skill_df['publisher'],
                "Diversity": analyst_skill_df['diversity']
            }
        )
        fig.update_traces(marker=dict(size=12, opacity=0.6), mode='markers+text', textposition='top center')
        fig.show()
    else:
        print("No analyst skill scores could be calculated.")
