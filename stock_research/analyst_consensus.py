import os
import requests
import argparse
import urllib.parse
from datetime import datetime, timedelta
from tqdm import tqdm
import yfinance as yf
from dotenv import load_dotenv

# Load the API key from the .env file
load_dotenv()
API_KEY = os.getenv("FMP_API_KEY")

# Function to fetch the latest price targets for a given analyst within the last month
def get_latest_price_targets(analyst_name):
    # Encode the analyst name for URL
    encoded_name = urllib.parse.quote(analyst_name)
    url = f"https://financialmodelingprep.com/api/v4/price-target-analyst-name?name={encoded_name}&apikey={API_KEY}"
    
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data:
            # Filter data to only those within the last month
            one_month_ago = datetime.utcnow() - timedelta(days=30)
            data = [item for item in data if datetime.strptime(item['publishedDate'], '%Y-%m-%dT%H:%M:%S.%fZ') >= one_month_ago]
            if not data:
                return None
            # Sort data by publishedDate descending
            data.sort(key=lambda x: x['publishedDate'], reverse=True)
            # For each symbol, keep only the latest alert
            latest_alerts = {}
            for item in data:
                symbol = item['symbol']
                if symbol not in latest_alerts:
                    latest_alerts[symbol] = item
            # Return the list of latest alerts per ticker
            return list(latest_alerts.values())
        else:
            return None
    else:
        raise Exception(f"Error fetching data for {analyst_name}: HTTP {response.status_code}")

# Function to get current stock price using yfinance
def get_current_stock_price(symbol):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period='1d')
        if not data.empty:
            current_price = data['Close'].iloc[-1]
            return current_price
        else:
            return None
    except Exception:
        return None

# Main function to process the input file and print the latest alerts
def fetch_latest_alerts(input_file):
    # Read analyst names from the input file
    try:
        with open(input_file, 'r') as f:
            analyst_names = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Input file {input_file} not found.")
        return

    if not analyst_names:
        print("No analyst names found in the input file.")
        return

    print(f"Fetching latest price target alerts for {len(analyst_names)} analysts...\n")
    
    # Iterate over the analyst names and fetch their latest alerts
    for analyst_name in tqdm(analyst_names, desc="Processing analysts"):
        try:
            latest_alerts = get_latest_price_targets(analyst_name)
            if latest_alerts:
                for alert in latest_alerts:
                    # Extract relevant information
                    symbol = alert.get('symbol', 'N/A')
                    published_date = alert.get('publishedDate', 'N/A')
                    news_url = alert.get('newsURL', 'N/A')
                    news_title = alert.get('newsTitle', 'N/A')
                    price_target = alert.get('priceTarget', 'N/A')
                    price_when_posted = alert.get('priceWhenPosted', 'N/A')
                    analyst_company = alert.get('analystCompany', 'N/A')
                    news_publisher = alert.get('newsPublisher', 'N/A')

                    # Convert published_date to readable format
                    if published_date != 'N/A':
                        published_date_dt = datetime.strptime(published_date, '%Y-%m-%dT%H:%M:%S.%fZ')
                        published_date_str = published_date_dt.strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        published_date_str = 'N/A'

                    # Fetch current stock price using yfinance
                    current_price = get_current_stock_price(symbol)
                    if current_price is not None and price_target is not None:
                        # Calculate delta between current price and price target
                        delta = price_target - current_price
                        delta_percentage = (delta / current_price) * 100
                        delta_str = f"{delta:.2f} ({delta_percentage:.2f}%)"
                        current_price_str = f"{current_price:.2f}"
                    else:
                        current_price_str = 'N/A'
                        delta_str = 'N/A'

                    # Print the latest alert information
                    print(f"Analyst: {analyst_name}")
                    print(f"Company: {analyst_company}")
                    print(f"Symbol: {symbol}")
                    print(f"Published Date: {published_date_str}")
                    print(f"Price Target: {price_target}")
                    print(f"Current Price: {current_price_str}")
                    print(f"Delta to Price Target: {delta_str}")
                    print(f"News Title: {news_title}")
                    print(f"News URL: {news_url}")
                    print(f"News Publisher: {news_publisher}")
                    print("-" * 80)
            else:
                print(f"No recent price target data available for analyst: {analyst_name}")
                print("-" * 80)
        except Exception as e:
            print(f"Error processing analyst {analyst_name}: {e}")
            print("-" * 80)
