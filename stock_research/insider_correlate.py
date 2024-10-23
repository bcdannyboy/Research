#!/usr/bin/env python3
"""
Script to analyze the correlation between insider trades and price movements
for a list of stock symbols provided in symbols.txt.

This script fetches insider trading data, historical price data, earnings dates,
news, and social sentiment for each symbol concurrently. It performs both linear
and non-linear (copula-based) correlation analyses to identify the 'skill' of
specific insiders and general patterns that correlate to price movements.

Interactive plots are generated to visualize the findings.

Usage:
    python insider_trading_analysis.py

Dependencies:
    - Python 3.x
    - pandas
    - numpy
    - scipy
    - statsmodels
    - requests
    - matplotlib
    - seaborn
    - plotly
    - python-dotenv
    - copulas
    - yfinance
    - aiohttp
    - asyncio
    - nest_asyncio
"""

import os
import sys
import requests
import pandas as pd
import numpy as np
import datetime
from scipy.stats import kendalltau, spearmanr, pearsonr
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from copulas.multivariate import GaussianMultivariate
from dateutil.parser import parse as date_parse
from dotenv import load_dotenv
import yfinance as yf
import asyncio
import aiohttp
import nest_asyncio
from concurrent.futures import ThreadPoolExecutor

# Allow nested event loops (necessary when running in certain environments)
nest_asyncio.apply()

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv('FMP_API_KEY')

if not API_KEY:
    print("Error: FMP_API_KEY not found in .env file.")
    sys.exit(1)

# Base URL for FinancialModelingPrep API
BASE_URL_V4 = 'https://financialmodelingprep.com/api/v4'
BASE_URL_V3 = 'https://financialmodelingprep.com/api/v3'

# Maximum number of concurrent requests
MAX_CONCURRENT_REQUESTS = 400


def read_symbols(filename='test.txt'):
    """
    Read stock symbols from a text file.

    Args:
        filename (str): Path to the file containing stock symbols.

    Returns:
        list: List of stock symbols.
    """
    try:
        with open(filename, 'r') as file:
            symbols = [line.strip().upper() for line in file if line.strip()]
        return symbols
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        sys.exit(1)


async def fetch(session, url):
    """
    Asynchronous function to fetch data from a URL.

    Args:
        session (aiohttp.ClientSession): The aiohttp session.
        url (str): The URL to fetch.

    Returns:
        dict or list: The JSON response.
    """
    async with session.get(url) as response:
        if response.status == 200:
            try:
                data = await response.json()
                return data
            except Exception as e:
                print(f"Error parsing JSON response from {url}: {e}")
                return None
        else:
            print(f"Error fetching data from {url}: HTTP {response.status}")
            return None


async def fetch_insider_trading_data(session, symbol):
    """
    Asynchronously fetch insider trading data for a given symbol.

    Args:
        session (aiohttp.ClientSession): The aiohttp session.
        symbol (str): Stock symbol.

    Returns:
        pandas.DataFrame: Insider trading data.
    """
    url = f"{BASE_URL_V4}/insider-trading?symbol={symbol}&apikey={API_KEY}&limit=1000"
    data = await fetch(session, url)
    if data:
        return pd.DataFrame(data)
    else:
        return pd.DataFrame()


def fetch_historical_price_data(symbol, start_date, end_date):
    """
    Fetch historical price data for a given symbol using yfinance.

    Args:
        symbol (str): Stock symbol.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pandas.DataFrame: Historical price data.
    """
    try:
        print(f"Fetching historical price data for {symbol}, from {start_date} to {end_date}")
        # Ensure dates are strings
        start_date = start_date.strftime('%Y-%m-%d') if isinstance(start_date, datetime.datetime) else start_date
        end_date = end_date.strftime('%Y-%m-%d') if isinstance(end_date, datetime.datetime) else end_date

        # Fetch data using yfinance
        df = yf.download(symbol, start=start_date, end=end_date, progress=False, threads=False)
        if df.empty:
            print(f"No price data available for {symbol}.")
            return pd.DataFrame()
        df.reset_index(inplace=True)
        df.rename(columns={'Date': 'date', 'Close': 'close'}, inplace=True)
        return df[['date', 'close']]
    except Exception as e:
        print(f"Error fetching historical price data for {symbol}: {e}")
        return pd.DataFrame()


async def fetch_earnings_dates(session, symbol):
    """
    Asynchronously fetch earnings call dates for a given symbol.

    Args:
        session (aiohttp.ClientSession): The aiohttp session.
        symbol (str): Stock symbol.

    Returns:
        pandas.DataFrame: Earnings call dates.
    """
    url = f"{BASE_URL_V4}/earning_call_transcript?symbol={symbol}&apikey={API_KEY}"
    data = await fetch(session, url)
    if data:
        df = pd.DataFrame(data)
        if not df.empty and 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df.dropna(subset=['Date'], inplace=True)
            return df[['Quarter', 'Year', 'Date']]
        else:
            return pd.DataFrame()
    else:
        return pd.DataFrame()


async def fetch_social_sentiment(session, symbol):
    """
    Asynchronously fetch historical social sentiment data for a given symbol.

    Args:
        session (aiohttp.ClientSession): The aiohttp session.
        symbol (str): Stock symbol.

    Returns:
        pandas.DataFrame: Social sentiment data.
    """
    url = f"{BASE_URL_V4}/historical/social-sentiment?symbol={symbol}&apikey={API_KEY}&limit=1000"
    data = await fetch(session, url)
    if data:
        return pd.DataFrame(data)
    else:
        return pd.DataFrame()


async def fetch_news(session, symbol, start_date, end_date):
    """
    Asynchronously fetch stock news for a given symbol between specified dates.

    Args:
        session (aiohttp.ClientSession): The aiohttp session.
        symbol (str): Stock symbol.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pandas.DataFrame: Stock news data.
    """
    url = f"{BASE_URL_V3}/stock_news?tickers={symbol}&from={start_date}&to={end_date}&apikey={API_KEY}&limit=1000"
    data = await fetch(session, url)
    if data:
        return pd.DataFrame(data)
    else:
        return pd.DataFrame()


def process_insider_trades(insider_trades):
    """
    Process insider trades data.

    Args:
        insider_trades (pandas.DataFrame): Insider trades data.

    Returns:
        pandas.DataFrame: Processed insider trades data.
    """
    if insider_trades.empty:
        return insider_trades

    # Convert date columns to datetime
    insider_trades['transactionDate'] = pd.to_datetime(insider_trades['transactionDate'], errors='coerce')
    insider_trades['filingDate'] = pd.to_datetime(insider_trades['filingDate'], errors='coerce')

    # Drop rows with invalid dates
    insider_trades.dropna(subset=['transactionDate', 'filingDate'], inplace=True)

    # Remove trades with transactionDate in the future
    today = pd.Timestamp(datetime.datetime.today().date())
    insider_trades = insider_trades[insider_trades['transactionDate'] <= today]

    # Sort by transaction date
    insider_trades.sort_values('transactionDate', inplace=True)

    return insider_trades


def analyze_insider_skill(insider_trades, price_data):
    """
    Analyze the 'skill' of insiders by correlating their trades with future price movements.

    Args:
        insider_trades (pandas.DataFrame): Insider trades data.
        price_data (pandas.DataFrame): Historical price data.

    Returns:
        pandas.DataFrame: Insider skill analysis results.
    """
    results = []
    insiders = insider_trades['reportingName'].unique()

    for insider in insiders:
        insider_data = insider_trades[insider_trades['reportingName'] == insider]
        correlations = []
        for _, trade in insider_data.iterrows():
            # Get the transaction date
            transaction_date = trade['transactionDate']

            # Get price movement after the trade (next 30 days)
            future_prices = price_data[price_data['date'] > transaction_date].head(30)
            if not future_prices.empty:
                # Calculate price change percentage
                price_change = (future_prices['close'].values[-1] - future_prices['close'].values[0]) / future_prices['close'].values[0]

                # Assign +1 for buy, -1 for sell
                if trade['acquistionOrDisposition'] == 'A':
                    trade_direction = 1
                elif trade['acquistionOrDisposition'] == 'D':
                    trade_direction = -1
                else:
                    trade_direction = 0  # Neutral or unknown

                # Correlate trade direction with price change
                correlations.append(trade_direction * price_change)

        if correlations:
            average_correlation = np.mean(correlations)
            results.append({
                'Insider': insider,
                'AverageCorrelation': average_correlation,
                'NumberOfTrades': len(correlations)
            })

    return pd.DataFrame(results)


def perform_correlation_analysis(insider_trades, price_data):
    """
    Perform linear and non-linear correlation analysis between insider trades and price movements.

    Args:
        insider_trades (pandas.DataFrame): Insider trades data.
        price_data (pandas.DataFrame): Historical price data.

    Returns:
        dict: Correlation analysis results.
    """
    if insider_trades.empty or price_data.empty:
        return {}

    # Prepare data
    insider_trades['transactionDate'] = pd.to_datetime(insider_trades['transactionDate'], errors='coerce')
    price_data = price_data.copy()
    price_data['date'] = pd.to_datetime(price_data['date'], errors='coerce')

    # Drop rows with invalid dates
    insider_trades.dropna(subset=['transactionDate'], inplace=True)
    price_data.dropna(subset=['date'], inplace=True)

    # Calculate daily returns
    price_data.set_index('date', inplace=True)
    price_data.sort_index(inplace=True)
    price_data['Return'] = price_data['close'].pct_change()

    # Aggregate insider trades per day
    daily_trades = insider_trades.groupby('transactionDate').agg({
        'securitiesTransacted': 'sum',
        'price': 'mean'
    }).rename(columns={'price': 'AverageTradePrice'})

    # Merge with price returns
    analysis_data = daily_trades.join(price_data['Return'], how='inner')

    # Drop NaN values
    analysis_data.dropna(inplace=True)

    if analysis_data.empty:
        return {}

    # Linear correlations
    pearson_corr, _ = pearsonr(analysis_data['securitiesTransacted'], analysis_data['Return'])
    spearman_corr, _ = spearmanr(analysis_data['securitiesTransacted'], analysis_data['Return'])
    kendall_corr, _ = kendalltau(analysis_data['securitiesTransacted'], analysis_data['Return'])

    # Non-linear (copula-based) correlation
    copula_data = analysis_data[['securitiesTransacted', 'Return']].dropna()
    if len(copula_data) < 2:
        copula_corr = None
    else:
        copula_model = GaussianMultivariate()
        copula_model.fit(copula_data)
        copula_corr = copula_model.correlation  # Fixed here, use 'correlation' instead of 'copula.covariance'

    return {
        'pearson_correlation': pearson_corr,
        'spearman_correlation': spearman_corr,
        'kendall_correlation': kendall_corr,
        'copula_covariance_matrix': copula_corr,
        'analysis_data': analysis_data  # Return data for plotting
    }


def analyze_trade_patterns(insider_trades):
    """
    Analyze general patterns in insider trades.

    Args:
        insider_trades (pandas.DataFrame): Insider trades data.

    Returns:
        pandas.Series, pandas.Series: Trade size statistics and trade type counts.
    """
    if insider_trades.empty:
        return pd.Series(dtype=float), pd.Series(dtype=int)

    # Distribution of trade sizes
    trade_sizes = insider_trades['securitiesTransacted'].abs()
    trade_types = insider_trades['transactionType']

    # Summary statistics
    trade_size_stats = trade_sizes.describe()
    trade_type_counts = trade_types.value_counts()

    return trade_size_stats, trade_type_counts


def correlate_with_events(insider_trades, earnings_dates, news_data):
    """
    Analyze timing of insider trades relative to earnings events and news.

    Args:
        insider_trades (pandas.DataFrame): Insider trades data.
        earnings_dates (pandas.DataFrame): Earnings call dates.
        news_data (pandas.DataFrame): News articles data.

    Returns:
        pandas.DataFrame: Event correlation analysis results.
    """
    if insider_trades.empty:
        return pd.DataFrame()

    # Convert dates
    insider_trades['transactionDate'] = pd.to_datetime(insider_trades['transactionDate'], errors='coerce')
    if not earnings_dates.empty:
        earnings_dates['Date'] = pd.to_datetime(earnings_dates['Date'], errors='coerce')
    if not news_data.empty:
        news_data['publishedDate'] = pd.to_datetime(news_data['publishedDate'], errors='coerce')

    # Drop rows with invalid dates
    insider_trades.dropna(subset=['transactionDate'], inplace=True)
    if not earnings_dates.empty:
        earnings_dates.dropna(subset=['Date'], inplace=True)
    if not news_data.empty:
        news_data.dropna(subset=['publishedDate'], inplace=True)

    # Calculate days to next earnings call
    if not earnings_dates.empty:
        earnings_dates_sorted = earnings_dates.sort_values('Date')
        insider_trades['DaysToNextEarnings'] = insider_trades['transactionDate'].apply(
            lambda x: (earnings_dates_sorted[earnings_dates_sorted['Date'] >= x]['Date'] - x).dt.days.min()
        )
    else:
        insider_trades['DaysToNextEarnings'] = np.nan

    # Merge news data with insider trades
    if not news_data.empty:
        news_data['Date'] = news_data['publishedDate'].dt.date
        insider_trades['Date'] = insider_trades['transactionDate'].dt.date

        merged_data = pd.merge(insider_trades, news_data, on='Date', how='left', suffixes=('', '_news'))
    else:
        merged_data = insider_trades.copy()

    return merged_data


def perform_sentiment_analysis(sentiment_data):
    """
    Analyze social sentiment data.

    Args:
        sentiment_data (pandas.DataFrame): Social sentiment data.

    Returns:
        pandas.DataFrame: Sentiment analysis results.
    """
    if sentiment_data.empty:
        return pd.DataFrame()

    sentiment_data['date'] = pd.to_datetime(sentiment_data['date'], errors='coerce')
    sentiment_data.dropna(subset=['date'], inplace=True)
    # Calculate average sentiment
    sentiment_data['AverageSentiment'] = (sentiment_data['stocktwitsSentiment'] + sentiment_data['twitterSentiment']) / 2

    return sentiment_data


def generate_interactive_plots(symbol, insider_trades, insider_skill, correlation_results, trade_size_stats, trade_type_counts, sentiment_analysis):
    """
    Generate interactive plots using Plotly.

    Args:
        symbol (str): Stock symbol.
        insider_trades (pandas.DataFrame): Insider trades data.
        insider_skill (pandas.DataFrame): Insider skill analysis results.
        correlation_results (dict): Correlation analysis results.
        trade_size_stats (pandas.Series): Trade size statistics.
        trade_type_counts (pandas.Series): Trade type counts.
        sentiment_analysis (pandas.DataFrame): Sentiment analysis results.
    """
    # Plot insider skill
    if not insider_skill.empty:
        fig_skill = px.bar(insider_skill, x='Insider', y='AverageCorrelation',
                           title=f'Insider Skill Analysis for {symbol}')
        fig_skill.show()

    # Plot correlation scatter
    if correlation_results and 'analysis_data' in correlation_results:
        analysis_data = correlation_results['analysis_data']
        fig_corr = px.scatter(analysis_data, x='securitiesTransacted', y='Return',
                              title=f'Correlation between Insider Trades and Returns for {symbol}',
                              trendline='ols')
        fig_corr.show()

    # Plot trade size distribution
    if not insider_trades.empty:
        fig_trade_size = px.histogram(insider_trades, x='securitiesTransacted',
                                      title=f'Trade Size Distribution for {symbol}')
        fig_trade_size.show()

    # Plot trade type counts
    if not trade_type_counts.empty:
        fig_trade_type = px.bar(trade_type_counts, x=trade_type_counts.index, y=trade_type_counts.values,
                                title=f'Trade Type Counts for {symbol}',
                                labels={'x': 'Transaction Type', 'y': 'Count'})
        fig_trade_type.show()

    # Plot sentiment over time
    if not sentiment_analysis.empty:
        fig_sentiment = px.line(sentiment_analysis, x='date', y='AverageSentiment',
                                title=f'Social Sentiment Over Time for {symbol}')
        fig_sentiment.show()


async def main():
    """
    Main function to orchestrate the analysis.
    """
    # Read symbols from file
    symbols = read_symbols()

    # Create an aiohttp session
    async with aiohttp.ClientSession() as session:

        # Limit the number of concurrent tasks
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

        # Define tasks for fetching data concurrently
        tasks = []
        for symbol in symbols:
            tasks.append(process_symbol(session, semaphore, symbol))

        # Run tasks concurrently
        await asyncio.gather(*tasks)

    print("Analysis completed for all symbols.")


async def process_symbol(session, semaphore, symbol):
    """
    Process data and analysis for a single symbol.

    Args:
        session (aiohttp.ClientSession): The aiohttp session.
        semaphore (asyncio.Semaphore): Semaphore to limit concurrent tasks.
        symbol (str): Stock symbol.
    """
    async with semaphore:
        print(f"Processing symbol: {symbol}")

        # Fetch data concurrently
        insider_trades_task = fetch_insider_trading_data(session, symbol)
        earnings_dates_task = fetch_earnings_dates(session, symbol)
        sentiment_data_task = fetch_social_sentiment(session, symbol)

        # Fetch insider trades, earnings dates, and sentiment data
        insider_trades, earnings_dates, sentiment_data = await asyncio.gather(
            insider_trades_task, earnings_dates_task, sentiment_data_task
        )

        # Process data
        insider_trades = process_insider_trades(insider_trades)

        if insider_trades.empty:
            print(f"No insider trading data for {symbol}. Skipping analysis.")
            return

        # Determine date range from insider trades
        min_transaction_date = insider_trades['transactionDate'].min()
        if pd.isnull(min_transaction_date):
            print(f"No valid transaction dates for {symbol}. Skipping analysis.")
            return
        start_date = min_transaction_date

        # Ensure start_date is not later than today
        today = datetime.datetime.today()
        if start_date > today:
            print(f"Start date {start_date} is after today for {symbol}. Skipping analysis.")
            return

        # For safety, set a maximum lookback period, e.g., 5 years
        earliest_date = today - datetime.timedelta(days=5*365)
        if start_date < earliest_date:
            start_date = earliest_date

        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = today.strftime('%Y-%m-%d')

        # Fetch news data concurrently
        news_data_task = fetch_news(session, symbol, start_date_str, end_date_str)

        # Fetch historical price data using ThreadPoolExecutor
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            price_data_future = loop.run_in_executor(executor, fetch_historical_price_data, symbol, start_date_str, end_date_str)
            news_data = await news_data_task
            price_data = await price_data_future

        if price_data.empty:
            print(f"No price data for {symbol}. Skipping analysis.")
            return

        # Analyze insider skill
        insider_skill = analyze_insider_skill(insider_trades, price_data)
        print(f"Insider skill analysis for {symbol}:")
        print(insider_skill)

        # Perform correlation analysis
        correlation_results = perform_correlation_analysis(insider_trades, price_data)
        if correlation_results:
            print(f"Correlation analysis for {symbol}:")
            print({
                'pearson_correlation': correlation_results.get('pearson_correlation'),
                'spearman_correlation': correlation_results.get('spearman_correlation'),
                'kendall_correlation': correlation_results.get('kendall_correlation')
            })
        else:
            print(f"Not enough data for correlation analysis for {symbol}.")

        # Analyze trade patterns
        trade_size_stats, trade_type_counts = analyze_trade_patterns(insider_trades)
        print(f"Trade size statistics for {symbol}:")
        print(trade_size_stats)
        print(f"Trade type counts for {symbol}:")
        print(trade_type_counts)

        # Correlate with events
        event_correlation = correlate_with_events(insider_trades, earnings_dates, news_data)
        print(f"Event correlation analysis for {symbol} completed.")

        # Perform sentiment analysis
        sentiment_analysis = perform_sentiment_analysis(sentiment_data)
        print(f"Sentiment analysis for {symbol} completed.")

        # Generate interactive plots
        generate_interactive_plots(symbol, insider_trades, insider_skill, correlation_results, trade_size_stats,
                                   trade_type_counts, sentiment_analysis)


if __name__ == '__main__':
    # Run the main function using asyncio
    asyncio.run(main())
