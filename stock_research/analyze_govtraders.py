import asyncio
import aiohttp
import yfinance as yf
import pandas as pd
import numpy as np
import argparse
import math
import logging
from datetime import datetime, timedelta
from collections import defaultdict
from aiohttp import ClientSession, TCPConnector
from tqdm import tqdm
import os
import sys
import re
from aiolimiter import AsyncLimiter

API_KEY = 'Your_API_Key_Here'  # Replace with your actual API key

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Rate limiter settings
MAX_REQUESTS_PER_MINUTE = 500  # Adjusted to match your actual rate limit

# Global in-memory cache for price data
price_data_cache = {}

def parse_amount(amount_str):
    """
    Parses the transaction amount string and returns the maximum amount.

    Handles various formats such as:
    - "$1,001 - $15,000"
    - "Over $50,000,000"
    - "> $1,000,000"
    - "$15,000"

    Always assumes the maximum amount for estimation.
    """
    if not amount_str:
        logging.warning("Amount string is None or empty.")
        return 0.0
    amount_str = amount_str.strip().replace(',', '')
    try:
        if 'Over' in amount_str:
            # Example: "Over $50,000,000"
            match = re.search(r'Over\s*\$\s*(\d+(?:\.\d+)?)', amount_str)
            if match:
                max_amount = float(match.group(1))
                logging.debug(f"Parsed amount (Over): {max_amount}")
                return max_amount
        elif '-' in amount_str:
            # Example: "$1,001 - $15,000"
            match = re.search(r'\$\s*(\d+(?:\.\d+)?)\s*-\s*\$\s*(\d+(?:\.\d+)?)', amount_str)
            if match:
                max_amount = float(match.group(2))
                logging.debug(f"Parsed amount (Range): {max_amount}")
                return max_amount
        elif '>' in amount_str:
            # Example: "> $1,000,000"
            match = re.search(r'>\s*\$\s*(\d+(?:\.\d+)?)', amount_str)
            if match:
                max_amount = float(match.group(1))
                logging.debug(f"Parsed amount (Greater than): {max_amount}")
                return max_amount
        else:
            # Example: "$15,000"
            amount_value = float(amount_str.replace('$', ''))
            logging.debug(f"Parsed amount (Single value): {amount_value}")
            return amount_value
    except Exception as e:
        logging.warning(f"Unable to parse amount in standard formats: '{amount_str}'. Error: {e}")
        pass  # Continue to fallback

    # Fallback: Extract all digits and try to parse
    try:
        amount_value = float(re.sub(r'[^\d.]', '', amount_str))
        logging.debug(f"Parsed amount (Fallback): {amount_value}")
        return amount_value
    except Exception as e:
        logging.warning(f"Unable to parse amount: '{amount_str}'. Error: {e}")
        return 0.0  # Return 0.0 if parsing fails

async def rate_limited_fetch(session, url, limiter):
    """
    Fetches data from the given URL while respecting the rate limit.

    Args:
        session: The aiohttp ClientSession to use for the request.
        url: The URL to fetch data from.
        limiter: The AsyncLimiter instance for rate limiting.

    Returns:
        The JSON response if successful, None otherwise.
    """
    async with limiter:
        async with session.get(url) as response:
            if response.status == 200:
                return await response.json()
            elif response.status == 429:
                # Handle rate limit exceeded
                logging.error(f"Rate limit exceeded when fetching {url}: Status {response.status}")
                return None
            else:
                logging.error(f"Failed to fetch {url}: Status {response.status}")
                return None

async def fetch_trades(symbols, limiter):
    """
    Fetches trade data for the given list of symbols.

    Args:
        symbols: List of stock symbols to fetch trade data for.
        limiter: The AsyncLimiter instance for rate limiting.

    Returns:
        A list of trade dictionaries with standardized keys.
    """
    tasks = []
    trades = []

    async with ClientSession(connector=TCPConnector(ssl=False)) as session:
        for symbol in symbols:
            # URLs for Senate and House trade data
            senate_url = f'https://financialmodelingprep.com/api/v4/senate-trading?symbol={symbol}&apikey={API_KEY}'
            house_url = f'https://financialmodelingprep.com/api/v4/house-disclosure?symbol={symbol}&apikey={API_KEY}'

            # Schedule fetching tasks
            tasks.append(rate_limited_fetch(session, senate_url, limiter))
            tasks.append(rate_limited_fetch(session, house_url, limiter))

        # Collect responses asynchronously
        responses = []
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Fetching trade data"):
            result = await f
            responses.append(result)

    # Process responses and standardize trade data
    for response in responses:
        if response:
            for trade in response:
                # Normalize and standardize trade data
                if 'firstName' in trade:
                    # Senate trade
                    trader_name = f"{trade.get('firstName', '')} {trade.get('lastName', '')}".strip()
                    trade_data = {
                        'trader': trader_name,
                        'symbol': trade.get('symbol'),
                        'transaction_date': trade.get('transactionDate'),
                        'type': trade.get('type'),
                        'amount': trade.get('amount'),
                        'owner': trade.get('owner'),
                        'asset_description': trade.get('assetDescription'),
                        'source': 'Senate'
                    }
                elif 'representative' in trade:
                    # House trade
                    trader_name = trade.get('representative', '').strip()
                    trade_data = {
                        'trader': trader_name,
                        'symbol': trade.get('ticker'),
                        'transaction_date': trade.get('transactionDate'),
                        'type': trade.get('type'),
                        'amount': trade.get('amount'),
                        'owner': trade.get('owner'),
                        'asset_description': trade.get('assetDescription'),
                        'source': 'House'
                    }
                else:
                    continue  # Skip unrecognized data
                trades.append(trade_data)

    return trades

def get_price_data(symbol):
    """
    Fetches historical price data for the given symbol using yfinance.

    Args:
        symbol: The stock symbol to fetch data for.

    Returns:
        A pandas DataFrame with historical price data, or None if not available.
    """
    if symbol in price_data_cache:
        return price_data_cache[symbol]
    try:
        data = yf.download(symbol, period='max', progress=False)
        if not data.empty:
            data.index = data.index.strftime('%Y-%m-%d')
            price_data_cache[symbol] = data
            return data
        else:
            logging.warning(f"No price data found for {symbol}")
    except Exception as e:
        logging.error(f"Error fetching price data for {symbol}: {e}")
    return None

def compute_trade_performance(trade, price_data, holding_period_days):
    """
    Computes the performance of a single trade over a holding period.

    Args:
        trade: A dictionary containing trade information.
        price_data: A pandas DataFrame with historical price data for the symbol.
        holding_period_days: The holding period in days.

    Returns:
        A dictionary with profit, holding period, and return percentage, or None if calculation fails.
    """
    symbol = trade['symbol']
    transaction_date = trade['transaction_date']
    trade_type = trade['type'].lower()  # Convert to lowercase for case-insensitive comparison
    amount = parse_amount(trade['amount'])

    # Check if amount is valid
    if amount is None or amount == 0:
        logging.warning(f"Amount is None or zero for trade: {trade}")
        return None

    # Find the closest available date on or after the transaction date
    available_dates = price_data.index
    if transaction_date not in available_dates:
        transaction_datetime = datetime.strptime(transaction_date, '%Y-%m-%d')
        future_dates = [date for date in available_dates if datetime.strptime(date, '%Y-%m-%d') >= transaction_datetime]
        if not future_dates:
            logging.warning(f"No future price data for {symbol} after {transaction_date}")
            return None
        transaction_date = future_dates[0]

    # Get transaction price
    transaction_price = price_data.loc[transaction_date]['Close']
    if transaction_price is None or transaction_price == 0 or math.isnan(transaction_price):
        logging.warning(f"Invalid transaction price for {symbol} on {transaction_date}")
        return None

    shares = amount / transaction_price

    # Assume we hold the position for the specified holding period or until the last available date
    start_date = datetime.strptime(transaction_date, '%Y-%m-%d')
    end_date = start_date + timedelta(days=holding_period_days)
    end_date_str = end_date.strftime('%Y-%m-%d')

    # Find the closest available date on or after the end date
    if end_date_str not in available_dates:
        future_dates = [date for date in available_dates if datetime.strptime(date, '%Y-%m-%d') >= end_date]
        if not future_dates:
            end_date_str = available_dates[-1]  # Use the latest available date
        else:
            end_date_str = future_dates[0]

    # Get closing price at the end of the holding period
    end_price = price_data.loc[end_date_str]['Close']
    if end_price is None or end_price == 0 or math.isnan(end_price):
        logging.warning(f"Invalid end price for {symbol} on {end_date_str}")
        return None

    # Calculate profit based on trade type
    profit = 0.0
    # Define buy and sell types
    buy_types = ['purchase', 'buy', 'receive', 'exchange']
    sell_types = ['sale', 'sell']

    if any(buy_type in trade_type for buy_type in buy_types):
        # For buy trades, profit is calculated as (End Price - Transaction Price) * Shares
        profit = (end_price - transaction_price) * shares
    elif any(sell_type in trade_type for sell_type in sell_types):
        # For sell trades, profit is calculated as (Transaction Price - End Price) * Shares
        profit = (transaction_price - end_price) * shares
    else:
        logging.warning(f"Unknown trade type: {trade_type}")
        return None

    return {
        'profit': profit,
        'holding_period': (end_date - start_date).days,
        'return_percentage': (profit / amount) * 100 if amount != 0 else 0
    }

def calculate_trader_skill(trades, holding_period_days):
    """
    Calculates the skill metrics for each trader based on their trades.

    Args:
        trades: A list of trade dictionaries.
        holding_period_days: The holding period in days for backtesting.

    Returns:
        A dictionary mapping traders to their skill metrics.
    """
    trader_performance = defaultdict(list)

    # Fetch price data for all symbols involved
    symbols = set(trade['symbol'] for trade in trades)
    for symbol in tqdm(symbols, desc="Fetching price data"):
        get_price_data(symbol)

    # Compute individual trade performances
    for trade in tqdm(trades, desc="Computing trade performance"):
        symbol = trade['symbol']
        price_data = price_data_cache.get(symbol)
        if price_data is None:
            continue

        performance = compute_trade_performance(trade, price_data, holding_period_days)
        if performance:
            trader_performance[trade['trader']].append(performance)

    # Calculate skill metrics for each trader
    trader_skills = {}
    MIN_TRADES = 10  # Set your desired minimum number of trades

    for trader, performances in trader_performance.items():
        trade_count = len(performances)
        if trade_count < MIN_TRADES:
            continue  # Skip traders with fewer than MIN_TRADES

        profits = [p['profit'] for p in performances]
        returns = [p['return_percentage'] for p in performances]
        holding_periods = [p['holding_period'] for p in performances]

        total_profit = sum(profits)
        average_return = np.mean(returns)
        average_holding_period = np.mean(holding_periods)
        trade_breadth = len(set(trade['symbol'] for trade in trades if trade['trader'] == trader))

        # Calculate standard deviation of returns
        if len(returns) > 1:
            return_std = np.std(returns, ddof=1)
            if return_std != 0:
                sharpe_ratio = average_return / return_std
            else:
                sharpe_ratio = 0.0
        else:
            return_std = 0.0  # Not enough data to compute std deviation
            sharpe_ratio = 0.0

        # Skill score combining multiple factors
        skill_score = (
            (average_return * total_profit * math.log1p(trade_count) * math.log1p(trade_breadth) * sharpe_ratio)
            / (1 + return_std)
        )

        trader_skills[trader] = {
            'Total Profit ($)': total_profit,
            'Average Return (%)': average_return,
            'Return Std Dev (%)': return_std,
            'Sharpe Ratio': sharpe_ratio,
            'Average Holding Period (days)': average_holding_period,
            'Number of Trades': trade_count,
            'Trade Breadth': trade_breadth,
            'Skill Score': skill_score
        }

    return trader_skills

def save_results(trader_skills, outfile):
    """
    Saves the trader skills data to a CSV file.

    Args:
        trader_skills: A dictionary mapping traders to their skill metrics.
        outfile: The path to the output CSV file.
    """
    df = pd.DataFrame.from_dict(trader_skills, orient='index')
    df.index.name = 'Trader'
    df.sort_values(by='Skill Score', ascending=False, inplace=True)
    df.to_csv(outfile)
    logging.info(f"Results saved to {outfile}")

async def main():
    """
    Main function to parse arguments, fetch data, compute skills, and save results.
    """
    parser = argparse.ArgumentParser(description='Analyze government trades and rank traders by skill.')
    parser.add_argument('--symbols', required=True, help='Path to symbols file (one symbol per line)')
    parser.add_argument('--outfile', required=True, help='Output CSV file path')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--holding-period', type=int, default=30, help='Holding period in days for backtesting')
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Read symbols from file
    if not os.path.exists(args.symbols):
        logging.error(f"Symbols file not found: {args.symbols}")
        sys.exit(1)

    with open(args.symbols, 'r') as f:
        symbols = [line.strip().upper() for line in f if line.strip()]

    if not symbols:
        logging.error("No symbols provided.")
        sys.exit(1)

    # Initialize limiter for rate limiting
    limiter = AsyncLimiter(MAX_REQUESTS_PER_MINUTE, time_period=60)  # Rate limiter

    # Fetch trades
    trades = await fetch_trades(symbols, limiter)

    if not trades:
        logging.error("No trades found.")
        sys.exit(1)

    # Calculate trader skills
    trader_skills = calculate_trader_skill(trades, args.holding_period)

    if not trader_skills:
        logging.error("No trader skills calculated.")
        sys.exit(1)

    # Save results
    save_results(trader_skills, args.outfile)

if __name__ == "__main__":
    asyncio.run(main())
