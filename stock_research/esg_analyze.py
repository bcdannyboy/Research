import os
import pandas as pd
import numpy as np
import datetime
from dotenv import load_dotenv
from tqdm import tqdm
import asyncio
import aiohttp
import time
import plotly.express as px
from scipy.stats import kendalltau, spearmanr
from statsmodels.distributions.copula.api import (
    ClaytonCopula, FrankCopula, GumbelCopula,
    GaussianCopula, StudentTCopula
)
from statsmodels.distributions.empirical_distribution import ECDF

# Load API key from .env file
load_dotenv()
FMP_API_KEY = os.getenv('FMP_API_KEY')

# Check if API key is loaded
if not FMP_API_KEY:
    raise ValueError("FMP_API_KEY not found in .env file.")

# 1. Load the list of symbols from symbols.txt
with open('symbols.txt', 'r') as file:
    symbols = [line.strip() for line in file.readlines()]

# Remove any empty symbols
symbols = [symbol for symbol in symbols if symbol]

# Optimized RateLimiter class for FMP API
class RateLimiter:
    def __init__(self, max_calls, period):
        self._max_calls = max_calls
        self._period = period
        self._tokens = max_calls
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            self._last = now
            self._tokens += elapsed * (self._max_calls / self._period)
            if self._tokens > self._max_calls:
                self._tokens = self._max_calls
            if self._tokens < 1:
                await asyncio.sleep((1 - self._tokens) * (self._period / self._max_calls))
                self._tokens = 0
            else:
                self._tokens -= 1

# 2. Fetch ESG data asynchronously
async def fetch_esg_data(session, symbol, rate_limiter):
    await rate_limiter.acquire()
    url = f'https://financialmodelingprep.com/api/v4/esg-environmental-social-governance-data?symbol={symbol}&apikey={FMP_API_KEY}'
    try:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                if data:
                    return 'esg', symbol, data[0]
                else:
                    return 'esg', symbol, None
            else:
                return 'esg', symbol, None
    except Exception as e:
        return 'esg', symbol, None

# 3. Fetch historical stock prices from FMP API asynchronously
async def fetch_stock_price_fmp(session, symbol, start_date, end_date, rate_limiter):
    await rate_limiter.acquire()
    url = f'https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?from={start_date}&to={end_date}&apikey={FMP_API_KEY}'
    try:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                if 'historical' in data and data['historical']:
                    df = pd.DataFrame(data['historical'])
                    df['date'] = pd.to_datetime(df['date'])
                    df = df[['date', 'close']].sort_values('date')
                    df['symbol'] = symbol  # Add symbol column for later merging
                    return 'price', symbol, df
                else:
                    return 'price', symbol, None
            else:
                return 'price', symbol, None
    except Exception as e:
        return 'price', symbol, None

# 4. Compute cumulative return
def compute_cumulative_return(df):
    df = df.sort_values('date')
    initial_price = df['close'].iloc[0]
    final_price = df['close'].iloc[-1]
    cumulative_return = (final_price - initial_price) / initial_price
    return cumulative_return

# Main function to run the async tasks with optimizations
async def main():
    # Rate limiter allows 10 requests per second (adjust based on API limits)
    rate_limiter = RateLimiter(max_calls=10, period=1)

    esg_data_list = []
    stock_prices = {}

    # Define date range (e.g., past one year)
    end_date = datetime.date.today().isoformat()
    start_date = (datetime.date.today() - datetime.timedelta(days=365)).isoformat()

    async with aiohttp.ClientSession() as session:
        tasks = []
        for symbol in symbols:
            esg_task = asyncio.create_task(fetch_esg_data(session, symbol, rate_limiter))
            tasks.append(esg_task)

            price_task = asyncio.create_task(fetch_stock_price_fmp(session, symbol, start_date, end_date, rate_limiter))
            tasks.append(price_task)

        print("\nFetching ESG data and stock prices:")
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Data Fetching", unit="task"):
            task_type, symbol, data = await f
            if task_type == 'esg':
                if data:
                    esg_data_list.append(data)
                else:
                    print(f"ESG data not found for symbol: {symbol}")
            elif task_type == 'price':
                if data is not None:
                    stock_prices[symbol] = data
                else:
                    print(f"Stock price data not found or empty for symbol: {symbol}")

        if not esg_data_list:
            raise ValueError("No ESG data was fetched. Please check the symbols and API responses.")
        if not stock_prices:
            raise ValueError("No stock price data was fetched. Please check the symbols and API responses.")

    # Compute cumulative returns
    cumulative_returns = []
    for symbol, df in stock_prices.items():
        cum_return = compute_cumulative_return(df)
        cumulative_returns.append({'symbol': symbol, 'cumulative_return': cum_return})

    cumulative_returns_df = pd.DataFrame(cumulative_returns)

    # Prepare data for correlation analysis
    esg_scores = pd.DataFrame(esg_data_list)
    esg_scores = esg_scores[['symbol', 'environmentalScore', 'socialScore', 'governanceScore', 'ESGScore']]

    # Merge ESG scores with cumulative returns
    merged_data = pd.merge(esg_scores, cumulative_returns_df, on='symbol')
    merged_data.dropna(inplace=True)  # Remove rows with missing data

    # Check if merged_data is empty
    if merged_data.empty:
        raise ValueError("No data available after merging ESG scores with cumulative returns.")

    # Analyze dependence using copulas
    esg_score_columns = ['environmentalScore', 'socialScore', 'governanceScore', 'ESGScore']

    # Create a DataFrame to store dependence measures and best copulas
    dependence_results = []

    for score in esg_score_columns:
        print(f"\nAnalyzing dependence between {score} and cumulative_return using copulas.")

        data = merged_data[[score, 'cumulative_return']].dropna()

        u = data[score].values
        v = data['cumulative_return'].values

        # Check for constant values in data
        if np.all(u == u[0]) or np.all(v == v[0]):
            print(f"Data for {score} or cumulative_return is constant. Skipping...")
            continue

        # Convert data to uniform margins using rank transformation
        u_ranks = (pd.Series(u).rank(method='average').values - 0.5) / len(u)
        v_ranks = (pd.Series(v).rank(method='average').values - 0.5) / len(v)

        # Ensure the data is in (0,1)
        u_uniform = u_ranks
        v_uniform = v_ranks

        # Stack the data
        data_uniform = np.column_stack([u_uniform, v_uniform])

        # Compute empirical Kendall's tau
        tau_empirical, _ = kendalltau(u, v)

        # Define copulas and estimate parameters from Kendall's tau
        copula_models = {}

        # GaussianCopula
        rho_gaussian, _ = spearmanr(u, v)
        corr_matrix = np.array([[1, rho_gaussian],
                                [rho_gaussian, 1]])
        copula_gaussian = GaussianCopula()
        copula_gaussian.corr = corr_matrix
        copula_models['GaussianCopula'] = copula_gaussian

        # StudentTCopula
        copula_student_t = StudentTCopula(df=4)
        copula_student_t.corr = corr_matrix
        copula_models['StudentTCopula'] = copula_student_t

        # ClaytonCopula
        if 0 < tau_empirical < 1:
            theta_clayton = 2 * tau_empirical / (1 - tau_empirical)
            if theta_clayton > 0:
                copula_clayton = ClaytonCopula(theta=theta_clayton)
                copula_models['ClaytonCopula'] = copula_clayton
            else:
                print(f"Invalid theta for Clayton Copula: {theta_clayton}")
        else:
            print(f"Invalid Kendall's tau for Clayton Copula: {tau_empirical}")

        # GumbelCopula
        if 0 < tau_empirical < 1:
            theta_gumbel = 1 / (1 - tau_empirical)
            if theta_gumbel >= 1:
                copula_gumbel = GumbelCopula(theta=theta_gumbel)
                copula_models['GumbelCopula'] = copula_gumbel
            else:
                print(f"Invalid theta for Gumbel Copula: {theta_gumbel}")
        else:
            print(f"Invalid Kendall's tau for Gumbel Copula: {tau_empirical}")

        # FrankCopula
        try:
            copula_frank = FrankCopula()
            theta_frank = copula_frank.params_from_tau(tau_empirical)
            copula_frank.theta = theta_frank
            copula_models['FrankCopula'] = copula_frank
        except Exception as e:
            print(f"Failed to estimate parameters for FrankCopula: {e}")

        best_copula = None
        best_log_likelihood = -np.inf
        best_copula_name = None

        for name, copula in copula_models.items():
            try:
                # Compute log-likelihood
                log_likelihood = np.sum(copula.logpdf(data_uniform))
                print(f"{name} log-likelihood: {log_likelihood:.4f}")
                if log_likelihood > best_log_likelihood:
                    best_log_likelihood = log_likelihood
                    best_copula = copula
                    best_copula_name = name
            except Exception as e:
                print(f"Failed to compute log-likelihood for {name}: {e}")

        if best_copula is None:
            print(f"No copula could be fitted for {score}. Skipping...")
            continue

        print(f"Best copula: {best_copula_name} with log-likelihood {best_log_likelihood:.4f}")

        # Use the best copula to sample data
        try:
            samples_uniform = best_copula.rvs(len(u_uniform), random_state=42)
        except Exception as e:
            print(f"Failed to generate samples using {best_copula_name}: {e}")
            continue

        # Convert uniform samples back to data using inverse rank transformation
        samples_u = np.quantile(u, samples_uniform[:, 0])
        samples_v = np.quantile(v, samples_uniform[:, 1])

        # Compute dependence measures
        original_kendall_tau, _ = kendalltau(u, v)
        synthetic_kendall_tau, _ = kendalltau(samples_u, samples_v)

        original_spearman_rho, _ = spearmanr(u, v)
        synthetic_spearman_rho, _ = spearmanr(samples_u, samples_v)

        print(f"Original Kendall's tau: {original_kendall_tau:.4f}")
        print(f"Synthetic Kendall's tau: {synthetic_kendall_tau:.4f}")

        print(f"Original Spearman's rho: {original_spearman_rho:.4f}")
        print(f"Synthetic Spearman's rho: {synthetic_spearman_rho:.4f}")

        # Save the results
        dependence_results.append({
            'score': score,
            'best_copula': best_copula_name,
            'log_likelihood': best_log_likelihood,
            'original_kendall_tau': original_kendall_tau,
            'synthetic_kendall_tau': synthetic_kendall_tau,
            'original_spearman_rho': original_spearman_rho,
            'synthetic_spearman_rho': synthetic_spearman_rho
        })

        # Visualize original data vs synthetic data
        fig = px.scatter(
            x=u, y=v,
            title=f'Original Data: {score} vs. Cumulative Return',
            labels={'x': score, 'y': 'Cumulative Return'}
        )
        fig.show()

        fig = px.scatter(
            x=samples_u, y=samples_v,
            title=f'Synthetic Data ({best_copula_name}): {score} vs. Cumulative Return',
            labels={'x': score, 'y': 'Cumulative Return'}
        )
        fig.show()

    # Convert dependence results to DataFrame
    dependence_df = pd.DataFrame(dependence_results)
    print("\nDependence analysis results:")
    print(dependence_df)

    # Save dependence results to CSV
    dependence_df.to_csv('esg_dependence_results.csv', index=False)
    print("\nDependence results saved to 'esg_dependence_results.csv'.")

    # Save merged data to CSV for further analysis
    merged_data.to_csv('esg_returns_data.csv', index=False)
    print("\nMerged data saved to 'esg_returns_data.csv'.")

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
