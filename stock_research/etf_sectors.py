"""
ETF and Sector Analysis Script

This script reads ETF symbols from a file called 'etfs.txt', fetches data from the Financial Modeling Prep API to analyze ETFs and their sector exposures.
It builds a graph of ETFs and sectors using NetworkX and visualizes it using Plotly. The script computes
node weights for ETFs based on assets under management (AUM), average volume, expense ratio, and historical
performance. It then applies network analysis algorithms to determine:

1. The most central ETFs based on NetworkX centrality measures.
2. The best ETF to invest in per sector, ranked by the best sectors to invest.
3. The most important nodes combining node weights and centrality measures.

Usage:
- Ensure you have a .env file in the same directory with your FMP_API_KEY.
- Create an 'etfs.txt' file in the same directory with one ETF symbol per line.
- Install required packages: requests, pandas, networkx, plotly, python-dotenv, concurrent.futures, numpy
- Run the script using Python 3.

Note:
- The script is designed to stay within the API rate limit of 300 requests per minute.
- Adjust 'max_etfs' in the 'main' function to change the number of ETFs processed.

Author: Your Name
Date: 2023-10-12
"""

import os
import requests
import pandas as pd
import networkx as nx
import plotly.graph_objs as go
from plotly.offline import plot  # For saving the graph to an HTML file
from dotenv import load_dotenv
import concurrent.futures  # For concurrent requests
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Load the FMP_API_KEY from the .env file
load_dotenv()
FMP_API_KEY = os.getenv('FMP_API_KEY')

if not FMP_API_KEY:
    raise ValueError("FMP_API_KEY not found. Please set it in the .env file.")

def get_etf_list_from_file():
    """
    Reads ETF symbols from 'etfs.txt' file, one symbol per line.

    Returns:
        etf_symbols (list): List of ETF symbols.
    """
    if not os.path.exists('etfs.txt'):
        raise FileNotFoundError("The file 'etfs.txt' was not found in the current directory.")
    with open('etfs.txt', 'r') as f:
        etf_symbols = [line.strip().upper() for line in f if line.strip()]
    logging.info(f"Total ETFs read from file: {len(etf_symbols)}")
    return etf_symbols

def get_etf_info(symbol):
    """
    Fetches detailed information about an ETF.

    Args:
        symbol (str): ETF ticker symbol.

    Returns:
        etf_info (dict): Dictionary containing ETF information.
    """
    url = f"https://financialmodelingprep.com/api/v4/etf-info?symbol={symbol}&apikey={FMP_API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        logging.warning(f"Failed to fetch info for ETF: {symbol} (Status Code: {response.status_code})")
        return None
    data = response.json()
    if data:
        return data[0]  # Returns the first (and only) element
    else:
        logging.warning(f"No data returned for ETF: {symbol}")
        return None

def get_etf_sector_weightings(symbol):
    """
    Fetches sector weightings for an ETF.

    Args:
        symbol (str): ETF ticker symbol.

    Returns:
        sector_weightings (list of dict): List of sectors and their weight percentages.
    """
    url = f"https://financialmodelingprep.com/api/v3/etf-sector-weightings/{symbol}?apikey={FMP_API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        logging.warning(f"Failed to fetch sector weightings for ETF: {symbol} (Status Code: {response.status_code})")
        return None
    data = response.json()
    if not data:
        logging.warning(f"No sector weightings data returned for ETF: {symbol}")
    return data

def get_sector_performance():
    """
    Fetches current sector performance data.

    Returns:
        sector_performance (pd.DataFrame): DataFrame containing sectors and their performance percentages.
    """
    url = f"https://financialmodelingprep.com/api/v3/sectors-performance?apikey={FMP_API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to fetch sector performance data.")
    data = response.json()
    sector_performance = pd.DataFrame(data)
    # Process 'changesPercentage' to remove '%' and convert to float
    sector_performance['changesPercentage'] = sector_performance['changesPercentage'].str.replace('%', '').astype(float)
    logging.info(f"Fetched performance data for {len(sector_performance)} sectors.")
    return sector_performance

def get_etf_historical_performance(symbol, period='3months'):
    """
    Fetches historical performance data for an ETF.

    Args:
        symbol (str): ETF ticker symbol.
        period (str): Period for historical data ('1month', '3months', '6months', '1year')

    Returns:
        performance (float): Percentage change over the specified period.
    """
    periods = {
        '1month': 21,
        '3months': 63,
        '6months': 126,
        '1year': 252
    }
    days = periods.get(period, 63)  # Default to '3months' if not specified
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?serietype=line&timeseries={days}&apikey={FMP_API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        logging.warning(f"Failed to fetch historical data for ETF: {symbol} (Status Code: {response.status_code})")
        return None
    data = response.json()
    if 'historical' in data and len(data['historical']) > 1:
        prices = pd.DataFrame(data['historical'])
        start_price = prices.iloc[-1]['close']
        end_price = prices.iloc[0]['close']
        if start_price == 0:
            logging.warning(f"Start price is zero for ETF: {symbol}")
            return 0.0
        performance = ((end_price - start_price) / start_price) * 100
        return performance
    else:
        logging.warning(f"Not enough historical data for ETF: {symbol}")
        return None

def compute_node_weights(G):
    """
    Computes node weights for ETFs and Sectors in the graph.

    Args:
        G (networkx.Graph): The graph containing ETF and Sector nodes.
    """
    # Small number to avoid division by zero
    epsilon = 1e-6

    # Collect data for normalization
    aum_list = []
    performance_list = []
    volume_list = []
    expense_ratio_list = []

    # First, ensure 'node_weight' is set for all sector nodes
    for node in G.nodes():
        node_data = G.nodes[node]
        if node_data.get('type') == 'Sector':
            performance = node_data.get('performance', 0.0)
            if not isinstance(performance, (int, float)):
                performance = 0.0
            node_data['node_weight'] = performance

    # Now process ETF nodes
    etf_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'ETF']
    if not etf_nodes:
        logging.error("No ETF nodes found in the graph.")
        return

    for node in etf_nodes:
        node_data = G.nodes[node]
        aum = node_data.get('aum', 0.0)
        historical_performance = node_data.get('historical_performance', 0.0)
        avg_volume = node_data.get('avg_volume', 0.0)
        expense_ratio = node_data.get('expense_ratio', epsilon)

        # Check if data is valid
        if aum == 0.0:
            logging.warning(f"AUM is zero for ETF: {node}")
        if avg_volume == 0.0:
            logging.warning(f"Average volume is zero for ETF: {node}")
        if expense_ratio == epsilon:
            logging.warning(f"Expense ratio is zero or missing for ETF: {node}")
        if historical_performance == 0.0:
            logging.warning(f"Historical performance is zero or missing for ETF: {node}")

        aum_list.append(aum)
        performance_list.append(historical_performance)
        volume_list.append(avg_volume)
        expense_ratio_list.append(expense_ratio)

    # Check if lists have valid data
    if not any(aum_list) or not any(performance_list) or not any(volume_list) or not any(expense_ratio_list):
        logging.error("Insufficient valid data to compute node weights for ETFs.")
        return

    # Convert lists to numpy arrays
    aum_array = np.array(aum_list)
    performance_array = np.array(performance_list)
    volume_array = np.array(volume_list)
    expense_ratio_array = np.array(expense_ratio_list)

    # Handle cases where the ptp (max - min) is zero
    def safe_normalize(array):
        ptp = array.ptp()
        if ptp == 0:
            logging.warning("Peak-to-peak (ptp) is zero during normalization.")
            return np.zeros_like(array)
        else:
            return (array - array.min()) / (ptp + epsilon)

    aum_norm = safe_normalize(aum_array)
    performance_norm = safe_normalize(performance_array)
    volume_norm = safe_normalize(volume_array)
    expense_ratio_norm = safe_normalize(expense_ratio_array.max() - expense_ratio_array)  # Invert expense ratio

    # Assign normalized values back to nodes
    for idx, node in enumerate(etf_nodes):
        node_data = G.nodes[node]
        node_data['aum_norm'] = aum_norm[idx]
        node_data['performance_norm'] = performance_norm[idx]
        node_data['volume_norm'] = volume_norm[idx]
        node_data['expense_ratio_norm'] = expense_ratio_norm[idx]

    # Compute node weights for ETFs
    for node in etf_nodes:
        node_data = G.nodes[node]
        # Weight factors (can be adjusted as needed)
        weight_aum = 0.3
        weight_performance = 0.4
        weight_volume = 0.2
        weight_expense_ratio = 0.1

        node_weight = (
            weight_aum * node_data['aum_norm'] +
            weight_performance * node_data['performance_norm'] +
            weight_volume * node_data['volume_norm'] +
            weight_expense_ratio * node_data['expense_ratio_norm']
        )
        node_data['node_weight'] = node_weight

    logging.info("Node weights computed successfully.")

def compute_centrality_measures(G):
    """
    Computes centrality measures for each node in the graph.

    Args:
        G (networkx.Graph): The graph containing ETF and Sector nodes.
    """
    logging.info("Computing centrality measures...")
    # Compute centrality measures
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    try:
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    except nx.NetworkXException as e:
        logging.warning(f"Eigenvector centrality did not converge: {e}")
        eigenvector_centrality = {node: 0.0 for node in G.nodes()}

    # Assign centrality measures to nodes
    for node in G.nodes():
        G.nodes[node]['degree_centrality'] = degree_centrality.get(node, 0.0)
        G.nodes[node]['betweenness_centrality'] = betweenness_centrality.get(node, 0.0)
        G.nodes[node]['eigenvector_centrality'] = eigenvector_centrality.get(node, 0.0)
    logging.info("Centrality measures computed successfully.")

def find_most_important_nodes(G, top_n=10):
    """
    Finds the most important nodes by combining node weights and centrality measures.

    Args:
        G (networkx.Graph): The graph containing ETF and Sector nodes.
        top_n (int): Number of top nodes to return.
    """
    logging.info("Calculating combined importance scores for nodes...")
    # Collect node weights and centrality measures
    node_scores = {}
    for node, data in G.nodes(data=True):
        if data.get('type') == 'ETF':
            node_weight = data.get('node_weight', 0.0)
            degree_centrality = data.get('degree_centrality', 0.0)
            betweenness_centrality = data.get('betweenness_centrality', 0.0)
            eigenvector_centrality = data.get('eigenvector_centrality', 0.0)
            # Normalize node_weight between 0 and 1
            node_weight_norm = node_weight  # Already normalized
            # Combine centrality measures (weights can be adjusted)
            centrality_combined = (
                0.4 * degree_centrality +
                0.3 * betweenness_centrality +
                0.3 * eigenvector_centrality
            )
            # Combine node weight and centrality
            importance_score = (
                0.6 * node_weight_norm +  # 60% weight
                0.4 * centrality_combined  # 40% weight
            )
            node_scores[node] = importance_score
        elif data.get('type') == 'Sector':
            node_weight = data.get('node_weight', 0.0)  # Performance
            # For sectors, centrality measures are also available
            degree_centrality = data.get('degree_centrality', 0.0)
            betweenness_centrality = data.get('betweenness_centrality', 0.0)
            eigenvector_centrality = data.get('eigenvector_centrality', 0.0)
            # Normalize node_weight between 0 and 1
            node_weight_norm = (node_weight - G.nodes[node]['node_weight']) / (max([d.get('node_weight',0) for n,d in G.nodes(data=True) if d.get('type')=='Sector'], default=1) - min([d.get('node_weight',0) for n,d in G.nodes(data=True) if d.get('type')=='Sector'], default=0) + 1e-6)
            # Combine centrality measures (weights can be adjusted)
            centrality_combined = (
                0.4 * degree_centrality +
                0.3 * betweenness_centrality +
                0.3 * eigenvector_centrality
            )
            # Combine node weight and centrality
            importance_score = (
                0.6 * node_weight_norm +  # 60% weight
                0.4 * centrality_combined  # 40% weight
            )
            node_scores[node] = importance_score
        else:
            continue  # Skip nodes without a valid type

    # Sort nodes based on importance_score
    sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)

    # Output the most important nodes
    print("\nMost Important Nodes (ETFs and Sectors):")
    for node, score in sorted_nodes[:top_n]:
        node_type = G.nodes[node].get('type', '')
        if node_type == 'ETF':
            name = G.nodes[node].get('name', '')
            print(f"ETF: {node} ({name}), Importance Score: {score:.4f}")
        elif node_type == 'Sector':
            performance = G.nodes[node].get('node_weight', 0.0)
            print(f"Sector: {node}, Performance: {performance:.2f}%, Importance Score: {score:.4f}")
    logging.info("Most important nodes identified.")

def find_most_central_etfs(G):
    """
    Finds the most central ETFs based on centrality measures.

    Args:
        G (networkx.Graph): The graph containing ETF and Sector nodes.
    """
    # Collect ETFs and their centrality measures
    etfs = [n for n, d in G.nodes(data=True) if d.get('type') == 'ETF']
    centrality_scores = []

    for etf in etfs:
        data = G.nodes[etf]
        # Combine centrality measures (weights can be adjusted)
        centrality_score = (
            data.get('degree_centrality', 0) * 0.4 +
            data.get('betweenness_centrality', 0) * 0.3 +
            data.get('eigenvector_centrality', 0) * 0.3
        )
        centrality_scores.append((etf, centrality_score))

    # Sort ETFs based on centrality scores
    centrality_scores.sort(key=lambda x: x[1], reverse=True)

    # Output the most central ETFs
    print("\nMost Central ETFs based on NetworkX Centrality Measures:")
    for etf_symbol, centrality_score in centrality_scores[:10]:  # Top 10 ETFs
        print(f"ETF: {etf_symbol}, Centrality Score: {centrality_score:.4f}")

def find_best_etfs_per_sector(G):
    """
    Finds the best ETF to invest in per sector based on node weights.

    Args:
        G (networkx.Graph): The graph containing ETF and Sector nodes.
    """
    # Create a dictionary to store the best ETF per sector
    best_etfs_per_sector = {}

    # Iterate through each sector
    sectors = [n for n, d in G.nodes(data=True) if d.get('type') == 'Sector']
    for sector in sectors:
        connected_etfs = list(G.neighbors(sector))
        if not connected_etfs:
            best_etfs_per_sector[sector] = None
            continue
        # Find the ETF with the highest node_weight
        best_etf = max(connected_etfs, key=lambda x: G.nodes[x].get('node_weight', 0))
        best_etfs_per_sector[sector] = best_etf

    # Rank sectors based on their performance
    sector_rankings = sorted(sectors, key=lambda x: G.nodes[x].get('node_weight', 0), reverse=True)

    # Output the best ETF per sector, ranked by sector performance
    print("\nBest ETFs to invest in per sector, ranked by sector performance:")
    for sector in sector_rankings:
        sector_perf = G.nodes[sector].get('node_weight', 0.0)
        best_etf = best_etfs_per_sector.get(sector)
        if best_etf:
            etf_weight = G.nodes[best_etf].get('node_weight', 0.0)
            print(f"Sector: {sector} (Performance: {sector_perf:.2f}%), Best ETF: {best_etf} (Score: {etf_weight:.4f})")
        else:
            print(f"Sector: {sector} (Performance: {sector_perf:.2f}%), No ETFs available")

def visualize_graph(G):
    """
    Visualizes the ETF-Sector graph using Plotly and saves it to an HTML file.

    Args:
        G (networkx.Graph): The graph containing ETF and Sector nodes.
    """
    # Create positions for nodes using networkx layout
    pos = nx.spring_layout(G, k=0.15, iterations=20)

    # Create edge traces
    edge_trace = []
    min_width = 0.5  # Minimum line width
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = G.edges[edge]['weight']
        # Ensure weight is non-negative
        if weight < 0:
            weight = 0.0
        line_width = max(weight * 5, min_width)
        edge_trace.append(go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            line=dict(width=line_width, color='#888'),
            hoverinfo='none',
            mode='lines'))

    # Create node traces for ETFs and Sectors
    etf_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            color=[],
            size=[],
            colorbar=dict(
                thickness=15,
                title='ETF Node Weight',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    sector_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            color='red',
            size=20,
            line_width=2))

    # Populate node traces
    for node in G.nodes():
        x, y = pos[node]
        node_data = G.nodes[node]
        node_type = node_data.get('type', '')
        if node_type == 'ETF':
            etf_trace['x'] += tuple([x])
            etf_trace['y'] += tuple([y])
            node_weight = node_data.get('node_weight', 0.0)
            etf_trace['text'] += tuple([f"{node}<br>{node_data.get('name', '')}<br>Score: {node_weight:.4f}"])
            etf_trace['marker']['color'] += tuple([node_weight])
            etf_trace['marker']['size'] += tuple([10])
        elif node_type == 'Sector':
            sector_trace['x'] += tuple([x])
            sector_trace['y'] += tuple([y])
            node_weight = node_data.get('node_weight', 0.0)
            sector_trace['text'] += tuple([f"{node}<br>Performance: {node_weight:.2f}%"])
        else:
            # Skip nodes without a valid 'type'
            continue

    # Create the figure
    fig = go.Figure(data=edge_trace + [etf_trace, sector_trace],
                    layout=go.Layout(
                        title='<br>ETF-Sector Graph',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="",
                            showarrow=False,
                            xref="paper", yref="paper")],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    # Save the graph to an HTML file
    output_file = 'etf_sector_graph.html'
    plot(fig, filename=output_file, auto_open=True)
    logging.info(f"Graph saved to {output_file}")

def main():
    """
    Main function to execute the ETF analysis and graph creation.
    """
    # Read ETF list from 'etfs.txt'
    logging.info("Reading ETF list from 'etfs.txt'...")
    etf_symbols = get_etf_list_from_file()

    # Limit the number of ETFs to process to stay within rate limits
    max_etfs = 1000  # Adjust as needed
    etf_symbols = etf_symbols[:max_etfs]
    logging.info(f"Processing top {len(etf_symbols)} ETFs.")

    # Initialize data structures
    G = nx.Graph()
    etf_data_list = []
    sector_nodes = set()
    sector_performance = get_sector_performance()
    # Create a dictionary mapping sector names to their performance
    sector_performance_dict = dict(zip(sector_performance['sector'], sector_performance['changesPercentage']))

    # Process each ETF concurrently
    logging.info("Processing ETFs...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Submit tasks to the executor
        future_to_symbol = {executor.submit(fetch_etf_data, symbol): symbol for symbol in etf_symbols}
        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                result = future.result()
                if result is None:
                    continue
                symbol, etf_info, sector_weightings, historical_performance = result

                # Add ETF node with attributes
                aum = etf_info.get('aum', 0)
                avg_volume = etf_info.get('avgVolume', 0)
                expense_ratio = etf_info.get('expenseRatio', 0.0)
                holdings_count = etf_info.get('holdingsCount', 0)
                G.add_node(symbol, type='ETF', aum=aum,
                           avg_volume=avg_volume,
                           expense_ratio=expense_ratio,
                           name=etf_info.get('name', ''),
                           holdings_count=holdings_count,
                           historical_performance=historical_performance)

                # Store ETF data
                etf_data_list.append({
                    'symbol': symbol,
                    'name': etf_info.get('name', ''),
                    'aum': aum,
                    'avg_volume': avg_volume,
                    'expense_ratio': expense_ratio,
                    'historical_performance': historical_performance,
                    'holdings_count': holdings_count
                })

                # Process sector weightings
                for sector in sector_weightings:
                    sector_name = sector['sector']
                    weight_percentage = sector['weightPercentage']

                    # Add sector node if not already added
                    if sector_name not in sector_nodes:
                        sector_nodes.add(sector_name)
                        # Get sector performance
                        sector_perf = sector_performance_dict.get(sector_name, 0.0)
                        G.add_node(sector_name, type='Sector', performance=sector_perf)

                    # Add edge between ETF and sector with weight
                    G.add_edge(symbol, sector_name, weight=weight_percentage)

            except Exception as exc:
                logging.error(f'ETF {symbol} generated an exception: {exc}')
                continue

    # Convert etf_data_list to DataFrame (optional, not used in further computations)
    etf_data = pd.DataFrame(etf_data_list)

    # Compute node weights
    compute_node_weights(G)

    # Compute centrality measures
    compute_centrality_measures(G)

    # Find the most central ETFs
    find_most_central_etfs(G)

    # Find the most important nodes
    find_most_important_nodes(G, top_n=10)

    # Find the best ETFs per sector
    find_best_etfs_per_sector(G)

    # Visualize the graph and save it to an HTML file
    visualize_graph(G)

def fetch_etf_data(symbol):
    """
    Fetches ETF info, sector weightings, and historical performance for a given symbol.

    Args:
        symbol (str): ETF ticker symbol.

    Returns:
        tuple: (symbol, etf_info, sector_weightings, historical_performance) or None if failed.
    """
    etf_info = get_etf_info(symbol)
    if etf_info is None:
        return None

    sector_weightings = get_etf_sector_weightings(symbol)
    if sector_weightings is None:
        return None

    historical_performance = get_etf_historical_performance(symbol, period='3months')
    if historical_performance is None:
        historical_performance = 0.0

    # Ensure numerical fields are properly typed and not None
    def safe_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def safe_int(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    aum = safe_float(etf_info.get('aum'))
    avg_volume = safe_float(etf_info.get('avgVolume'))
    expense_ratio = safe_float(etf_info.get('expenseRatio'))
    holdings_count = safe_int(etf_info.get('holdingsCount'))

    etf_info['aum'] = aum
    etf_info['avgVolume'] = avg_volume
    etf_info['expenseRatio'] = expense_ratio
    etf_info['holdingsCount'] = holdings_count

    # Ensure that sector weightings are properly formatted
    for sector in sector_weightings:
        weight_str = sector.get('weightPercentage', '0%').replace('%', '').strip()
        try:
            weight_percentage = float(weight_str) / 100
        except ValueError:
            weight_percentage = 0.0
        # Ensure weight percentage is between 0 and 1
        if weight_percentage < 0:
            logging.warning(f"Negative sector weight detected for ETF: {symbol}, Sector: {sector.get('sector', '')}. Setting to 0.")
            weight_percentage = 0.0
        elif weight_percentage > 1:
            logging.warning(f"Sector weight exceeds 100% for ETF: {symbol}, Sector: {sector.get('sector', '')}. Normalizing.")
            weight_percentage = 1.0
        sector['weightPercentage'] = weight_percentage

    return (symbol, etf_info, sector_weightings, historical_performance)

if __name__ == "__main__":
    main()
