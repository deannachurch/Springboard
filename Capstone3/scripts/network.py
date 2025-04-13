import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
import matplotlib.cm as cm
import seaborn as sns
import networkx as nx
from typing import List, Dict, Tuple, Union, Any, Optional
import time
import re
import ast
from collections import defaultdict
from tqdm import tqdm

def clean_author_names(df: pd.DataFrame, column_names: Union[str, List[str]] = ['authors', 'first_author']) -> pd.DataFrame:
    """
    Cleans author names in one or more columns by converting full first names to initials.
    Handles both individual author names and lists of author names.
    
    Args:
        df: pandas DataFrame containing author columns
        column_names: name(s) of columns containing author information (str or list of str)
        
    Returns:
        DataFrame with cleaned author names
    """
    # Ensure column_names is a list
    if isinstance(column_names, str):
        column_names = [column_names]
    
    # Create a copy of the dataframe to avoid modifying the original
    cleaned_df = df.copy()
    
    def clean_author_name(author: str) -> str:
        """Clean a single author name by converting first/middle names to initials."""
        # Remove any extra quotes or spaces
        author = author.strip().replace("'", "").replace('"', '')
        
        # Check if already in initial format
        if re.match(r'^([A-Z]\.\s*)+\w+', author):
            return author
        
        # Split the name by spaces
        parts = author.split()
        
        if len(parts) == 1:
            # Single name, return as is
            return author
        
        # Get the last name
        last_name = parts[-1]
        
        # Convert all parts except the last name to initials
        initials = ''.join([part[0].upper() + '.' for part in parts[:-1]])
        
        # Combine initials with last name
        return initials + ' ' + last_name
    
    def process_value(value: Any) -> Any:
        """Process a value which could be a single author or a list/string representation of authors."""
        # If it's a string that looks like a list, convert it to an actual list
        if isinstance(value, str):
            if value.startswith('[') and value.endswith(']'):
                try:
                    # Try to safely evaluate the string as a list
                    value = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    # If we can't parse it as a list, treat as a single author
                    pass
        
        # Now process based on the type
        if isinstance(value, list):
            # It's a list of authors, clean each one
            return [clean_author_name(author) for author in value]
        elif isinstance(value, str):
            # It's a single author name
            return clean_author_name(value)
        else:
            # Return as is for any other type (None, etc.)
            return value
    
    # Process each specified column
    for column in column_names:
        if column in cleaned_df.columns:
            cleaned_df[column] = cleaned_df[column].apply(process_value)
    
    return cleaned_df

def clean_name_analysis(author: str) -> str:
    """
    Cleans author names by removing unwanted characters and normalizing format.
    
    Args:
        author: Raw author name string
        
    Returns:
        Cleaned author name
    """
    # Remove common unwanted characters
    for char in ["[", "]", "'", "\"", "{", "}"]:
        author = author.replace(char, "")
    
    # Trim whitespace
    author = author.strip()
    
    return author

def create_author_coauthorship_network(df: pd.DataFrame, author_column: str = 'authors', 
                                      delimiter: str = ',', show_progress: bool = False) -> nx.Graph:
    """
    Creates a co-authorship network graph from a DataFrame of research papers.

    Each node in the graph represents an author, and an edge between two nodes
    indicates that the two authors have co-authored at least one paper together.
    The weight of the edge represents the number of times they have co-authored.

    Args:
        df: DataFrame containing paper information
        author_column: Column name containing author information
        delimiter: Character used to separate authors if stored as strings
        show_progress: Whether to show a progress bar for large datasets

    Returns:
        A networkx Graph object representing the co-authorship network.
    """
    graph = nx.Graph()
    
    # Performance optimization: pre-build co-authorship relationships
    coauthorships = defaultdict(int)
    
    # Error handling for the author column
    if author_column not in df.columns:
        raise ValueError(f"Column '{author_column}' not found in DataFrame. Available columns: {list(df.columns)}")
    
    # Create an iterable with or without a progress bar
    rows_iter = tqdm(df.iterrows(), total=len(df), desc="Building network") if show_progress else df.iterrows()
    
    # First pass - collect all authors and their relationships
    for _, row in rows_iter:
        try:
            # Extract authors with error handling
            if pd.isna(row[author_column]).all() if isinstance(pd.isna(row[author_column]), np.ndarray) else pd.isna(row[author_column]):
                continue
            
            if isinstance(row[author_column], str):
                authors = [clean_name_analysis(author.strip()) for author in row[author_column].split(delimiter) if author.strip()]
            elif isinstance(row[author_column], list):
                authors = [clean_name_analysis(author.strip()) for author in row[author_column] if author and isinstance(author, str)]
            else:
                continue  # Skip if no author information or incorrect format
            
            # Skip papers with no valid authors
            if not authors:
                continue
                
            # This code block is now handled inside the if/else below
            
            # Add all authors as nodes, even for single-author papers
            for author in authors:
                graph.add_node(author)
                
            # Record co-authorships only if there are multiple authors
            if len(authors) > 1:
                for i in range(len(authors)):
                    for j in range(i + 1, len(authors)):
                        author1, author2 = authors[i], authors[j]
                        # Skip self-references (same author appearing twice)
                        if author1 == author2:
                            continue
                        # Skip if either author is empty
                        if not author1 or not author2:
                            continue
                        # Use frozenset to ensure consistent key regardless of author order
                        coauthorships[frozenset([author1, author2])] += 1
        
        except Exception as e:
            print(f"Error processing row: {e}")
    
    # Second pass - add all edges at once
    for authors, weight in coauthorships.items():
        author1, author2 = list(authors)
        graph.add_edge(author1, author2, weight=weight)
        
    return graph

def calculate_author_influence(graph: nx.Graph, 
                              metrics: List[str] = ['degree', 'betweenness', 'eigenvector', 'pagerank']) -> Dict[str, Dict[str, float]]:
    """
    Calculates the influence of each author in the co-authorship network using
    multiple centrality metrics.

    Args:
        graph: A networkx Graph object representing the co-authorship network.
        metrics: List of centrality metrics to calculate. Options are:
                'degree': Number of co-authors (connections)
                'betweenness': How often an author is on the shortest path between other authors
                'eigenvector': Connection to other influential authors
                'pagerank': Google-like ranking algorithm

    Returns:
        A nested dictionary with authors as keys and another dictionary of their 
        centrality scores for each requested metric as values.
    """
    # Error handling for empty graph
    if len(graph.nodes()) == 0:
        raise ValueError("Graph has no nodes. Cannot calculate centrality on an empty graph.")
    
    # Calculate requested centrality metrics
    results = {}
    metrics_lower = [m.lower() for m in metrics]
    
    # Initialize result dictionary for all authors
    results = {author: {} for author in graph.nodes()}
    
    try:
        # Calculate degree centrality
        if 'degree' in metrics_lower:
            degree_centrality = nx.degree_centrality(graph)
            for author, score in degree_centrality.items():
                results[author]['degree'] = score
        
        # Calculate betweenness centrality (can be slow for large networks)
        if 'betweenness' in metrics_lower:
            print("Calculating betweenness centrality (this may take a while for large networks)...")
            betweenness_centrality = nx.betweenness_centrality(graph, normalized=True)
            for author, score in betweenness_centrality.items():
                results[author]['betweenness'] = score
        
        # Calculate eigenvector centrality
        if 'eigenvector' in metrics_lower:
            try:
                eigenvector_centrality = nx.eigenvector_centrality(graph, max_iter=1000)
                for author, score in eigenvector_centrality.items():
                    results[author]['eigenvector'] = score
            except nx.PowerIterationFailedConvergence:
                print("Warning: Eigenvector centrality calculation did not converge. Results may be inaccurate.")
                # Try with a different algorithm that might be more stable
                eigenvector_centrality = nx.eigenvector_centrality_numpy(graph)
                for author, score in eigenvector_centrality.items():
                    results[author]['eigenvector'] = score
        
        # Calculate PageRank
        if 'pagerank' in metrics_lower:
            pagerank = nx.pagerank(graph)
            for author, score in pagerank.items():
                results[author]['pagerank'] = score
                
    except Exception as e:
        print(f"Error calculating centrality: {e}")
        raise
        
    return results

def visualize_coauthorship_network(
    graph: nx.Graph,
    top_n: int = 10,
    centrality_data: Dict[str, Dict[str, float]] = None,
    centrality_metric: str = 'degree',
    layout: str = 'spring',
    node_size_multiplier: int = 5000,
    edge_width_multiplier: float = 0.5,
    label_top_n_only: bool = False,
    min_edge_weight: int = 1,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10)
) -> None:
    """
    Visualizes the co-authorship network graph with enhanced features.

    Args:
        graph: A networkx Graph object representing the co-authorship network.
        top_n: The number of most influential authors to highlight with larger nodes.
        centrality_data: A nested dictionary of author centrality scores.
        centrality_metric: Which centrality metric to use for node sizing and coloring.
        layout: The layout algorithm to use ('spring', 'circular', 'kamada_kawai', etc.).
        node_size_multiplier: Factor to scale node sizes.
        edge_width_multiplier: Factor to scale edge widths based on co-authorship count.
        label_top_n_only: Whether to show labels for only the top N influential authors.
        min_edge_weight: Minimum weight for an edge to be displayed (to reduce clutter).
        save_path: Path to save the visualization (if None, it will be displayed).
        figsize: Figure size as a tuple of (width, height) in inches.
    """
    # Input validation and error handling
    if not isinstance(graph, nx.Graph):
        raise TypeError("Graph must be a networkx Graph object")
    
    if not graph.nodes():
        raise ValueError("Graph has no nodes to visualize")
    
    # Create a copy of the graph to filter edges by minimum weight
    if min_edge_weight > 1:
        filtered_graph = nx.Graph()
        filtered_graph.add_nodes_from(graph.nodes())
        for u, v, data in graph.edges(data=True):
            if data.get('weight', 1) >= min_edge_weight:
                filtered_graph.add_edge(u, v, **data)
        working_graph = filtered_graph
    else:
        working_graph = graph
    
    # Select layout algorithm
    layout_funcs = {
        'spring': nx.spring_layout,
        'circular': nx.circular_layout,
        'kamada_kawai': nx.kamada_kawai_layout,
        'random': nx.random_layout,
        'shell': nx.shell_layout,
        'spectral': nx.spectral_layout
    }
    
    if layout not in layout_funcs:
        print(f"Warning: Layout '{layout}' not recognized. Using 'spring' layout instead.")
        layout = 'spring'
        
    # Calculate node positions
    try:
        pos = layout_funcs[layout](working_graph)
    except Exception as e:
        print(f"Error calculating layout: {e}. Falling back to spring layout.")
        pos = nx.spring_layout(working_graph)
    
    # Set up figure
    plt.figure(figsize=figsize)
    
    # Default node attributes
    node_sizes = [100 for _ in working_graph.nodes()]
    node_colors = ['lightblue' for _ in working_graph.nodes()]
    
    # Get list of nodes (authors) for consistent indexing
    node_list = list(working_graph.nodes())
    
    # Apply centrality to node size and color, if provided
    if centrality_data and centrality_metric in next(iter(centrality_data.values()), {}):
        # Extract the specific centrality metric data
        centrality = {author: data.get(centrality_metric, 0) for author, data in centrality_data.items()}
        
        # Find top authors
        sorted_centrality = sorted(centrality.items(), key=lambda item: item[1], reverse=True)
        top_authors = [author for author, _ in sorted_centrality[:top_n]]
        
        # Set node sizes based on centrality
        node_sizes = [centrality.get(node, 0) * node_size_multiplier for node in node_list]
        
        # Color nodes - use a color gradient from blue to red based on centrality
        min_centrality = min(centrality.values())
        max_centrality = max(centrality.values())
        centrality_range = max_centrality - min_centrality if max_centrality > min_centrality else 1
        
        # Create a color map
        cmap = plt.cm.coolwarm
        
        # Calculate normalized centrality values for coloring
        node_colors = []
        for node in node_list:
            if node in centrality:
                normalized_value = (centrality[node] - min_centrality) / centrality_range
                node_colors.append(cmap(normalized_value))
            else:
                node_colors.append('lightgrey')  # Default for nodes without centrality data
    
    # Calculate edge widths based on co-authorship count
    edge_widths = [data['weight'] * edge_width_multiplier for u, v, data in working_graph.edges(data=True)]
    
    # Draw the network
    nx.draw_networkx_edges(working_graph, pos, width=edge_widths, edge_color='gray', alpha=0.6)
    
    # Create node patches for legend
    node_collection = nx.draw_networkx_nodes(working_graph, pos, 
                                           node_size=node_sizes, 
                                           node_color=node_colors, 
                                           alpha=0.8)
    
    # Add labels (either for all nodes or just the top N)
    if label_top_n_only and centrality_data:
        labels = {node: node for node in top_authors}
        nx.draw_networkx_labels(working_graph, pos, labels=labels, font_size=10, font_weight='bold')
    else:
        nx.draw_networkx_labels(working_graph, pos, font_size=8)
    
    # Add colorbar if we have a color gradient
    if centrality_data and len(set(node_colors)) > 2:  # More than just default and top colors
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label(f'{centrality_metric.capitalize()} Centrality')
    
    # Add legend for top authors
    if centrality_data:
        plt.legend([plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(0.9), markersize=10),
                  plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(0.1), markersize=10)],
                 ['High Influence', 'Low Influence'],
                 loc='upper right')
    
    # Add title and other annotations
    plt.title(f"Co-authorship Network\n{working_graph.number_of_nodes()} Authors, "
             f"{working_graph.number_of_edges()} Collaborations")
    plt.axis('off')  # Hide axes
    
    # Save or show the visualization
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Visualization saved to {save_path}")
    else:
        plt.tight_layout()
        plt.show()

def print_top_authors(centrality_data: Dict[str, Dict[str, float]], 
                     metric: str = 'pagerank', 
                     top_n: int = 20,
                     show_scores: bool = True) -> List[str]:
    """
    Prints the top authors according to the specified centrality metric.
    
    Args:
        centrality_data: Nested dictionary with authors and their centrality scores
        metric: Which centrality metric to use for ranking
        top_n: Number of top authors to print
        show_scores: Whether to display the numerical scores
        
    Returns:
        List of top author names
    """
    # Check if the metric exists in the data
    if not centrality_data or metric not in next(iter(centrality_data.values()), {}):
        raise ValueError(f"Metric '{metric}' not found in centrality data")
    
    # Extract the specified metric and sort
    author_scores = [(author, data.get(metric, 0)) for author, data in centrality_data.items()]
    sorted_authors = sorted(author_scores, key=lambda x: x[1], reverse=True)
    
    # Print the results
    print(f"\nTop {top_n} Authors by {metric.capitalize()} Centrality:")
    for i, (author, score) in enumerate(sorted_authors[:top_n], 1):
        if show_scores:
            print(f"{i}. {author}: {score:.4f}")
        else:
            print(f"{i}. {author}")
            
    return [author for author, _ in sorted_authors[:top_n]]

def compare_centrality_measures(centrality_data: Dict[str, Dict[str, float]], 
                               top_n: int = 10) -> None:
    """
    Compares the rankings of authors across different centrality measures.
    
    Args:
        centrality_data: Nested dictionary with authors and their centrality scores
        top_n: Number of top authors to compare
    """
    # Get available metrics
    metrics = next(iter(centrality_data.values())).keys()
    
    # Create a table for comparison
    comparison = {}
    
    for metric in metrics:
        # Get top authors for this metric
        author_scores = [(author, data.get(metric, 0)) for author, data in centrality_data.items()]
        sorted_authors = sorted(author_scores, key=lambda x: x[1], reverse=True)
        top_authors = [author for author, _ in sorted_authors[:top_n]]
        
        # Add to comparison dict
        comparison[metric] = top_authors
    
    # Print the comparison
    print(f"\nComparison of Top {top_n} Authors Across Centrality Measures:")
    
    # Print as a table
    headers = list(metrics)
    print(" | ".join([f"{h.capitalize():20s}" for h in headers]))
    print("-" * (24 * len(headers)))
    
    for i in range(top_n):
        row = []
        for metric in metrics:
            if i < len(comparison[metric]):
                row.append(f"{i+1}. {comparison[metric][i][:17]:17s}")
            else:
                row.append(" " * 20)
        print(" | ".join(row))

def analyze_coauthorship_network(df: pd.DataFrame, 
                                author_column: str = 'authors',
                                centrality_metrics: List[str] = ['degree', 'betweenness', 'eigenvector', 'pagerank'],
                                top_n: int = 10,
                                visualize: bool = True,
                                save_visualization: Optional[str] = None,
                                verbose: bool = True) -> Tuple[nx.Graph, Dict[str, Dict[str, float]]]:
    """
    Complete analysis pipeline for co-authorship network.
    
    Args:
        df: DataFrame containing author data
        author_column: Column name with author information
        centrality_metrics: List of centrality metrics to calculate
        top_n: Number of top authors to analyze
        visualize: Whether to generate a visualization
        save_visualization: Path to save visualization (if None, display instead)
        verbose: Whether to print detailed information
        
    Returns:
        Tuple of (network graph, centrality data)
    """
    start_time = time.time()
    
    if verbose:
        print(f"Starting co-authorship network analysis on {len(df)} records...")
    
    # Create network
    graph = create_author_coauthorship_network(df, author_column=author_column, show_progress=verbose)
    
    if verbose:
        print(f"Network created with {graph.number_of_nodes()} authors and {graph.number_of_edges()} connections")
        print(f"Network density: {nx.density(graph):.6f}")
        
        # Calculate and print connected components
        components = list(nx.connected_components(graph))
        print(f"Number of connected components: {len(components)}")
        print(f"Size of largest component: {len(max(components, key=len))} authors")
    
    # Calculate influence metrics
    centrality_data = calculate_author_influence(graph, metrics=centrality_metrics)
    
    # Print top authors for each metric
    if verbose:
        for metric in centrality_metrics:
            print_top_authors(centrality_data, metric=metric, top_n=top_n)
        
        # Compare rankings across metrics
        compare_centrality_measures(centrality_data, top_n=top_n)
    
    # Visualize the network
    if visualize:
        # Default to PageRank for visualization if available, otherwise degree
        viz_metric = 'pagerank' if 'pagerank' in centrality_metrics else 'degree'
        
        visualize_coauthorship_network(
            graph, 
            top_n=top_n,
            centrality_data=centrality_data,
            centrality_metric=viz_metric,
            label_top_n_only=True,
            save_path=save_visualization
        )
    
    if verbose:
        print(f"Analysis completed in {time.time() - start_time:.2f} seconds")
        
    return graph, centrality_data

def analyze_author_count_distribution(df, author_count_column='author_count'):
    """
    Comprehensive analysis of the distribution of author counts in a dataset.
    
    Args:
        df: Pandas DataFrame containing paper data
        author_count_column: Name of the column containing the number of authors per paper
        
    Returns:
        Dictionary with analysis results
    """
    # Basic statistics
    print("Author count statistics:")
    stats = df[author_count_column].describe(percentiles=[.25, .5, .75, .9, .95, .99])
    print(stats)
    
    # Create visualizations
    create_author_count_visualizations(df, author_count_column)
    
    # Calculate percentage of papers with each author count
    percentage = (df[author_count_column].value_counts() / len(df) * 100).sort_index()
    print("\nPercentage of papers by author count:")
    for count, percent in percentage.head(20).items():
        print(f"{count} authors: {percent:.2f}%")
    
    # Calculate percentage of papers with more than N authors
    thresholds = [1, 2, 3, 5, 10, 20, 50, 100]
    print("\nCumulative distribution:")
    for threshold in thresholds:
        percent = (df[author_count_column] > threshold).mean() * 100
        print(f"Papers with more than {threshold} authors: {percent:.2f}%")
    
    # Identify patterns in collaboration size
    single_author = (df[author_count_column] == 1).mean() * 100
    small_teams = ((df[author_count_column] > 1) & (df[author_count_column] <= 5)).mean() * 100
    medium_teams = ((df[author_count_column] > 5) & (df[author_count_column] <= 20)).mean() * 100
    large_teams = ((df[author_count_column] > 20) & (df[author_count_column] <= 100)).mean() * 100
    mega_teams = (df[author_count_column] > 100).mean() * 100
    
    print("\nCollaboration patterns:")
    print(f"Single author papers: {single_author:.2f}%")
    print(f"Small teams (2-5 authors): {small_teams:.2f}%")
    print(f"Medium teams (6-20 authors): {medium_teams:.2f}%")
    print(f"Large teams (21-100 authors): {large_teams:.2f}%")
    print(f"Mega-collaborations (>100 authors): {mega_teams:.2f}%")
    
    # Return the analysis results
    return {
        "stats": stats,
        "percentage": percentage,
        "collaboration_patterns": {
            "single_author": single_author,
            "small_teams": small_teams,
            "medium_teams": medium_teams,
            "large_teams": large_teams,
            "mega_teams": mega_teams
        }
    }

def create_author_count_visualizations(df, author_count_column):
    """
    Create visualizations for author count distribution
    
    Args:
        df: Pandas DataFrame containing paper data
        author_count_column: Name of the column containing the number of authors per paper
    """
    # Set the style for better notebook display
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Histogram of all author counts
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x=author_count_column, kde=True, bins=50)
    plt.title('Distribution of Authors per Paper', fontsize=14)
    plt.xlabel('Number of Authors', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(left=0)
    plt.show()
    
    # Histogram with log scale for better visibility of tail
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x=author_count_column, kde=False, bins=50, log_scale=(False, True))
    plt.title('Distribution of Authors per Paper (Log Scale)', fontsize=14)
    plt.xlabel('Number of Authors', fontsize=12)
    plt.ylabel('Frequency (Log Scale)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(left=0)
    plt.show()
    
    # Zoomed histogram for common range (1-20 authors)
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df[df[author_count_column] <= 20], x=author_count_column, kde=True, 
               discrete=True, bins=range(1, 22))
    plt.title('Distribution of Authors per Paper (1-20 authors)', fontsize=14)
    plt.xlabel('Number of Authors', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()
    
    # Bar chart of most common author counts
    count_range = df[author_count_column].value_counts().sort_index()
    plt.figure(figsize=(14, 6))
    ax = count_range.head(20).plot(kind='bar', color='skyblue')
    plt.title('Most Common Author Counts', fontsize=14)
    plt.xlabel('Number of Authors', fontsize=12)
    plt.ylabel('Number of Papers', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Add value labels on top of each bar
    for i, v in enumerate(count_range.head(20)):
        ax.text(i, v + 0.1, str(v), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Cumulative distribution function
    plt.figure(figsize=(12, 6))
    
    # Calculate ECDF
    sorted_data = np.sort(df[author_count_column])
    ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    
    # Plot ECDF
    plt.plot(sorted_data, ecdf, marker='.', linestyle='none', alpha=0.3, markersize=1)
    plt.plot(sorted_data, ecdf, linewidth=2)
    
    plt.title('Cumulative Distribution of Authors per Paper', fontsize=14)
    plt.xlabel('Number of Authors', fontsize=12)
    plt.ylabel('Cumulative Probability', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add vertical lines at important thresholds
    thresholds = [1, 2, 5, 10, 20, 50, 100]
    for t in thresholds:
        if t <= max(sorted_data):
            prob = (df[author_count_column] <= t).mean()
            plt.axvline(x=t, color='r', linestyle='--', alpha=0.5)
            plt.text(t+0.5, 0.5, f"{t}: {prob:.2f}", rotation=90, verticalalignment='center')
    
    plt.show()
    
    # Team size distribution (categorical)
    team_sizes = pd.cut(df[author_count_column], 
                      bins=[0, 1, 5, 20, 100, float('inf')],
                      labels=['Single author', 'Small team (2-5)', 
                             'Medium team (6-20)', 'Large team (21-100)',
                             'Mega collaboration (>100)'])
    
    team_counts = team_sizes.value_counts().sort_index()
    team_percents = team_sizes.value_counts(normalize=True).sort_index() * 100
    
    plt.figure(figsize=(12, 6))
    ax = team_counts.plot(kind='bar', color='skyblue')
    plt.title('Paper Distribution by Team Size Category', fontsize=14)
    plt.xlabel('Team Size Category', fontsize=12)
    plt.ylabel('Number of Papers', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Add value and percentage labels on top of each bar
    for i, v in enumerate(team_counts):
        ax.text(i, v + 0.1, f"{v}\n({team_percents[i]:.1f}%)", ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
def visualize_top_authors_network(graph, centrality_data, metric='pagerank', 
                                 top_n=4, degree_threshold=0, include_connections=True,
                                 node_size_multiplier=5000, figsize=(14, 12)):
    """
    Visualize a network focused on the top N authors based on centrality measures,
    including their immediate connections.
    
    Args:
        graph: NetworkX graph of the full co-authorship network
        centrality_data: Dictionary with centrality measures for all authors
        metric: Which centrality metric to use ('pagerank', 'degree', 'eigenvector')
        top_n: Number of top authors to highlight
        degree_threshold: Only include connections with at least this many collaborations
        include_connections: Whether to include connections between the top authors
        node_size_multiplier: Factor to scale the node sizes
        figsize: Size of the figure (width, height) in inches
        
    Returns:
        The subgraph containing the top authors and their connections
    """
    # Extract the specific centrality metric data
    if metric not in next(iter(centrality_data.values())):
        raise ValueError(f"Metric '{metric}' not found in centrality data. Available metrics: {list(next(iter(centrality_data.values())).keys())}")
    
    centrality = {author: data.get(metric, 0) for author, data in centrality_data.items()}
    
    # Identify top authors
    top_authors = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_author_names = [name for name, _ in top_authors]
    
    print(f"Top {top_n} authors by {metric} centrality:")
    for i, (author, score) in enumerate(top_authors, 1):
        print(f"{i}. {author}: {score:.6f}")
    
    # Create a subgraph with the top authors and their immediate connections
    if include_connections:
        # Include the top authors and all their immediate connections
        subgraph_nodes = set(top_author_names)
        
        # Add immediate neighbors of top authors
        for author in top_author_names:
            subgraph_nodes.update(graph.neighbors(author))
        
        # Create the subgraph
        subgraph = graph.subgraph(subgraph_nodes).copy()
        
        # Remove edges below the threshold (if specified)
        if degree_threshold > 0:
            edges_to_remove = [(u, v) for u, v, d in subgraph.edges(data=True) 
                              if d.get('weight', 1) < degree_threshold]
            subgraph.remove_edges_from(edges_to_remove)
            
            # Remove any isolated nodes after edge removal
            isolated_nodes = [node for node in subgraph.nodes() 
                             if subgraph.degree(node) == 0 and node not in top_author_names]
            subgraph.remove_nodes_from(isolated_nodes)
    else:
        # Only include connections between top authors
        subgraph = graph.subgraph(top_author_names).copy()
    
    # Calculate node sizes based on centrality
    max_size = max(centrality.values()) if centrality else 1
    node_sizes = {}
    
    for node in subgraph.nodes():
        if node in top_author_names:
            # Top authors get size based on their centrality score
            node_sizes[node] = (centrality.get(node, 0) / max_size) * node_size_multiplier
        else:
            # Other authors get a smaller fixed size
            node_sizes[node] = 300
    
    # Calculate edge widths based on collaboration count
    edge_widths = [data.get('weight', 1) * 0.8 for _, _, data in subgraph.edges(data=True)]
    
    # Set up node colors: top authors get special colors, others are gray
    node_colors = []
    colormap = cm.tab10
    
    for node in subgraph.nodes():
        if node in top_author_names:
            # Use the index in top_author_names to assign a unique color
            idx = top_author_names.index(node)
            node_colors.append(colormap(idx))
        else:
            node_colors.append('lightgray')
    
    # Use a suitable layout algorithm
    if len(subgraph) < 100:
        # For smaller graphs, force-directed layouts work well
        pos = nx.spring_layout(subgraph, k=0.5, iterations=50, seed=42)
    else:
        # For larger graphs, faster layouts like Kamada-Kawai
        pos = nx.kamada_kawai_layout(subgraph)
    
    # Create the visualization
    plt.figure(figsize=figsize)
    
    # Draw the network
    nx.draw_networkx_edges(subgraph, pos, width=edge_widths, 
                          alpha=0.6, edge_color='gray')
    
    nx.draw_networkx_nodes(subgraph, pos, 
                          node_size=[node_sizes[node] for node in subgraph.nodes()],
                          node_color=node_colors, 
                          alpha=0.8, linewidths=1, edgecolors='black')
    
    # Add labels for top authors only to reduce clutter
    labels = {node: node for node in top_author_names}
    nx.draw_networkx_labels(subgraph, pos, labels=labels, 
                           font_size=12, font_weight='bold')
    
    # Add a title and remove axes
    metric_name = metric.capitalize()
    plt.title(f"Co-authorship Network for Top {top_n} Authors by {metric_name} Centrality", 
             fontsize=16)
    plt.axis('off')
    
    # Add a legend for the top authors
    legend_elements = []
    for i, author in enumerate(top_author_names):
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label=author,
                                         markerfacecolor=colormap(i), markersize=10))
    
    plt.legend(handles=legend_elements, title="Top Authors", 
              loc='upper right', title_fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    # Report some statistics about the subgraph
    print(f"\nSubgraph statistics:")
    print(f"Number of nodes (authors): {subgraph.number_of_nodes()}")
    print(f"Number of edges (collaborations): {subgraph.number_of_edges()}")
    
    # Calculate average degree for top authors
    avg_degree = sum(subgraph.degree(node) for node in top_author_names) / len(top_author_names)
    print(f"Average number of connections for top authors: {avg_degree:.2f}")
    
    return subgraph