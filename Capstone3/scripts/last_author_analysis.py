import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Set, Tuple
import numpy as np
from tqdm import tqdm

def extract_position_authors(df: pd.DataFrame, 
                           position: str = 'last', 
                           author_column: str = 'authors',
                           delimiter: str = ',') -> Dict[str, int]:
    """
    Extracts authors from a specific position (first, last) in each paper.
    
    Args:
        df: DataFrame containing paper information
        position: Which position to extract ('first', 'last', or 'both')
        author_column: Column name containing author information
        delimiter: Character used to separate authors if stored as strings
        
    Returns:
        Dictionary mapping author names to the count of how many times 
        they appear in that position
    """
    position_counts = {}
    
    # Process each paper
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {position} authors"):
        try:
            # Extract authors
            if pd.isna(row[author_column]).all() if isinstance(pd.isna(row[author_column]), np.ndarray) else pd.isna(row[author_column]):
                continue
                
            if isinstance(row[author_column], str):
                authors = [author.strip() for author in row[author_column].split(delimiter) if author.strip()]
            elif isinstance(row[author_column], list):
                authors = [author.strip() for author in row[author_column] if author and isinstance(author, str)]
            else:
                continue
                
            # Skip papers with insufficient authors
            if len(authors) < 1:
                continue
                
            # Extract authors based on position
            if position == 'first' or position == 'both':
                first_author = authors[0]
                position_counts[first_author] = position_counts.get(first_author, 0) + 1
                
            if position == 'last' or position == 'both':
                if len(authors) > 1:  # Ensure there's more than one author
                    last_author = authors[-1]
                    position_counts[last_author] = position_counts.get(last_author, 0) + 1
                    
        except Exception as e:
            print(f"Error processing row: {e}")
            
    return position_counts

def create_position_focused_subgraph(full_graph: nx.Graph, 
                                    position_authors: Dict[str, int],
                                    min_papers: int = 5,
                                    include_connections: bool = True) -> nx.Graph:
    """
    Creates a subgraph focused on authors in a specific position,
    optionally including connections between these authors.
    
    Args:
        full_graph: The full co-authorship network
        position_authors: Dictionary mapping author names to position counts
        min_papers: Minimum number of papers an author must have in the position
        include_connections: Whether to include connections between position authors
        
    Returns:
        NetworkX subgraph containing only the specified position authors
    """
    # Filter authors by minimum paper count
    qualified_authors = {author for author, count in position_authors.items() 
                        if count >= min_papers and author in full_graph}
    
    # Create subgraph
    if include_connections:
        # Get the induced subgraph (includes all edges between these authors)
        subgraph = full_graph.subgraph(qualified_authors).copy()
    else:
        # Create a new graph with just the nodes (no edges)
        subgraph = nx.Graph()
        for author in qualified_authors:
            subgraph.add_node(author)
    
    # Add position count as a node attribute
    for author in subgraph:
        subgraph.nodes[author]['position_count'] = position_authors.get(author, 0)
        
    return subgraph

def analyze_position_network(graph: nx.Graph, 
                           position_counts: Dict[str, int],
                           position: str = 'last',
                           top_n: int = 20) -> None:
    """
    Analyzes a network of authors in a specific position.
    
    Args:
        graph: The network of position-specific authors
        position_counts: Dictionary mapping author names to position counts
        position: Which position being analyzed ('first', 'last', etc.)
        top_n: Number of top authors to display
    """
    # Get all authors in the graph
    authors_in_graph = set(graph.nodes())
    
    # Filter position counts for authors in the graph
    filtered_counts = {author: count for author, count in position_counts.items()
                      if author in authors_in_graph}
    
    # Sort by frequency
    sorted_authors = sorted(filtered_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Calculate centrality measures
    degree_centrality = nx.degree_centrality(graph)
    
    # Handle disconnected graphs for eigenvector centrality
    try:
        eigenvector_centrality = nx.eigenvector_centrality_numpy(graph)
    except nx.AmbiguousSolution:
        print("\nNote: Graph is disconnected, calculating eigenvector centrality for the largest component only.")
        components = list(nx.connected_components(graph))
        largest_component = max(components, key=len)
        largest_subgraph = graph.subgraph(largest_component).copy()
        
        # Calculate for largest component and use zeros for other nodes
        eigenvector_centrality = {node: 0.0 for node in graph.nodes()}
        component_centrality = nx.eigenvector_centrality_numpy(largest_subgraph)
        eigenvector_centrality.update(component_centrality)
    
    pagerank = nx.pagerank(graph)
    
    # Print top authors by position count
    print(f"\nTop {top_n} {position} authors by number of papers:")
    for i, (author, count) in enumerate(sorted_authors[:top_n], 1):
        print(f"{i}. {author}: {count} papers")
    
    # Print top authors by degree centrality (most connections)
    print(f"\nTop {position} authors by degree centrality (most connections):")
    for i, (author, score) in enumerate(sorted(degree_centrality.items(), 
                                             key=lambda x: x[1], 
                                             reverse=True)[:top_n], 1):
        papers = position_counts.get(author, 0)
        print(f"{i}. {author}: {score:.4f} (appears as {position} author in {papers} papers)")
    
    # Print top authors by eigenvector centrality (connected to well-connected authors)
    print(f"\nTop {position} authors by eigenvector centrality (connected to other influential authors):")
    for i, (author, score) in enumerate(sorted(eigenvector_centrality.items(), 
                                             key=lambda x: x[1], 
                                             reverse=True)[:top_n], 1):
        papers = position_counts.get(author, 0)
        print(f"{i}. {author}: {score:.4f} (appears as {position} author in {papers} papers)")
    
    # Print top authors by PageRank
    print(f"\nTop {position} authors by PageRank:")
    for i, (author, score) in enumerate(sorted(pagerank.items(), 
                                             key=lambda x: x[1], 
                                             reverse=True)[:top_n], 1):
        papers = position_counts.get(author, 0)
        print(f"{i}. {author}: {score:.6f} (appears as {position} author in {papers} papers)")
    
    # Calculate network statistics
    print(f"\nNetwork statistics for {position} author network:")
    print(f"Number of {position} authors: {graph.number_of_nodes()}")
    print(f"Number of connections: {graph.number_of_edges()}")
    print(f"Network density: {nx.density(graph):.6f}")
    
    # Calculate and print connected components
    components = list(nx.connected_components(graph))
    print(f"Number of connected components: {len(components)}")
    print(f"Size of largest component: {len(max(components, key=len))} authors")

def visualize_position_network(graph: nx.Graph, 
                              position_counts: Dict[str, int],
                              position: str = 'last',
                              top_n: int = 15, 
                              min_edge_weight: int = 2,
                              node_size_multiplier: int = 50,
                              figsize: Tuple[int, int] = (14, 12)) -> None:
    """
    Visualizes a network of authors in a specific position.
    
    Args:
        graph: The network of position-specific authors
        position_counts: Dictionary mapping author names to position counts
        position: Which position being visualized ('first', 'last', etc.)
        top_n: Number of top authors to label
        min_edge_weight: Minimum edge weight to display
        node_size_multiplier: Factor to scale node sizes
        figsize: Figure size as tuple (width, height)
    """
    # Create a copy of the graph for visualization
    viz_graph = graph.copy()
    
    # Remove weak connections
    edges_to_remove = [(u, v) for u, v, d in viz_graph.edges(data=True) 
                      if d.get('weight', 1) < min_edge_weight]
    viz_graph.remove_edges_from(edges_to_remove)
    
    # Remove isolated nodes
    isolates = list(nx.isolates(viz_graph))
    viz_graph.remove_nodes_from(isolates)
    
    # If graph is empty after filtering, return with a message
    if viz_graph.number_of_nodes() == 0:
        print(f"No nodes remain after filtering. Try reducing min_edge_weight.")
        return
    
    # Get the top authors for labeling
    top_authors = sorted(position_counts.items(), key=lambda x: x[1], reverse=True)
    top_author_names = [author for author, _ in top_authors[:top_n] 
                       if author in viz_graph.nodes()]
    
    # Calculate node sizes based on position count
    max_count = max(position_counts.values()) if position_counts else 1
    node_sizes = {}
    for node in viz_graph.nodes():
        count = position_counts.get(node, 0)
        size = (count / max_count) * node_size_multiplier * 20  # Scale for visibility
        node_sizes[node] = max(size, 100)  # Minimum size for visibility
    
    # Layout calculation
    print("Calculating network layout...")
    if viz_graph.number_of_nodes() < 500:
        pos = nx.spring_layout(viz_graph, k=0.3, iterations=50, seed=42)
    else:
        pos = nx.kamada_kawai_layout(viz_graph)
    
    # Visualization
    plt.figure(figsize=figsize)
    
    # Draw edges
    edge_weights = [d.get('weight', 1) for _, _, d in viz_graph.edges(data=True)]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [0.5 + (w / max_weight) * 3 for w in edge_weights]  # Scale edge widths
    
    nx.draw_networkx_edges(viz_graph, pos, 
                          width=edge_widths, 
                          alpha=0.3, 
                          edge_color='gray')
    
    # Create a colormap for nodes
    # Use size (position count) to determine color
    node_colors = []
    for node in viz_graph.nodes():
        if node in top_author_names:
            # Top authors get red color with intensity based on position
            position = top_author_names.index(node) / len(top_author_names)
            # Red with varying brightness
            node_colors.append((1.0, 0.4 * position, 0.4 * position))
        else:
            # Other nodes get blue color
            node_colors.append('skyblue')
    
    # Draw nodes
    nx.draw_networkx_nodes(viz_graph, pos, 
                          node_size=[node_sizes[node] for node in viz_graph.nodes()],
                          node_color=node_colors, 
                          alpha=0.8, 
                          linewidths=0.5, 
                          edgecolors='black')
    
    # Add labels for top authors only
    labels = {node: node for node in top_author_names if node in viz_graph.nodes()}
    nx.draw_networkx_labels(viz_graph, pos, 
                           labels=labels, 
                           font_size=10, 
                           font_weight='bold')
    
    # Add title and remove axes
    position_labels = {
        'first': 'First Authors (Primary Researchers)',
        'last': 'Last Authors (Lab Heads/PIs)',
        'both': 'First and Last Authors'
    }
    
    title = f"Network of {position_labels.get(position, position)}"
    subtitle = f"(Top {len(labels)} of {viz_graph.number_of_nodes()} authors labeled, min {min_edge_weight} collaborations)"
    
    plt.title(f"{title}\n{subtitle}", fontsize=14)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics about the visualization
    print(f"\nVisualization statistics:")
    print(f"Showing {viz_graph.number_of_nodes()} authors with {viz_graph.number_of_edges()} connections")
    print(f"Filtered out {len(isolates)} isolated authors")
    print(f"Minimum edge weight: {min_edge_weight} collaborations")
    print(f"Labeled the top {len(labels)} authors")
