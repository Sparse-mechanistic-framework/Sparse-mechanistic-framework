"""
Circuit Visualization and Analysis Tools for SMA
Provides visualization of discovered circuits and attention patterns
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any
import networkx as nx
from pathlib import Path
import json
import logging
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class CircuitVisualizer:
    """
    Visualize discovered circuits and information flow
    """
    
    def __init__(
        self,
        model_config: Dict,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize circuit visualizer
        
        Args:
            model_config: Model configuration (n_layers, n_heads, etc.)
            output_dir: Directory for saving visualizations
        """
        self.n_layers = model_config.get('n_layers', 12)
        self.n_heads = model_config.get('n_heads', 12)
        self.hidden_size = model_config.get('hidden_size', 768)
        
        self.output_dir = Path(output_dir) if output_dir else Path('./visualizations')
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def visualize_circuit_graph(
        self,
        circuits: List[Any],
        save_name: str = 'circuit_graph.html'
    ):
        """
        Create interactive circuit graph visualization
        
        Args:
            circuits: List of Circuit objects
            save_name: Name for saved visualization
        """
        # Create graph
        G = nx.DiGraph()
        
        # Add nodes for each layer and component
        for layer in range(self.n_layers):
            # Add attention nodes
            G.add_node(
                f'L{layer}_attn',
                layer=layer,
                type='attention',
                label=f'Layer {layer}\nAttention'
            )
            # Add MLP nodes
            G.add_node(
                f'L{layer}_mlp',
                layer=layer,
                type='mlp',
                label=f'Layer {layer}\nMLP'
            )
            
            # Add connections between layers
            if layer > 0:
                G.add_edge(f'L{layer-1}_attn', f'L{layer}_attn', weight=0.5)
                G.add_edge(f'L{layer-1}_mlp', f'L{layer}_attn', weight=0.3)
                G.add_edge(f'L{layer-1}_attn', f'L{layer}_mlp', weight=0.3)
                G.add_edge(f'L{layer-1}_mlp', f'L{layer}_mlp', weight=0.5)
        
        # Highlight discovered circuits
        for circuit in circuits:
            for i, layer in enumerate(circuit.layers):
                if 'attention' in circuit.components[i]:
                    node_id = f'L{layer}_attn'
                else:
                    node_id = f'L{layer}_mlp'
                
                # Update node properties
                G.nodes[node_id]['importance'] = circuit.importance_score
                G.nodes[node_id]['in_circuit'] = True
        
        # Create Plotly visualization
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Create edge traces
        edge_traces = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=0.5, color='#888'),
                hoverinfo='none'
            )
            edge_traces.append(edge_trace)
        
        # Create node traces
        node_trace = go.Scatter(
            x=[],
            y=[],
            text=[],
            mode='markers+text',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlOrRd',
                size=20,
                color=[],
                colorbar=dict(
                    thickness=15,
                    title='Importance',
                    xanchor='left',
                    title_side='right'
                )
            )
        )
        
        for node in G.nodes():
            x, y = pos[node]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
            
            # Color based on importance
            importance = G.nodes[node].get('importance', 0)
            node_trace['marker']['color'] += tuple([importance])
            
            # Add label
            node_info = G.nodes[node]['label']
            if G.nodes[node].get('in_circuit', False):
                node_info += f"\nImportance: {importance:.3f}"
            node_trace['text'] += tuple([node_info])
        
        # Create figure
        fig = go.Figure(
            data=edge_traces + [node_trace],
            layout=go.Layout(
                title='Circuit Discovery Graph',
                showlegend=False,
                hovermode='closest',
                margin=dict(b=0, l=0, r=0, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='white'
            )
        )
        
        # Save
        output_path = self.output_dir / save_name
        fig.write_html(str(output_path))
        logger.info(f"Saved circuit graph to {output_path}")
    
    def visualize_attention_patterns(
        self,
        attention_weights: torch.Tensor,
        query_tokens: List[str],
        doc_tokens: List[str],
        layer: int,
        head: Optional[int] = None,
        save_name: Optional[str] = None
    ):
        """
        Visualize attention patterns between query and document
        
        Args:
            attention_weights: Attention weights tensor
            query_tokens: Query tokens
            doc_tokens: Document tokens
            layer: Layer index
            head: Specific head to visualize (None for average)
            save_name: Name for saved figure
        """
        # Process attention weights
        if len(attention_weights.shape) == 4:  # [batch, heads, seq, seq]
            attention_weights = attention_weights[0]  # Remove batch dimension
        
        if head is not None:
            attn = attention_weights[head].cpu().numpy()
            title = f'Layer {layer}, Head {head} Attention'
        else:
            attn = attention_weights.mean(dim=0).cpu().numpy()
            title = f'Layer {layer} Average Attention'
        
        # Separate query and document regions
        n_query = len(query_tokens)
        n_doc = len(doc_tokens)
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Query to Query attention
        ax = axes[0]
        sns.heatmap(
            attn[:n_query, :n_query],
            xticklabels=query_tokens,
            yticklabels=query_tokens,
            cmap='Blues',
            ax=ax,
            cbar_kws={'label': 'Attention Weight'}
        )
        ax.set_title('Query → Query')
        ax.set_xlabel('Query Tokens (Target)')
        ax.set_ylabel('Query Tokens (Source)')
        
        # Query to Document attention
        ax = axes[1]
        sns.heatmap(
            attn[:n_query, n_query:n_query+n_doc],
            xticklabels=doc_tokens[:n_doc],
            yticklabels=query_tokens,
            cmap='Reds',
            ax=ax,
            cbar_kws={'label': 'Attention Weight'}
        )
        ax.set_title('Query → Document')
        ax.set_xlabel('Document Tokens')
        ax.set_ylabel('Query Tokens')
        
        # Document to Query attention
        ax = axes[2]
        sns.heatmap(
            attn[n_query:n_query+n_doc, :n_query],
            xticklabels=query_tokens,
            yticklabels=doc_tokens[:n_doc],
            cmap='Greens',
            ax=ax,
            cbar_kws={'label': 'Attention Weight'}
        )
        ax.set_title('Document → Query')
        ax.set_xlabel('Query Tokens')
        ax.set_ylabel('Document Tokens')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_name:
            output_path = self.output_dir / save_name
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved attention pattern to {output_path}")
        
        plt.show()
    
    def visualize_importance_distribution(
        self,
        importance_scores: Dict[str, float],
        save_name: str = 'importance_distribution.png'
    ):
        """
        Visualize distribution of importance scores across components
        
        Args:
            importance_scores: Dictionary of component importance scores
            save_name: Name for saved figure
        """
        # Organize scores by layer and type
        layer_attention_scores = {}
        layer_mlp_scores = {}
        
        for component, score in importance_scores.items():
            parts = component.split('_')
            if len(parts) >= 3:
                layer = int(parts[1])
                comp_type = parts[2]
                
                if comp_type == 'attention':
                    layer_attention_scores[layer] = score
                elif comp_type == 'mlp':
                    layer_mlp_scores[layer] = score
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Attention scores
        ax = axes[0]
        layers = sorted(layer_attention_scores.keys())
        scores = [layer_attention_scores[l] for l in layers]
        
        bars = ax.bar(layers, scores, color='steelblue', alpha=0.7)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Importance Score')
        ax.set_title('Attention Component Importance Across Layers')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.3f}', ha='center', va='bottom', fontsize=8)
        
        # MLP scores
        ax = axes[1]
        layers = sorted(layer_mlp_scores.keys())
        scores = [layer_mlp_scores[l] for l in layers]
        
        bars = ax.bar(layers, scores, color='coral', alpha=0.7)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Importance Score')
        ax.set_title('MLP Component Importance Across Layers')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.suptitle('Component Importance Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved importance distribution to {output_path}")
        plt.show()
    
    def create_circuit_summary_report(
        self,
        circuits: List[Any],
        importance_scores: Dict[str, float],
        save_name: str = 'circuit_summary.html'
    ):
        """
        Create comprehensive HTML report of discovered circuits
        
        Args:
            circuits: List of discovered circuits
            importance_scores: Component importance scores
            save_name: Report filename
        """
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>SMA Circuit Discovery Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
                h2 {{ color: #555; margin-top: 30px; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .circuit-box {{ 
                    background: #f9f9f9; 
                    border: 1px solid #ddd; 
                    padding: 15px; 
                    margin: 15px 0;
                    border-radius: 5px;
                }}
                .importance-high {{ color: #d32f2f; font-weight: bold; }}
                .importance-medium {{ color: #f57c00; }}
                .importance-low {{ color: #388e3c; }}
            </style>
        </head>
        <body>
            <h1>SMA Phase 1: Circuit Discovery Report</h1>
            
            <h2>Summary Statistics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Total Circuits Discovered</td>
                    <td>{n_circuits}</td>
                </tr>
                <tr>
                    <td>Average Circuit Importance</td>
                    <td>{avg_importance:.4f}</td>
                </tr>
                <tr>
                    <td>Max Circuit Importance</td>
                    <td>{max_importance:.4f}</td>
                </tr>
                <tr>
                    <td>Components Analyzed</td>
                    <td>{n_components}</td>
                </tr>
            </table>
            
            <h2>Top Discovered Circuits</h2>
            {circuits_html}
            
            <h2>Component Importance Rankings</h2>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Component</th>
                    <th>Importance Score</th>
                    <th>Type</th>
                </tr>
                {importance_table}
            </table>
            
            <h2>Layer-wise Analysis</h2>
            {layer_analysis}
            
        </body>
        </html>
        """
        
        # Calculate statistics
        n_circuits = len(circuits)
        avg_importance = np.mean([c.importance_score for c in circuits]) if circuits else 0
        max_importance = max([c.importance_score for c in circuits]) if circuits else 0
        n_components = len(importance_scores)
        
        # Generate circuits HTML
        circuits_html = ""
        for i, circuit in enumerate(circuits[:10]):  # Top 10 circuits
            importance_class = (
                "importance-high" if circuit.importance_score > 0.7
                else "importance-medium" if circuit.importance_score > 0.4
                else "importance-low"
            )
            
            circuits_html += f"""
            <div class="circuit-box">
                <h3>Circuit {i+1}: {circuit.name}</h3>
                <p><strong>Importance:</strong> <span class="{importance_class}">{circuit.importance_score:.4f}</span></p>
                <p><strong>Layers:</strong> {', '.join(map(str, circuit.layers))}</p>
                <p><strong>Components:</strong> {', '.join(circuit.components)}</p>
                <p><strong>Function:</strong> {circuit.function_description or 'Analysis pending'}</p>
            </div>
            """
        
        # Generate importance table
        sorted_importance = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        importance_table = ""
        for rank, (component, score) in enumerate(sorted_importance[:20], 1):
            comp_type = component.split('_')[2] if len(component.split('_')) > 2 else 'unknown'
            importance_table += f"""
            <tr>
                <td>{rank}</td>
                <td>{component}</td>
                <td>{score:.4f}</td>
                <td>{comp_type}</td>
            </tr>
            """
        
        # Generate layer analysis
        layer_stats = {}
        for component, score in importance_scores.items():
            parts = component.split('_')
            if len(parts) >= 2 and parts[1].isdigit():
                layer = int(parts[1])
                if layer not in layer_stats:
                    layer_stats[layer] = []
                layer_stats[layer].append(score)
        
        layer_analysis = "<table><tr><th>Layer</th><th>Avg Importance</th><th>Max Importance</th><th># Components</th></tr>"
        for layer in sorted(layer_stats.keys()):
            scores = layer_stats[layer]
            layer_analysis += f"""
            <tr>
                <td>Layer {layer}</td>
                <td>{np.mean(scores):.4f}</td>
                <td>{np.max(scores):.4f}</td>
                <td>{len(scores)}</td>
            </tr>
            """
        layer_analysis += "</table>"
        
        # Format final HTML
        html_final = html_content.format(
            n_circuits=n_circuits,
            avg_importance=avg_importance,
            max_importance=max_importance,
            n_components=n_components,
            circuits_html=circuits_html,
            importance_table=importance_table,
            layer_analysis=layer_analysis
        )
        
        # Save report
        output_path = self.output_dir / save_name
        with open(output_path, 'w') as f:
            f.write(html_final)
        
        logger.info(f"Saved circuit summary report to {output_path}")


class MetricsTracker:
    """
    Track and visualize metrics during analysis
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize metrics tracker"""
        self.output_dir = Path(output_dir) if output_dir else Path('./metrics')
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.metrics = {
            'causal_effects': [],
            'circuit_scores': [],
            'component_importance': {},
            'analysis_time': []
        }
    
    def log_causal_effect(
        self,
        layer: int,
        component: str,
        effect: float,
        query: str
    ):
        """Log causal effect measurement"""
        self.metrics['causal_effects'].append({
            'layer': layer,
            'component': component,
            'effect': effect,
            'query': query[:50]  # Truncate query for storage
        })
    
    def log_circuit(self, circuit: Any):
        """Log discovered circuit"""
        self.metrics['circuit_scores'].append({
            'name': circuit.name,
            'importance': circuit.importance_score,
            'n_layers': len(circuit.layers),
            'n_components': len(circuit.components)
        })
    
    def log_importance(self, component: str, score: float):
        """Log component importance score"""
        self.metrics['component_importance'][component] = score
    
    def save_metrics(self, filename: str = 'metrics.json'):
        """Save all metrics to file"""
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"Saved metrics to {output_path}")
    
    def plot_convergence(self, save_name: str = 'convergence.png'):
        """Plot convergence of importance scores over iterations"""
        if not self.metrics['causal_effects']:
            logger.warning("No causal effects to plot")
            return
        
        # Group by layer
        effects_by_layer = {}
        for effect in self.metrics['causal_effects']:
            layer = effect['layer']
            if layer not in effects_by_layer:
                effects_by_layer[layer] = []
            effects_by_layer[layer].append(effect['effect'])
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for layer, effects in effects_by_layer.items():
            # Compute running average
            running_avg = np.cumsum(effects) / np.arange(1, len(effects) + 1)
            ax.plot(running_avg, label=f'Layer {layer}', alpha=0.7)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Average Causal Effect')
        ax.set_title('Convergence of Causal Effects by Layer')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved convergence plot to {output_path}")
        plt.show()