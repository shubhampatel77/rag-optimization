import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union
import pandas as pd
from pathlib import Path
import torch
from collections import defaultdict

class MetricsVisualizer:
    def __init__(
        self,
        results: Dict[str, Dict[str, List[Dict[str, Union[str, List[str], Dict[str, float]]]]]],
        output_dir: Optional[str] = None,
        metrics: Optional[List[str]] = None
    ):
        """
        Initialize visualizer for model comparison.
        
        Args:
            results: Dict mapping model_name -> {
                doc_id -> [ 
                    {
                        question -> str, 
                        reference_context -> [str],
                        ...,
                        metrics -> {accuracy -> float, precision -> float, ...}
                    }
                ]
            }
            output_dir: Directory to save plots
            metrics: List of metrics to visualize
        """
        self.results = self._preprocess_results(results)
        self.output_dir = Path(output_dir) if output_dir else None
        self.metrics = metrics or list(next(iter(self.results.values())).keys())
        
        # Set style
        # plt.style.use('seaborn')
        sns.set_palette("husl")
        

    def _preprocess_results(self, results) -> Dict[str, Dict[str, List[float]]]:
        processed = {}
        for model_name, docs in results.items():
            # Initialize metrics dict for this model
            metrics_dict = defaultdict(list)
            
            # Iterate through each doc's QA results
            for doc_id, qa_results in docs.items():
                for qa_result in qa_results:
                    # Extract metrics values from each QA result
                    for metric, value in qa_result['metrics'].items():
                        # TODO: remove it once all results are generated using updated evaluate() which has .item()
                        metrics_dict[metric].append(value.item() if isinstance(value, torch.Tensor) else value)
                        
            processed[model_name] = dict(metrics_dict)
                
        return processed
    
    def plot_metric_distributions(self, figsize=(20, 12)):
        """Plot KDE distributions for each metric."""
        n_metrics = len(self.metrics)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()
        
        for idx, metric in enumerate(self.metrics):
            ax = axes[idx]
            for model, metrics in self.results.items():
                if metric in metrics:
                    values = metrics[metric]
                    mean = np.mean(values.cpu().numpy() if isinstance(values, torch.Tensor) else values)
                    sns.kdeplot(
                        values, 
                        ax=ax, 
                        label=f'{model} (μ={mean:.3f})',
                        fill=True, 
                        alpha=0.3
                    )
                    ax.axvline(mean, linestyle='--', alpha=0.5)
            
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.legend()
            
        # Remove empty subplots
        for idx in range(len(self.metrics), len(axes)):
            fig.delaxes(axes[idx])
            
        plt.tight_layout()
        if self.output_dir:
            plt.savefig(self.output_dir / 'metric_distributions.png')
        plt.show()
    
    def plot_correlation_matrix(self, figsize=(10, 8)):
        """Plot correlation matrix between metrics."""
        # Create DataFrame with all metrics
        data = []
        for model, metrics in self.results.items():
            for i in range(len(next(iter(metrics.values())))):
                row = {'model': model}
                for metric in self.metrics:
                    row[metric] = metrics[metric][i]
                data.append(row)
        
        df = pd.DataFrame(data)
        
        # Plot correlation matrix
        plt.figure(figsize=figsize)
        corr = df[self.metrics].corr()
        mask = np.triu(np.ones_like(corr), k=1)
        sns.heatmap(
            corr,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            mask=mask,
            center=0,
            square=True
        )
        plt.title('Metric Correlations')
        
        if self.output_dir:
            plt.savefig(self.output_dir / 'correlation_matrix.png')
        plt.show()
        
    def plot_radar_chart(self, figsize=(10, 10)):
        """Plot radar chart comparing models across metrics."""
        metrics = self.metrics
        num_metrics = len(metrics)
        angles = [n / float(num_metrics) * 2 * np.pi for n in range(num_metrics)]
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        
        for model, model_metrics in self.results.items():
            values = [np.mean(model_metrics[metric]) for metric in metrics]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, linewidth=1, linestyle='solid', label=model)
            ax.fill(angles, values, alpha=0.1)
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        plt.xticks(angles[:-1], metrics)
        
        ax.set_rlabel_position(0)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        if self.output_dir:
            plt.savefig(self.output_dir / 'radar_chart.png')
        plt.show()
    
    def generate_summary_table(self) -> pd.DataFrame:
        """Generate summary statistics for each model/metric."""
        data = []
        for model, metrics in self.results.items():
            row = {'Model': model}
            for metric in self.metrics:
                values = metrics[metric]
                row[f'{metric}_mean'] = np.mean(values)
                row[f'{metric}_std'] = np.std(values)
            data.append(row)
        
        df = pd.DataFrame(data)
        if self.output_dir:
            df.to_csv(self.output_dir / 'summary_stats.csv', index=False)
        return df
    
    def plot_all(self, save: bool = True):
        """Generate all visualizations."""
        if save and not self.output_dir:
            raise ValueError("output_dir must be specified to save plots")
        
        self.plot_metric_distributions()
        self.plot_correlation_matrix()
        self.plot_radar_chart()
        self.generate_summary_table()
