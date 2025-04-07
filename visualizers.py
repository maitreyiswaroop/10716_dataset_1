# visualizers.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os

def plot_gan_training_history(train_history, test_metrics=None, save_path=None):
    """
    Plot GAN training history showing generator and discriminator losses.
    
    Args:
        train_history: Dictionary with training metrics
            {
                'g_loss': list of generator loss values,
                'd_loss': list of discriminator loss values,
                'epochs': list of epoch numbers
            }
        test_metrics: Optional dictionary with test metrics
            {
                'mse': list of MSE values,
                'mae': list of MAE values,
                'epochs': list of epoch numbers
            }
        save_path: Path to save the plot
    """
    # Set Seaborn style
    sns.set(style='whitegrid')
    
    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot generator and discriminator losses
    axes[0].plot(train_history['epochs'], train_history['g_loss'], 
                label='Generator Loss', color='blue')
    axes[0].set_ylabel('Generator Loss')
    axes[0].set_title('GAN Training Losses')
    axes[0].legend(loc='upper left')
    
    # Add discriminator loss on secondary y-axis
    ax_d = axes[0].twinx()
    ax_d.plot(train_history['epochs'], train_history['d_loss'], 
             label='Discriminator Loss', color='red')
    ax_d.set_ylabel('Discriminator Loss')
    ax_d.legend(loc='upper right')
    
    # Plot test metrics if available
    if test_metrics and 'epochs' in test_metrics and len(test_metrics['epochs']) > 0:
        axes[1].plot(test_metrics['epochs'], test_metrics['mse'], 
                     label='MSE', color='green')
        axes[1].set_ylabel('MSE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_title('Validation Metrics')
        axes[1].legend(loc='upper left')
        
        ax_mae = axes[1].twinx()
        ax_mae.plot(test_metrics['epochs'], test_metrics['mae'], 
                   label='MAE', color='purple')
        ax_mae.set_ylabel('MAE')
        ax_mae.legend(loc='upper right')
    else:
        axes[1].set_xlabel('Epoch')
        axes[1].set_title('No Validation Metrics Available')
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_model_comparison(model_results, metric='mse', save_path=None):
    """
    Plot performance comparison of multiple models for a specific metric.
    
    Args:
        model_results: Dictionary of model results
            {
                'model_name': {
                    'mse': list of MSE values,
                    'mae': list of MAE values,
                    'epochs': list of epoch numbers
                }
            }
        metric: Metric to plot ('mse' or 'mae')
        save_path: Path to save the plot
    """
    sns.set(style='whitegrid')
    plt.figure(figsize=(12, 8))
    
    colors = sns.color_palette('husl', n_colors=len(model_results))
    
    for i, (model_name, results) in enumerate(model_results.items()):
        if metric in results and 'epochs' in results:
            plt.plot(results['epochs'], results[metric], 
                   label=model_name, color=colors[i])
    
    metric_name = 'Mean Squared Error' if metric == 'mse' else 'Mean Absolute Error'
    plt.ylabel(metric_name)
    plt.xlabel('Epoch')
    plt.title(f'Model Comparison - {metric_name}')
    plt.legend(loc='best')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_all_model_metrics(model_results, save_dir=None):
    """
    Plot all metrics for all models, creating separate plots for MSE and MAE.
    
    Args:
        model_results: Dictionary of model results
        save_dir: Directory to save the plots
    """
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Plot MSE comparison
    mse_path = os.path.join(save_dir, 'model_comparison_mse.png') if save_dir else None
    plot_model_comparison(model_results, metric='mse', save_path=mse_path)
    
    # Plot MAE comparison
    mae_path = os.path.join(save_dir, 'model_comparison_mae.png') if save_dir else None
    plot_model_comparison(model_results, metric='mae', save_path=mae_path)

def plot_return_predictions(real_returns, model_predictions, stock_ids=None, dates=None, save_path=None):
    """
    Plot actual vs. predicted returns for one or more stocks.
    
    Args:
        real_returns: Array or dictionary of actual returns
            If dictionary: {stock_id: array of returns}
        model_predictions: Dictionary of model predictions
            {model_name: array or dictionary of predictions}
            If dictionary: {model_name: {stock_id: array of predictions}}
        stock_ids: Optional list of stock IDs to plot
        dates: Optional array of dates for x-axis
        save_path: Path to save the plot
    """
    sns.set(style='whitegrid')
    
    # Handle the case where real_returns is a dictionary (multiple stocks)
    if isinstance(real_returns, dict):
        # If stock_ids not specified, use all available
        if stock_ids is None:
            stock_ids = list(real_returns.keys())
        
        # Create a subplot for each stock
        n_stocks = len(stock_ids)
        fig, axes = plt.subplots(n_stocks, 1, figsize=(14, 6 * n_stocks), sharex=True)
        
        # Handle the case where there's only one stock
        if n_stocks == 1:
            axes = [axes]
        
        for i, stock_id in enumerate(stock_ids):
            ax = axes[i]
            
            # Plot actual returns for this stock
            if dates is not None:
                ax.plot(dates, real_returns[stock_id], label='Actual', color='black', linewidth=2)
            else:
                ax.plot(real_returns[stock_id], label='Actual', color='black', linewidth=2)
            
            # Plot model predictions for this stock
            colors = sns.color_palette('husl', n_colors=len(model_predictions))
            for j, (model_name, predictions) in enumerate(model_predictions.items()):
                # Handle case where predictions is a dictionary (keyed by stock_id)
                if isinstance(predictions, dict):
                    if stock_id in predictions:
                        if dates is not None:
                            ax.plot(dates, predictions[stock_id], label=f'{model_name}', 
                                  color=colors[j], alpha=0.7)
                        else:
                            ax.plot(predictions[stock_id], label=f'{model_name}', 
                                  color=colors[j], alpha=0.7)
                # Handle case where predictions is a single array (same shape as real_returns)
                else:
                    if dates is not None:
                        ax.plot(dates, predictions, label=f'{model_name}', 
                              color=colors[j], alpha=0.7)
                    else:
                        ax.plot(predictions, label=f'{model_name}', 
                              color=colors[j], alpha=0.7)
            
            ax.set_title(f'Stock ID: {stock_id}')
            ax.set_ylabel('Return')
            ax.legend(loc='best')
            ax.grid(True)
        
        plt.xlabel('Time' if dates is None else 'Date')
        plt.tight_layout()
    
    # Handle the case where real_returns is a single array
    else:
        plt.figure(figsize=(14, 8))
        
        # Plot actual returns
        if dates is not None:
            plt.plot(dates, real_returns, label='Actual Returns', color='black', linewidth=2)
        else:
            plt.plot(real_returns, label='Actual Returns', color='black', linewidth=2)
        
        # Plot model predictions
        colors = sns.color_palette('husl', n_colors=len(model_predictions))
        for i, (model_name, predictions) in enumerate(model_predictions.items()):
            if dates is not None:
                plt.plot(dates, predictions, label=f'{model_name} Predictions', 
                        color=colors[i], alpha=0.7)
            else:
                plt.plot(predictions, label=f'{model_name} Predictions', 
                        color=colors[i], alpha=0.7)
        
        plt.xlabel('Time' if dates is None else 'Date')
        plt.ylabel('Return')
        plt.title('Actual vs. Predicted Returns')
        plt.legend(loc='best')
        plt.grid(True)
        
        if dates is not None:
            plt.xticks(rotation=45)
        
        plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_stock_clustering(forecasts, method='tsne', n_components=2, save_path=None):
    """
    Plot a clustering of stocks based on their forecast patterns.
    
    Args:
        forecasts: Dictionary mapping stock_ids to forecast arrays
        method: Dimensionality reduction method ('tsne', 'pca', or 'umap')
        n_components: Number of components for dimensionality reduction
        save_path: Path to save the plot
    """
    try:
        # Convert forecasts to a matrix (stocks x time)
        stock_ids = list(forecasts.keys())
        forecast_matrix = np.array([forecasts[sid] for sid in stock_ids])
        
        # Handle missing values
        nan_mask = np.isnan(forecast_matrix)
        if np.any(nan_mask):
            # Impute NaNs with column means
            col_means = np.nanmean(forecast_matrix, axis=0)
            forecast_matrix = np.where(nan_mask, np.take(col_means, np.indices(forecast_matrix.shape)[1]), forecast_matrix)
        
        # Apply dimensionality reduction
        if method == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=n_components, random_state=42)
        elif method == 'pca':
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=n_components)
        elif method == 'umap':
            import umap
            reducer = umap.UMAP(n_components=n_components, random_state=42)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Fit and transform
        embedding = reducer.fit_transform(forecast_matrix)
        
        # Apply clustering (optional)
        from sklearn.cluster import KMeans
        n_clusters = min(8, len(stock_ids))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embedding)
        
        # Plot the results
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=clusters, cmap='viridis', alpha=0.8)
        
        # Add stock ID labels for a subset of points
        max_labels = min(20, len(stock_ids))
        step = max(1, len(stock_ids) // max_labels)
        for i in range(0, len(stock_ids), step):
            plt.annotate(stock_ids[i], (embedding[i, 0], embedding[i, 1]), fontsize=8)
        
        plt.colorbar(scatter, label='Cluster')
        plt.title(f'Stock Clustering based on Forecast Patterns ({method.upper()})')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            
    except Exception as e:
        print(f"Error in plotting stock clustering: {e}")

def load_model_results(directory):
    """
    Load model results from a directory of JSON files.
    
    Args:
        directory: Directory containing model result files
        
    Returns:
        Dictionary of model results
    """
    model_results = {}
    
    if not os.path.exists(directory):
        return model_results
        
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            model_name = filename.split('.')[0]
            filepath = os.path.join(directory, filename)
            
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    
                if 'test_metrics' in data:
                    model_results[model_name] = data['test_metrics']
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                
    return model_results

def plot_multiple_stock_forecasts(time_steps, real_values, predicted_values, stock_ids=None, num_stocks=4, save_path=None):
    """
    Plot actual vs. predicted values for multiple stocks.
    
    Args:
        time_steps: Array of time steps (e.g., days) for x-axis
        real_values: Dictionary mapping stock IDs to arrays of actual values
        predicted_values: Dictionary mapping stock IDs to arrays of predicted values
        stock_ids: Optional list of specific stock IDs to plot (will select randomly if None)
        num_stocks: Number of stocks to plot
        save_path: Path to save the plot
    """
    # Set style
    sns.set(style='whitegrid')
    
    # If stock_ids not provided, randomly select from available stocks
    if stock_ids is None:
        all_stock_ids = list(real_values.keys())
        num_to_plot = min(num_stocks, len(all_stock_ids))
        stock_ids = np.random.choice(all_stock_ids, size=num_to_plot, replace=False)
    else:
        num_to_plot = min(num_stocks, len(stock_ids))
        stock_ids = stock_ids[:num_to_plot]
    
    # Create subplots
    fig, axes = plt.subplots(num_to_plot, 1, figsize=(12, 4 * num_to_plot), sharex=True)
    
    # Handle case with only one subplot
    if num_to_plot == 1:
        axes = [axes]
    
    # Plot each stock
    for i, stock_id in enumerate(stock_ids):
        ax = axes[i]
        
        # Plot actual values
        if stock_id in real_values:
            ax.plot(time_steps, real_values[stock_id], 'b-', label='Actual', linewidth=2)
        
        # Plot predicted values
        if stock_id in predicted_values:
            ax.plot(time_steps, predicted_values[stock_id], 'r--', label='Predicted', linewidth=2)
            
            # Calculate and display error metrics
            mse = np.mean((real_values[stock_id] - predicted_values[stock_id])**2)
            mae = np.mean(np.abs(real_values[stock_id] - predicted_values[stock_id]))
            ax.text(0.02, 0.95, f'MSE: {mse:.4f}, MAE: {mae:.4f}', 
                   transform=ax.transAxes, fontsize=10, verticalalignment='top')
        
        ax.set_title(f'Stock ID: {stock_id}')
        ax.set_ylabel('Value')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    # Set common x label
    plt.xlabel('Time')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def create_performance_report(model_results, output_dir=None):
    """
    Create a comprehensive performance report for all models.
    
    Args:
        model_results: Dictionary of model results
        output_dir: Directory to save the report files
    """
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create a DataFrame for comparison
    comparison_data = []
    
    for model_name, metrics in model_results.items():
        if 'mse' in metrics and len(metrics['mse']) > 0:
            # Get metrics from the last epoch
            last_idx = -1
            row = {
                'Model': model_name,
                'Final MSE': metrics['mse'][last_idx],
                'Final MAE': metrics['mae'][last_idx] if 'mae' in metrics else None,
                'Best MSE': min(metrics['mse']),
                'Best MSE Epoch': metrics['epochs'][np.argmin(metrics['mse'])]
            }
            comparison_data.append(row)
    
    if comparison_data:
        # Create and save comparison table
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Final MSE')
        
        print("\nModel Performance Comparison:")
        print(comparison_df)
        
        if output_dir:
            # Save comparison to CSV
            comparison_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
            
            # Create plots
            plot_all_model_metrics(model_results, save_dir=output_dir)
            
            # Save text report
            with open(os.path.join(output_dir, 'performance_report.txt'), 'w') as f:
                f.write("MODEL PERFORMANCE REPORT\n")
                f.write("======================\n\n")
                f.write("Overall Comparison:\n")
                f.write(comparison_df.to_string())
                f.write("\n\n")
                
                for model_name, metrics in model_results.items():
                    f.write(f"\n{model_name} Details:\n")
                    f.write("-------------------\n")
                    if 'mse' in metrics and len(metrics['mse']) > 0:
                        f.write(f"Final MSE: {metrics['mse'][-1]:.6f}\n")
                        f.write(f"Best MSE: {min(metrics['mse']):.6f} (Epoch {metrics['epochs'][np.argmin(metrics['mse'])]})\n")
                    if 'mae' in metrics and len(metrics['mae']) > 0:
                        f.write(f"Final MAE: {metrics['mae'][-1]:.6f}\n")
                        f.write(f"Best MAE: {min(metrics['mae']):.6f} (Epoch {metrics['epochs'][np.argmin(metrics['mae'])]})\n")
    else:
        print("No model results available for comparison.")

def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
    """
    Plot feature importance for a trained model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n: Number of top features to display
        save_path: Path to save the plot
    """
    # Check if model has feature_importances_ attribute
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature_importances_ attribute")
        return
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    
    # Select top N features
    top_indices = indices[:top_n]
    top_importances = importances[top_indices]
    top_names = [feature_names[i] for i in top_indices]
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(top_names)), top_importances, align='center')
    plt.yticks(range(len(top_names)), top_names)
    plt.xlabel('Feature Importance')
    plt.title('Top Feature Importances')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()