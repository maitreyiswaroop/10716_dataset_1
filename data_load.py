import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_path):
    """
    Load data from a .npy file
        x_data:  200 alpha signals, spanning 3 years of data in ~1200 stocks. Each row/data 
        point corresponds to some stock day tuple, and there are 1.123m data points.  
        x_data shape (1123742, 200) 
        
        y_data: target y is next day return.  
        y_data shape (1123742,) 
        
        si: stock index 
        si shape (1123742,) 
        
        di: day index 
        di shape (1123742,) 
        start day index: 3776 (corresponding to 20210104); end day index: 4528 (corresponding 
        to 20231229) 
        
        raw_data: 
        11 raw data variables that may be interesting to include in the models and analysis. 
        
        list_of_data ['close', 'open', 'low', 'high', 'volume', 'trading_days_til_next_ann', 
        'trading_days_since_last_ann', 'close_VIX', 'ret1_SPX', 'sector', 'industry'] 

    """
    data_array = np.load(file_path, allow_pickle=True)
    data_dict = data_array.item()
    return data_dict

def get_data(data_dict):
    """
    Get data from the data dictionary
    """
    x_data = data_dict['x_data']
    y_data = data_dict['y_data']
    si = data_dict['si']
    di = data_dict['di']
    raw_data = data_dict['raw_data']
    list_of_data = data_dict['list_of_data']
    return x_data, y_data, si, di, raw_data, list_of_data

def get_raw_data(data_dict):
    """
    Get raw data from the data dictionary
    """
    raw_data = data_dict['raw_data']
    return raw_data

def preview_data(data_dict):
    """
    Preview the data
    """
    x_data, y_data, si, di, raw_data, list_of_data = get_data(data_dict)
    print('x_data shape:', x_data.shape)
    print('y_data shape:', y_data.shape)
    print('si shape:', si.shape)
    print('di shape:', di.shape)
    print('raw_data:', raw_data)
    print('list_of_data:', list_of_data)
    return

def summarize_stock_data(data_dict, top_n=10, plot=True, save_csv=False, csv_path='stock_data_summary.csv'):
    """
    Create a summary of how much data is available for each stock.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing the loaded dataset
    top_n : int, optional (default=10)
        Number of top stocks to display in the summary
    plot : bool, optional (default=True)
        Whether to generate a histogram of data distribution
    save_csv : bool, optional (default=False)
        Whether to save the full summary to a CSV file
    csv_path : str, optional (default='stock_data_summary.csv')
        Path where the CSV file will be saved if save_csv is True
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the summary statistics for each stock
    """
    # Extract stock indices
    stock_indices = data_dict['si']
    day_indices = data_dict['di']
    
    # Create a DataFrame with stock and day indices
    temp_df = pd.DataFrame({
        'stock_index': stock_indices,
        'day_index': day_indices
    })
    
    # Count occurrences of each stock index
    stock_counts = temp_df['stock_index'].value_counts().sort_values(ascending=False)
    
    # Create a summary dataframe
    stock_summary = pd.DataFrame({
        'stock_index': stock_counts.index,
        'num_days': stock_counts.values
    })
    
    # Add date range information
    date_ranges = {}
    for stock_idx in stock_summary['stock_index']:
        stock_days = temp_df[temp_df['stock_index'] == stock_idx]['day_index']
        date_ranges[stock_idx] = (stock_days.min(), stock_days.max())
    
    stock_summary['first_day'] = [date_ranges[idx][0] for idx in stock_summary['stock_index']]
    stock_summary['last_day'] = [date_ranges[idx][1] for idx in stock_summary['stock_index']]
    stock_summary['potential_days'] = stock_summary['last_day'] - stock_summary['first_day'] + 1
    stock_summary['coverage_pct'] = (stock_summary['num_days'] / stock_summary['potential_days'] * 100).round(2)
    
    # Display the first few rows of the summary
    print(f"Top {top_n} stocks by data availability:")
    print(stock_summary.head(top_n))
    
    # Add some basic statistics
    print("\nSummary Statistics:")
    print(f"Total number of stocks: {len(stock_summary)}")
    print(f"Average days per stock: {stock_counts.mean():.2f}")
    print(f"Median days per stock: {stock_counts.median():.2f}")
    print(f"Min days per stock: {stock_counts.min()}")
    print(f"Max days per stock: {stock_counts.max()}")
    
    # Create a histogram to visualize the distribution
    if plot:
        plt.figure(figsize=(10, 6))
        plt.hist(stock_counts, bins=30)
        plt.title('Distribution of Data Points per Stock')
        plt.xlabel('Number of Days')
        plt.ylabel('Number of Stocks')
        plt.grid(alpha=0.3)
        plt.show()
    
    # Save to CSV if requested
    if save_csv:
        stock_summary.to_csv(csv_path, index=False)
        print(f"Full summary saved to {csv_path}")
    
    return stock_summary

def merge_datasets(file_path1, file_path2):
    """
    Merge two dataset files into a single dataset.
    
    Parameters:
    -----------
    file_path1 : str
        Path to the first dataset file (.npy)
    file_path2 : str
        Path to the second dataset file (.npy)
        
    Returns:
    --------
    dict
        Merged data dictionary with combined data from both files
    """
    print("Loading first dataset...")
    data_dict1 = load_data(file_path1)
    print("Loading second dataset...")
    data_dict2 = load_data(file_path2)
    
    print("Merging datasets...")
    merged_dict = {}
    
    # Merge x_data (alpha signals)
    # Each file has 200 signals, we're combining to get 400 signals
    merged_dict['x_data'] = np.hstack((data_dict1['x_data'], data_dict2['x_data']))
    
    # For other data components, we'll use the first dataset
    # since these should be identical between the two files
    copy_keys = ['y_data', 'si', 'di', 'raw_data', 'list_of_data']
    for key in copy_keys:
        merged_dict[key] = data_dict1[key]
    
    print(f"Merged dataset created with {merged_dict['x_data'].shape[1]} alpha signals")
    return merged_dict

def save_merged_dataset(merged_dict, output_path='./merged_dataset.npy'):
    """
    Save the merged dataset to a file.
    
    Parameters:
    -----------
    merged_dict : dict
        The merged dataset dictionary
    output_path : str
        Path where the merged dataset will be saved
    """
    print(f"Saving merged dataset to {output_path}...")
    np.save(output_path, merged_dict)
    print("Merged dataset saved successfully!")

def merge_and_save_datasets():
    file_path1 = './data/10716 Spring 2025 Project-selected/dict_of_data_Jan2025_part1.npy'
    file_path2 = './data/10716 Spring 2025 Project-selected/dict_of_data_Jan2025_part2.npy'
    output_path = './data/10716 Spring 2025 Project-selected/merged_dataset.npy'
    
    # Merge datasets
    merged_dict = merge_datasets(file_path1, file_path2)
    
    # Preview the merged data
    print("Merged dataset preview:")
    preview_data(merged_dict)
    
    # Save the merged dataset
    save_merged_dataset(merged_dict, output_path)
    
    return merged_dict

def main():
    file_path = './data/10716 Spring 2025 Project-selected/dict_of_data_Jan2025_part1.npy'
    data_dict = load_data(file_path)
    preview_data(data_dict)
    
    # Generate stock data summary
    summarize_stock_data(data_dict, top_n=10, plot=True, save_csv=False)
    merge_and_save_datasets()
    return

if __name__ == '__main__':
    main()