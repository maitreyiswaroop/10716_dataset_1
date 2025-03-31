def analyze_stock_characteristics(data_dict):
    """
    Analyze characteristics of stocks in the dataset using raw_data.
    """
    # Extract stock indices and raw data
    si = data_dict['si']
    raw_data = data_dict['raw_data']
    list_of_data = data_dict['list_of_data']
    
    print("Analyzing stock characteristics...")
    print(f"Total number of data points: {len(si)}")
    print(f"Total number of unique stocks: {len(np.unique(si))}")
    print(f"Raw data type: {type(raw_data)}")
    
    # Determine the indices of 'sector' and 'industry' in list_of_data
    try:
        sector_idx = list_of_data.index('sector')
        industry_idx = list_of_data.index('industry')
    except ValueError as e:
        raise ValueError(f"Required fields 'sector' or 'industry' not found in list_of_data: {e}")
    
    unique_stocks = np.unique(si)
    
    # Create a mapping of stock index to sector/industry
    stock_info = {}
    for stock_idx in unique_stocks:
        # Find first occurrence of this stock
        first_idx = np.where(si == stock_idx)[0][0]
        
        # Extract sector and industry information using their indices
        sector = raw_data[first_idx, sector_idx]
        industry = raw_data[first_idx, industry_idx]
        num_observations = np.sum(si == stock_idx)
        
        stock_info[stock_idx] = {
            'sector': sector,
            'industry': industry,
            'num_observations': num_observations
        }
    
    # Convert to DataFrame and sort by number of observations
    df = pd.DataFrame.from_dict(stock_info, orient='index')
    df = df.sort_values('num_observations', ascending=False)
    
    # Display summary statistics
    print("\nStock Universe Composition:")
    print(f"\nTotal number of unique stocks: {len(df)}")
    print(f"Average observations per stock: {df['num_observations'].mean():.2f}")
    
    print("\nSector Distribution:")
    sector_dist = df['sector'].value_counts()
    print(sector_dist)
    
    print("\nTop 10 Industries:")
    print(df['industry'].value_counts().head(10))
    
    # Create visualizations
    plt.figure(figsize=(12, 6))
    sector_dist.plot(kind='bar')
    plt.title('Distribution of Stocks by Sector')
    plt.xlabel('Sector')
    plt.ylabel('Number of Stocks')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    return df