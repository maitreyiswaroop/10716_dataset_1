# Custom batch collation function for handling variable-sized batches

def custom_collate_fn(batch):
    """
    Custom collate function that handles batches with variable-sized tensors.
    Each item in the batch should be a dictionary with keys: 'x', 'time_indices', 'stock_indices', 'y'.
    
    This function will ensure that all tensors in a batch have the same sequence length
    by truncating to the shortest sequence in the batch.
    
    Parameters:
        batch: List of dictionaries containing 'x', 'time_indices', 'stock_indices', 'y'
    
    Returns:
        Dictionary containing batched tensors
    """
    import torch
    
    # Extract all items from the batch
    x_list = [item['x'] for item in batch]
    time_indices_list = [item['time_indices'] for item in batch]
    stock_indices_list = [item['stock_indices'] for item in batch]
    y_list = [item['y'] for item in batch]
    
    # Find the minimum sequence length in this batch
    min_seq_len = min(x.shape[0] for x in x_list)
    
    # Truncate all sequences to the minimum length
    x_truncated = [x[:min_seq_len] for x in x_list]
    time_indices_truncated = [t[:min_seq_len] for t in time_indices_list]
    stock_indices_truncated = [s[:min_seq_len] for s in stock_indices_list]
    
    # Stack the truncated tensors
    x_batch = torch.stack(x_truncated, dim=0)
    time_indices_batch = torch.stack(time_indices_truncated, dim=0)
    stock_indices_batch = torch.stack(stock_indices_truncated, dim=0)
    y_batch = torch.stack(y_list, dim=0)
    
    return {
        'x': x_batch,
        'time_indices': time_indices_batch,
        'stock_indices': stock_indices_batch,
        'y': y_batch
    }