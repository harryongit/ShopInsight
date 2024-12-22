# tests/test_data_processing.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from src.data_processing import DataProcessor

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    return pd.DataFrame({
        'order_id': [1, 2, 3, 4, 5],
        'customer_id': ['A1', 'A2', 'A3', 'A2', 'A1'],
        'product_id': ['P1', 'P2', 'P1', 'P3', 'P2'],
        'order_date': ['2024-01-01', '2024-01-02', '2024-01-02', '2024-01-03', '2024-01-03'],
        'price': [100.0, np.nan, 150.0, 200.0, 120.0],
        'quantity': [2, 1, 2, np.nan, 1],
        'category': ['Electronics', 'Clothing', None, 'Electronics', 'Clothing']
    })

def test_handle_missing_values(sample_data):
    processor = DataProcessor(sample_data)
    strategy = {
        'price': 'median',
        'quantity': 'mean',
        'category': 'mode'
    }
    processed_df = processor.handle_missing_values(strategy)
    
    assert processed_df['price'].isnull().sum() == 0
    assert processed_df['quantity'].isnull().sum() == 0
    assert processed_df['category'].isnull().sum() == 0

def test_handle_duplicates(sample_data):
    # Add duplicate row
    duplicate_data = pd.concat([sample_data, sample_data.iloc[[0]]])
    processor = DataProcessor(duplicate_data)
    processed_df = processor.handle_duplicates()
    
    assert len(processed_df) == len(sample_data)

def test_handle_outliers(sample_data):
    # Add outlier
    sample_data.loc[len(sample_data)] = [6, 'A4', 'P1', '2024-01-04', 1000.0, 1, 'Electronics']
    processor = DataProcessor(sample_data)
    processed_df = processor.handle_outliers(columns=['price'], method='iqr')
    
    assert len(processed_df) < len(sample_data)

def test_format_dates(sample_data):
    processor = DataProcessor(sample_data)
    processed_df = processor.format_dates(['order_date'])
    
    assert pd.api.types.is_datetime64_any_dtype(processed_df['order_date'])

