# tests/test_model_utils.py
import pytest
import pandas as pd
import numpy as np
from src.model_utils import ModelUtils

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    return pd.DataFrame({
        'order_id': range(1, 101),
        'customer_id': np.random.choice(['C1', 'C2', 'C3', 'C4', 'C5'], 100),
        'product_id': np.random.choice(['P1', 'P2', 'P3', 'P4'], 100),
        'order_date': pd.date_range(start='2024-01-01', periods=100),
        'price': np.random.uniform(50, 200, 100),
        'quantity': np.random.randint(1, 5, 100),
        'category': np.random.choice(['Electronics', 'Clothing', 'Books'], 100)
    })

def test_customer_segmentation(sample_data):
    model_utils = ModelUtils(sample_data)
    segments, metrics = model_utils.perform_customer_segmentation(n_clusters=3)
    
    assert len(segments) == len(sample_data['customer_id'].unique())
    assert 'silhouette_score' in metrics
    assert 'segment_sizes' in metrics
    assert len(metrics['segment_sizes']) == 3

def test_sales_forecast(sample_data):
    model_utils = ModelUtils(sample_data)
    forecast, metrics = model_utils.create_sales_forecast(periods=7)
    
    assert len(forecast) == 7
    assert 'mse' in metrics
    assert 'mae' in metrics

def test_customer_lifetime_value(sample_data):
    model_utils = ModelUtils(sample_data)
    clv_analysis = model_utils.analyze_customer_lifetime_value()
    
    assert len(clv_analysis) == len(sample_data['customer_id'].unique())
    assert 'clv' in clv_analysis.columns
    assert 'avg_order_value' in clv_analysis.columns
