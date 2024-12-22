# src/model_utils.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from typing import Tuple, Dict, Any
from src.config import CLUSTERING_CONFIG, FORECAST_CONFIG

class ModelUtils:
    """Class for all modeling operations"""
    
    def __init__(self, df: pd.DataFrame):
        self.data = df
        self.scaler = StandardScaler()
        
    def prepare_clustering_features(self,
                                  features: List[str]) -> np.ndarray:
        """Prepare features for clustering"""
        X = self.data[features].copy()
        return self.scaler.fit_transform(X)
    
    def perform_customer_segmentation(self,
                                    features: List[str],
                                    n_clusters: int = CLUSTERING_CONFIG['n_clusters']) -> Tuple[pd.DataFrame, Dict]:
        """
        Perform customer segmentation using K-means
        
        Parameters:
        -----------
        features : List[str]
            List of features to use for clustering
        n_clusters : int
            Number of clusters
            
        Returns:
        --------
        Tuple containing segmented DataFrame and metrics
        """
        # Prepare features
        X = self.prepare_clustering_features(features)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters,
                       random_state=CLUSTERING_CONFIG['random_state'])
        clusters = kmeans.fit_predict(X)
        
        # Add clusters to DataFrame
        result_df = self.data.copy()
        result_df['Segment'] = clusters
        
        # Calculate metrics
        metrics = {
            'silhouette_score': silhouette_score(X, clusters),
            'inertia': kmeans.inertia_,
            'cluster_sizes': pd.Series(clusters).value_counts().to_dict()
        }
        
        return result_df, metrics
    
    def create_sales_forecast(self,
                            target_col: str = 'price',
                            freq: str = 'D',
                            periods: int = FORECAST_CONFIG['periods']) -> Tuple[pd.Series, Dict]:
        """
        Create time series forecast using Holt-Winters method
        
        Parameters:
        -----------
        target_col : str
            Column to forecast
        freq : str
            Frequency of time series
        periods : int
            Number of periods to forecast
            
        Returns:
        --------
        Tuple containing forecast and metrics
        """
        # Prepare time series data
        ts_data = self.data.groupby('order_date')[target_col].sum()
        ts_data = ts_data.resample(freq).sum().fillna(0)
        
        # Split data
        train_size = int(len(ts_data) * 0.8)
        train = ts_data[:train_size]
        test = ts_data[train_size:]
        
        # Fit model
        model = ExponentialSmoothing(
            train,
            seasonal_periods=FORECAST_CONFIG['seasonal_periods'],
            trend='add',
            seasonal='add'
        )
        fitted_model = model.fit()
        
        # Make forecast
        forecast = fitted_model.forecast(periods)
        
        # Calculate metrics
        metrics = {
            'mse': np.mean((test - fitted_model.forecast(len(test)))**2),
            'mae': np.mean(np.abs(test - fitted_model.forecast(len(test)))),
            'training_size': len(train),
            'test_size': len(test)
        }
        
        return forecast, metrics
    
    def calculate_customer_lifetime_value(self) -> pd.DataFrame:
        """Calculate customer lifetime value"""
        clv_data = self.data.groupby('customer_id').agg({
            'price': ['sum', 'mean', 'count'],
            'order_date': ['min', 'max']
        })
        
        clv_data.columns = [
            'total_revenue',
            'avg_order_value',
            'order_count',
            'first_purchase',
            'last_purchase'
        ]
        
        # Calculate customer age
        clv_data['customer_age_days'] = (
            clv_data['last_purchase'] - clv_data['first_purchase']
        ).dt.days
        
        # Calculate purchase frequency
        clv_data['purchase_frequency'] = (
            clv_data['order_count'] / clv_data['customer_age_days']
        )
        
        # Calculate CLV
        clv_data['customer_lifetime_value'] = (
            clv_data['avg_order_value'] *
            clv_data['purchase_frequency'] *
            365  # Annualized
        )
        
        return clv_data
    
    def perform_basket_analysis(self,
                              min_support: float = 0.01) -> pd.DataFrame:
        """Perform market basket analysis"""
        # Prepare transaction data
        transactions = self.data.groupby(['order_id', 'product_id'])['quantity'].sum().unstack().fillna(0)
        transactions = transactions.applymap(lambda x: 1 if x > 0 else 0)
        
        # Calculate support for each product pair
        n_transactions = len(transactions)
        product_pairs = []
        
        for i in range(len(transactions.columns)):
            for j in range(i + 1, len(transactions.columns)):
                product1 = transactions.columns[i]
                product2 = transactions.columns[j]
                
                support = (
                    (transactions[product1] & transactions[product2]).sum() /
                    n_transactions
                )
                
                if support >= min_support:
                    confidence_1_2 = (
                        (transactions[product1] & transactions[product2]).sum() /
                        transactions[product1].sum()
                    )
                    
                    confidence_2_1 = (
                        (transactions[product1] & transactions[product2]).sum() /
                        transactions[product2].sum()
                    )
                    
                    product_pairs.append({
                        'product1': product1,
                        'product2': product2,
                        'support': support,
                        'confidence_1_2': confidence_1_2,
                        'confidence_2_1': confidence_2_1
                    })
        
        return pd.DataFrame(product_pairs)
