# src/feature_engineering.py
import pandas as pd
import numpy as np
from typing import List, Dict
from datetime import datetime
from src.config import DATE_FORMAT

class FeatureEngineer:
    """Class for creating and transforming features"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize feature engineering class
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame with raw features
        """
        self.df = df.copy()
        self.original_columns = df.columns.tolist()
        
    def create_time_features(self) -> pd.DataFrame:
        """Create time-based features from date columns"""
        
        if 'order_date' in self.df.columns:
            self.df['order_hour'] = self.df['order_date'].dt.hour
            self.df['order_day'] = self.df['order_date'].dt.day
            self.df['order_month'] = self.df['order_date'].dt.month
            self.df['order_year'] = self.df['order_date'].dt.year
            self.df['order_dayofweek'] = self.df['order_date'].dt.dayofweek
            self.df['order_quarter'] = self.df['order_date'].dt.quarter
            self.df['is_weekend'] = self.df['order_dayofweek'].isin([5, 6]).astype(int)
            
        return self.df
    
    def create_customer_features(self) -> pd.DataFrame:
        """Create customer-related features"""
        
        # Customer purchase history
        customer_history = self.df.groupby('customer_id').agg({
            'order_id': 'count',
            'price': ['sum', 'mean', 'std'],
            'order_date': ['min', 'max']
        })
        
        customer_history.columns = [
            'total_orders',
            'total_spent',
            'avg_order_value',
            'std_order_value',
            'first_order_date',
            'last_order_date'
        ]
        
        # Calculate customer lifetime value
        customer_history['customer_lifetime_days'] = (
            customer_history['last_order_date'] - 
            customer_history['first_order_date']
        ).dt.days
        
        customer_history['purchase_frequency'] = (
            customer_history['total_orders'] / 
            customer_history['customer_lifetime_days']
        )
        
        # Merge features back to main DataFrame
        self.df = self.df.merge(
            customer_history,
            left_on='customer_id',
            right_index=True,
            how='left'
        )
        
        return self.df
    
    def create_product_features(self) -> pd.DataFrame:
        """Create product-related features"""
        
        if 'product_id' in self.df.columns:
            # Product popularity metrics
            product_metrics = self.df.groupby('product_id').agg({
                'order_id': 'count',
                'price': ['mean', 'std']
            })
            
            product_metrics.columns = [
                'product_order_count',
                'product_avg_price',
                'product_price_std'
            ]
            
            # Create price segments
            self.df['price_segment'] = pd.qcut(
                self.df['price'],
                q=4,
                labels=['Budget', 'Economy', 'Premium', 'Luxury']
            )
            
            # Merge product metrics
            self.df = self.df.merge(
                product_metrics,
                left_on='product_id',
                right_index=True,
                how='left'
            )
        
        return self.df
    
    def create_interaction_features(self) -> pd.DataFrame:
        """Create interaction features between existing features"""
        
        if 'price' in self.df.columns and 'quantity' in self.df.columns:
            self.df['total_item_value'] = self.df['price'] * self.df['quantity']
        
        if 'product_order_count' in self.df.columns and 'total_orders' in self.df.columns:
            self.df['product_customer_ratio'] = (
                self.df['product_order_count'] / self.df['total_orders']
            )
        
        return self.df
    
    def handle_categorical_features(self, method: str = 'label') -> pd.DataFrame:
        """
        Encode categorical features
        
        Parameters:
        -----------
        method : str
            Encoding method ('label' or 'onehot')
        """
        categorical_columns = self.df.select_dtypes(include=['object']).columns
        
        if method == 'label':
            for col in categorical_columns:
                self.df[f'{col}_encoded'] = pd.factorize(self.df[col])[0]
                
        elif method == 'onehot':
            self.df = pd.get_dummies(
                self.df,
                columns=categorical_columns,
                prefix=categorical_columns
            )
        
        return self.df
    
    def create_all_features(self, categorical_method: str = 'label') -> pd.DataFrame:
        """Create all available features"""
        
        self.df = (self.df
                  .pipe(self.create_time_features)
                  .pipe(self.create_customer_features)
                  .pipe(self.create_product_features)
                  .pipe(self.create_interaction_features)
                  .pipe(self.handle_categorical_features, categorical_method))
        
        return self.df
    
    def get_feature_names(self) -> Dict[str, List[str]]:
        """Get names of created features by category"""
        
        new_columns = set(self.df.columns) - set(self.original_columns)
        
        feature_names = {
            'time_features': [col for col in new_columns if 'order_' in col],
            'customer_features': [col for col in new_columns if 'customer_' in col or 'total_' in col],
            'product_features': [col for col in new_columns if 'product_' in col],
            'interaction_features': [col for col in new_columns if '_ratio' in col],
            'encoded_features': [col for col in new_columns if '_encoded' in col or any(x in col for x in ['_0', '_1'])]
        }
        
        return feature_names
