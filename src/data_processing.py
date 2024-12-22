# src/data_processing.py
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
from datetime import datetime
from src.config import DATE_FORMAT, PROCESSED_DATA_DIR

class DataProcessor:
    """Class for handling all data processing operations"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize data processor
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw input DataFrame
        """
        self.raw_data = df.copy()
        self.processed_data = None
        
    def check_missing_values(self) -> Dict[str, float]:
        """Calculate missing value percentages for each column"""
        
        missing_percentages = (self.raw_data.isnull().sum() / len(self.raw_data) * 100)
        return missing_percentages.to_dict()
    
    def handle_missing_values(self, strategy: Dict[str, str]) -> pd.DataFrame:
        """
        Handle missing values according to specified strategy
        
        Parameters:
        -----------
        strategy : Dict[str, str]
            Dictionary mapping column names to imputation strategy
            ('mean', 'median', 'mode', 'drop', 'zero')
        """
        df = self.raw_data.copy()
        
        for column, method in strategy.items():
            if column in df.columns:
                if method == 'mean':
                    df[column] = df[column].fillna(df[column].mean())
                elif method == 'median':
                    df[column] = df[column].fillna(df[column].median())
                elif method == 'mode':
                    df[column] = df[column].fillna(df[column].mode()[0])
                elif method == 'zero':
                    df[column] = df[column].fillna(0)
                elif method == 'drop':
                    df = df.dropna(subset=[column])
                    
        self.processed_data = df
        return df
    
    def handle_duplicates(self, subset: Optional[List[str]] = None) -> pd.DataFrame:
        """Remove duplicate rows based on specified columns"""
        
        if self.processed_data is None:
            self.processed_data = self.raw_data.copy()
            
        self.processed_data = self.processed_data.drop_duplicates(subset=subset)
        return self.processed_data
    
    def handle_outliers(self, 
                       columns: List[str],
                       method: str = 'iqr',
                       threshold: float = 1.5) -> pd.DataFrame:
        """
        Handle outliers in specified columns
        
        Parameters:
        -----------
        columns : List[str]
            Columns to check for outliers
        method : str
            Method to use ('iqr' or 'zscore')
        threshold : float
            Threshold for outlier detection
        """
        if self.processed_data is None:
            self.processed_data = self.raw_data.copy()
            
        df = self.processed_data.copy()
        
        for column in columns:
            if column in df.columns and df[column].dtype in ['int64', 'float64']:
                if method == 'iqr':
                    Q1 = df[column].quantile(0.25)
                    Q3 = df[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    
                    df = df[
                        (df[column] >= lower_bound) & 
                        (df[column] <= upper_bound)
                    ]
                    
                elif method == 'zscore':
                    z_scores = np.ab
