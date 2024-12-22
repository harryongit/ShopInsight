# src/visualization.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Tuple, List, Dict
from src.config import VIZ_CONFIG

class DataVisualizer:
    """Class for creating all data visualizations"""
    
    def __init__(self, df: pd.DataFrame):
        self.data = df
        plt.style.use(VIZ_CONFIG['style'])
        
    def plot_sales_trends(self, 
                         time_unit: str = 'M',
                         figsize: Tuple[int, int] = VIZ_CONFIG['figsize']) -> None:
        """Plot sales trends over time"""
        sales_data = self.data.groupby(
            pd.Grouper(key='order_date', freq=time_unit)
        )['price'].sum()
        
        plt.figure(figsize=figsize)
        plt.plot(sales_data.index, sales_data.values, marker='o')
        plt.title(f'Sales Trend ({time_unit})')
        plt.xlabel('Date')
        plt.ylabel('Total Sales')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
    def plot_customer_segments(self,
                             x: str,
                             y: str,
                             hue: Optional[str] = None,
                             figsize: Tuple[int, int] = VIZ_CONFIG['figsize']) -> None:
        """Create customer segmentation scatter plot"""
        plt.figure(figsize=figsize)
        sns.scatterplot(data=self.data, x=x, y=y, hue=hue, alpha=0.6)
        plt.title('Customer Segmentation')
        plt.xlabel(x.replace('_', ' ').title())
        plt.ylabel(y.replace('_', ' ').title())
        plt.tight_layout()
        
    def create_product_analysis(self,
                              top_n: int = 10,
                              figsize: Tuple[int, int] = VIZ_CONFIG['figsize']) -> None:
        """Create product performance analysis plots"""
        # Product performance metrics
        product_metrics = self.data.groupby('product_id').agg({
            'price': ['sum', 'count'],
            'quantity': 'sum'
        }).round(2)
        
        product_metrics.columns = ['total_revenue', 'order_count', 'units_sold']
        top_products = product_metrics.nlargest(top_n, 'total_revenue')
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Revenue plot
        sns.barplot(data=top_products.reset_index(),
                   x='total_revenue',
                   y='product_id',
                   ax=ax1)
        ax1.set_title('Top Products by Revenue')
        
        # Order count plot
        sns.barplot(data=top_products.reset_index(),
                   x='order_count',
                   y='product_id',
                   ax=ax2)
        ax2.set_title('Top Products by Order Count')
        
        plt.tight_layout()
        
    def plot_customer_behavior(self,
                             metric: str = 'order_count',
                             figsize: Tuple[int, int] = VIZ_CONFIG['figsize']) -> None:
        """Analyze and plot customer behavior patterns"""
        if metric == 'order_count':
            data = self.data['customer_id'].value_counts()
            title = 'Customer Order Frequency Distribution'
        elif metric == 'order_value':
            data = self.data.groupby('customer_id')['price'].mean()
            title = 'Average Order Value Distribution'
            
        plt.figure(figsize=figsize)
        sns.histplot(data=data, bins=30)
        plt.title(title)
        plt.xlabel(metric.replace('_', ' ').title())
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
    def create_interactive_dashboard(self) -> None:
        """Create interactive dashboard using Plotly"""
        # Sales trend
        daily_sales = self.data.groupby('order_date')['price'].sum().reset_index()
        fig1 = px.line(daily_sales,
                      x='order_date',
                      y='price',
                      title='Daily Sales Trend')
        fig1.show()
        
        # Product category performance
        category_sales = self.data.groupby('category').agg({
            'price': 'sum',
            'order_id': 'count'
        }).reset_index()
        
        fig2 = go.Figure(data=[
            go.Bar(name='Revenue',
                  x=category_sales['category'],
                  y=category_sales['price']),
            go.Bar(name='Orders',
                  x=category_sales['category'],
                  y=category_sales['order_id'])
        ])
        
        fig2.update_layout(title='Category Performance',
                          barmode='group')
        fig2.show()
        
    def create_cohort_analysis(self,
                             metric: str = 'retention',
                             figsize: Tuple[int, int] = VIZ_CONFIG['figsize']) -> None:
        """Create customer cohort analysis visualization"""
        # Prepare cohort data
        self.data['cohort'] = self.data['order_date'].dt.to_period('M')
        self.data['order_month'] = self.data['order_date'].dt.to_period('M')
        
        if metric == 'retention':
            cohort_data = self.data.groupby(['cohort', 'order_month']).agg({
                'customer_id': 'nunique'
            }).reset_index()
            
            cohort_data['period_number'] = (
                cohort_data['order_month'] - cohort_data['cohort']
            ).apply(lambda x: x.n)
            
            cohort_pivot = cohort_data.pivot_table(
                index='cohort',
                columns='period_number',
                values='customer_id'
            )
            
            retention_matrix = cohort_pivot.divide(cohort_pivot[0], axis=0) * 100
            
            plt.figure(figsize=figsize)
            sns.heatmap(retention_matrix,
                       annot=True,
                       fmt='.1f',
                       cmap='YlOrRd')
            plt.title('Customer Retention by Cohort')
            plt.xlabel('Period Number')
            plt.ylabel('Cohort')
            plt.tight_layout()
