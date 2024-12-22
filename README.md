# Ecom Data Analysis

## Project Overview
The **Ecom Data Analysis** project is a comprehensive framework designed to analyze and gain actionable insights from e-commerce sales data. It aims to address key business questions like identifying customer behavior patterns, forecasting sales trends, and optimizing marketing strategies. The project covers the entire data analysis pipeline, including data cleaning, exploratory analysis, customer segmentation, and predictive modeling for sales forecasting. 

This project structure ensures modularity, scalability, and reproducibility, making it a robust framework suitable for both academic research and real-world business applications.

![ecommerce-01](https://github.com/user-attachments/assets/a2a32541-7fd0-48b9-a7ab-2a3b690e6d88)

## Directory Structure

```
ecom_data_analysis/
├── data/
│   ├── raw/
│   │   └── sales_data.csv
│   └── processed/
│       └── cleaned_sales_data.csv
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_customer_segmentation.ipynb
│   └── 04_sales_forecasting.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── visualization.py
│   ├── model_utils.py
│   └── config.py
│
├── tests/
│   ├── __init__.py
│   ├── test_data_processing.py
│   └── test_model_utils.py
│
├── requirements.txt
├── README.md
├── setup.py
└── .gitignore
```

### Description of Components

#### **Data Directory**
- **raw/**: Contains the original, unprocessed sales data (`sales_data.csv`).
- **processed/**: Stores cleaned and transformed datasets (`cleaned_sales_data.csv`) that are ready for analysis and modeling.

#### **Notebooks**
1. **01_data_cleaning.ipynb**: Explains the step-by-step process of cleaning the raw data, such as removing duplicates, handling missing values, and dealing with outliers.
2. **02_exploratory_analysis.ipynb**: Offers visual insights into key metrics like sales trends, top-performing products, and customer demographics using data visualization techniques.
3. **03_customer_segmentation.ipynb**: Implements clustering algorithms (e.g., K-Means, DBSCAN) to group customers into meaningful segments based on their purchasing behavior.
4. **04_sales_forecasting.ipynb**: Develops predictive models using machine learning (e.g., ARIMA, Random Forest) to forecast future sales and revenue trends.

#### **Source Code (src/)**
- **data_processing.py**: Contains reusable functions for data cleaning and preprocessing.
- **feature_engineering.py**: Includes methods to create and optimize new features to improve machine learning models.
- **visualization.py**: Offers functions to generate detailed and interactive plots for better data interpretation.
- **model_utils.py**: Provides utilities for model training, evaluation, and saving/loading models.
- **config.py**: Centralized configuration file for defining global variables and settings such as file paths and hyperparameters.

#### **Tests**
- **test_data_processing.py**: Validates the correctness of data cleaning and preprocessing functions.
- **test_model_utils.py**: Ensures the reliability of model training and evaluation methods.

---

## Installation

### Prerequisites
1. Ensure Python 3.7+ is installed on your system.
2. Install a virtual environment (recommended).

### Steps to Install
1. Clone the repository:
   ```bash
   git clone https://github.com/harryongit/ecom_data_analysis.git
   cd ecom_data_analysis
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows, use `env\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the setup script to install the project package:
   ```bash
   python setup.py install
   ```

---

## Usage Instructions

### Data Preparation
1. Place your raw sales data in the `data/raw/` directory as `sales_data.csv`.
2. Run the `01_data_cleaning.ipynb` notebook to process and clean the data. The cleaned data will be saved in the `data/processed/` directory.

### Analysis and Modeling
- **Exploratory Analysis**: Use the `02_exploratory_analysis.ipynb` notebook to understand key trends and patterns in the data.
- **Customer Segmentation**: Segment your customers using clustering techniques in the `03_customer_segmentation.ipynb` notebook.
- **Sales Forecasting**: Predict future sales trends using machine learning models in the `04_sales_forecasting.ipynb` notebook.

### Running Tests
To ensure all components work as expected, run:
```bash
pytest tests/
```

---

## Requirements
This project relies on the following key libraries:
- **pandas**: For data manipulation and analysis.
- **numpy**: For numerical computations.
- **matplotlib & seaborn**: For data visualization.
- **scikit-learn**: For clustering and machine learning models.
- **statsmodels**: For time series forecasting.

For a full list of dependencies, see `requirements.txt`.

---

## Contribution Guidelines
We welcome contributions to enhance the functionality of this project. Follow these steps to contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature-name
   ```
3. Make your changes and test thoroughly.
4. Submit a pull request with a detailed description of your changes.

### Development Notes
- Adhere to PEP 8 coding standards.
- Ensure that all tests pass before creating a pull request.

---

## License
This project is licensed under the MIT License. For more details, see the `LICENSE` file.

---

## Acknowledgments
This project is made possible by the contributions of the open-source community and the developers of the libraries and tools used within this framework.

---

## Future Work
- Add advanced customer lifetime value (CLV) analysis.
- Integrate additional machine learning algorithms for more robust forecasting.
- Develop a dashboard for real-time visualization of sales metrics.
