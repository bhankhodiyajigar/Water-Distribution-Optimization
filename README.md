# AI-Based Water Distribution Optimization

## Project Overview

This project uses Artificial Intelligence and Linear Programming to optimize water distribution in a network.

The system predicts water demand at different nodes using a Random Forest regression model. The predicted demand is then used by a Linear Programming optimization model to determine how water should be distributed through the network.

## Project Workflow

1. Water demand data preparation
2. Feature engineering
3. Demand prediction using Random Forest
4. Water distribution network design
5. Linear Programming optimization
6. Shortage minimization
7. Scenario analysis
8. Streamlit dashboard

## Machine Learning

The project uses a Random Forest regression model for water demand prediction.

The model uses:

- Lagged demand
- Rolling mean demand
- Node position
- Hour-based cyclic features
- Day-based cyclic features

The trained model is stored in:

models/random_forest_demand_model.pkl
## Optimization

The water distribution problem is formulated as a Linear Programming problem.

The optimization considers:

- Water flow through network links
- Link capacity limits
- Node water demand
- Water shortage
- Transportation cost

The objective is to minimize transportation cost and shortage penalty.

## Scenario Analysis

The dashboard supports three demand scenarios:

- Low Demand: 0.80
- Normal Demand: 1.00
- High Demand: 1.20

## Results

The project evaluates:

- Predicted water demand
- Optimized water flow
- Total water supplied
- Total water shortage
- Demand satisfaction
- Different demand scenarios

The tested normal-demand scenario achieved approximately 60.8% demand satisfaction.

## Streamlit Dashboard

The project includes an interactive Streamlit dashboard.

The dashboard allows the user to:

- Select a demand scenario
- View predicted demand
- Run the optimization
- View optimized water flow
- View node-wise water supply
- View demand satisfaction
- Analyze different demand scenarios
## Project Structure

AI-Water-Distribution-Optimization/
├── app.py
├── requirements.txt
├── data/
│   ├── dashboard_input_features.csv
│   ├── node_predicted_demand.csv
│   └── water_network_links.csv
├── models/
│   ├── model_features.pkl
│   └── random_forest_demand_model.pkl
└── results/
    ├── forecast_model_comparison.csv
    ├── normal_vs_optimized_flow.csv
    ├── optimal_water_flows.csv
    ├── scenario_optimization_results.csv
    └── water_demand_satisfaction.csv

## Installation

Create and activate a virtual environment:

python3 -m venv venv
source venv/bin/activate

Install the required libraries:

pip install -r requirements.txt

## Run the Dashboard

streamlit run app.py

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest
- PuLP
- HiGHS
- Streamlit
- Matplotlib
- Joblib

## Project Type

CMAI Optimization Project

AI-Based Water Distribution Optimization
