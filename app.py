
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pulp

st.set_page_config(
    page_title="AI-Based Water Distribution Optimization",
    page_icon="💧",
    layout="wide"
)

st.title("AI-Based Water Distribution Optimization")

SHORTAGE_PENALTY = 1000

model = joblib.load("models/random_forest_demand_model.pkl")
features = joblib.load("models/model_features.pkl")

dashboard_input = pd.read_csv("data/dashboard_input_features.csv")
network_links = pd.read_csv("data/water_network_links.csv")

st.sidebar.header("Optimization Settings")

scenario = st.sidebar.selectbox(
    "Select Demand Scenario",
    ["Normal Demand", "High Demand", "Low Demand"]
)

scenario_factors = {
    "Normal Demand": 1.00,
    "High Demand": 1.20,
    "Low Demand": 0.80
}

demand_factor = scenario_factors[scenario]

st.sidebar.write("Demand factor:", demand_factor)

prediction_input = dashboard_input[features].copy()

predicted_values = model.predict(prediction_input)

demand_data = pd.DataFrame({
    "nodeID": dashboard_input["nodeID"],
    "predicted_demand": predicted_values
})

demand_data["predicted_demand"] = (
    demand_data["predicted_demand"] * demand_factor
)

demand_data["predicted_demand"] = demand_data["predicted_demand"].clip(lower=0)

st.success("Demand prediction completed successfully.")

st.subheader("Predicted Water Demand")

st.dataframe(
    demand_data,
    width="stretch"
)

total_demand = demand_data["predicted_demand"].sum()

st.metric(
    "Total Predicted Demand",
    f"{total_demand:.4f}"
)

node_demand = dict(
    zip(
        demand_data["nodeID"].astype(str),
        demand_data["predicted_demand"]
    )
)

links = network_links.copy()

links["source"] = links["source"].astype(str)
links["target"] = links["target"].astype(str)

nodes = set(node_demand.keys())

problem = pulp.LpProblem(
    "Water_Distribution_Optimization",
    pulp.LpMinimize
)

flow_variables = {}

for _, row in links.iterrows():
    link_id = str(row["linkID"])

    flow_variables[link_id] = pulp.LpVariable(
        f"flow_{link_id}",
        lowBound=0,
        upBound=float(row["capacity"])
    )

shortage_variables = {}

for node in nodes:
    shortage_variables[node] = pulp.LpVariable(
        f"shortage_{node}",
        lowBound=0,
        upBound=float(node_demand[node])
    )

transport_cost = []

for _, row in links.iterrows():
    link_id = str(row["linkID"])

    transport_cost.append(
        float(row["transport_cost"]) *
        flow_variables[link_id]
    )

shortage_cost = []

for node in nodes:
    shortage_cost.append(
        SHORTAGE_PENALTY *
        shortage_variables[node]
    )

problem += (
    pulp.lpSum(transport_cost) +
    pulp.lpSum(shortage_cost)
)

for node in nodes:

    incoming_flow = []

    outgoing_flow = []

    for _, row in links.iterrows():

        link_id = str(row["linkID"])

        if str(row["target"]) == node:
            incoming_flow.append(flow_variables[link_id])

        if str(row["source"]) == node:
            outgoing_flow.append(flow_variables[link_id])

    problem += (
        pulp.lpSum(incoming_flow)
        - pulp.lpSum(outgoing_flow)
        + shortage_variables[node]
        == float(node_demand[node])
    )

solver = pulp.HiGHS(msg=False)

problem.solve(solver)

status = pulp.LpStatus[problem.status]

st.subheader("Optimization Result")

st.write(f"Optimization Status: **{status}**")

if status == "Optimal":

    optimized_flows = []

    for _, row in links.iterrows():

        link_id = str(row["linkID"])

        optimized_flows.append({
            "linkID": row["linkID"],
            "source": row["source"],
            "target": row["target"],
            "capacity": row["capacity"],
            "transport_cost": row["transport_cost"],
            "optimized_flow": flow_variables[link_id].value()
        })

    optimized_flow_df = pd.DataFrame(optimized_flows)

    shortage_data = []

    for node in demand_data["nodeID"].astype(str):

        shortage_value = shortage_variables[node].value()

        if shortage_value is None:
            shortage_value = 0

        demand_value = node_demand[node]

        supplied_value = demand_value - shortage_value

        shortage_data.append({
            "nodeID": node,
            "predicted_demand": demand_value,
            "shortage": shortage_value,
            "supplied_water": supplied_value
        })

    node_result_df = pd.DataFrame(shortage_data)

    total_shortage = node_result_df["shortage"].sum()

    total_supplied = node_result_df["supplied_water"].sum()

    if total_demand > 0:
        satisfaction = (
            total_supplied / total_demand
        ) * 100
    else:
        satisfaction = 0

    st.success(
        "Water distribution optimization completed successfully."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Demand",
        f"{total_demand:.4f}"
    )

    col2.metric(
        "Water Supplied",
        f"{total_supplied:.4f}"
    )

    col3.metric(
        "Total Shortage",
        f"{total_shortage:.4f}"
    )

    col4.metric(
        "Demand Satisfaction",
        f"{satisfaction:.2f}%"
    )

    st.subheader("Optimized Water Flow")

    st.dataframe(
        optimized_flow_df,
        width="stretch"
    )

    st.subheader("Node-wise Water Supply")

    st.dataframe(
        node_result_df,
        width="stretch"
    )

    st.subheader("Optimized Flow by Link")

    chart_data = optimized_flow_df[
        ["linkID", "optimized_flow"]
    ].copy()

    chart_data = chart_data.set_index("linkID")

    st.bar_chart(chart_data)

    st.subheader("Node-wise Demand vs Supplied Water")

    node_chart = node_result_df[
        ["nodeID", "predicted_demand", "supplied_water"]
    ].copy()

    node_chart = node_chart.set_index("nodeID")

    st.bar_chart(node_chart)

else:

    st.error(
        "Optimization could not find an optimal solution."
    )

st.subheader("Current Scenario")

scenario_summary = pd.DataFrame({
    "Scenario": [scenario],
    "Demand Factor": [demand_factor],
    "Total Demand": [total_demand]
})

st.dataframe(
    scenario_summary,
    width="stretch"
)
