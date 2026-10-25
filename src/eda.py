import pandas as pd
import plotly.express as px

# Load dataset
df = pd.read_csv("data/raw/creditcard.csv")

# Count legitimate and fraudulent transactions
class_counts = df["Class"].value_counts()

# Create a simple table for the graph
graph_data = pd.DataFrame({
    "Transaction Type": ["Legitimate", "Fraud"],
    "Number of Transactions": [
        class_counts.get(0, 0),
        class_counts.get(1, 0)
    ]
})

# Create bar chart
fig = px.bar(
    graph_data,
    x="Transaction Type",
    y="Number of Transactions",
    title="Legitimate vs Fraudulent Transactions"
)

# Save graph
fig.write_html("visualizations/class_distribution.html")

print("Graph saved successfully!")