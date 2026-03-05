import sys
sys.path.insert(0, "mcp_server")
from mcp_server.server import describe_dataset, feature_importance, generate_chart, run_python_analysis, generate_chart, train_ml_model

# result = describe_dataset("uploads/test_data.csv")
# print(result)

# result2 = run_python_analysis("uploads/test_data.csv", "revenue")
# print(result2)


# # Test 1 - Scatter plot
# print(generate_chart("uploads/test_data.csv", "scatter", "marketing_spend", "revenue"))

# # Test 2 - Bar chart
# print(generate_chart("uploads/test_data.csv", "bar", "month", "revenue"))

# # Test 3 - Heatmap
# print(generate_chart("uploads/test_data.csv", "correlation_heatmap", "", ""))

# result = train_ml_model("uploads/test_data.csv", "revenue")
# print(result)

result = feature_importance("uploads/test_data.csv", "revenue")
print(result)