from mcp.server.fastmcp import FastMCP
import base64
import io
# Initialize our MCP Server with a name
mcp = FastMCP("Data Science Agent")

# ── Tool stubs (we'll fill these in one by one) ──────────────────

@mcp.tool()
def describe_dataset(file_path: str) -> str:
    """Inspect a CSV dataset and return summary statistics."""
    import pandas as pd
    import json

    df = pd.read_csv(file_path)

    info = {
        "shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_summary": json.loads(
            df.describe().round(2).to_json()
        ),
        "sample_rows": json.loads(
            df.head(3).to_json(orient="records")
        ),
    }

    return json.dumps(info, indent=2)

@mcp.tool()
def run_python_analysis(file_path: str, target_column: str) -> str:
    """Run statistical analysis and correlations on a dataset."""
    import pandas as pd
    import json

    df = pd.read_csv(file_path)

    # Only work with numeric columns
    numeric_df = df.select_dtypes(include="number")

    if target_column not in numeric_df.columns:
        return json.dumps({"error": f"'{target_column}' is not a numeric column."})

    # Correlation of all columns with the target
    correlations = (
        numeric_df.corr()[target_column]
        .drop(target_column)
        .round(4)
        .sort_values(ascending=False)
        .to_dict()
    )

    # Full correlation matrix
    corr_matrix = numeric_df.corr().round(4).to_dict()

    # Per-column stats
    stats = numeric_df.describe().round(2).to_dict()

    result = {
        "target_column": target_column,
        "correlations_with_target": correlations,
        "interpretation": {
            col: (
                "strong positive" if val > 0.7
                else "moderate positive" if val > 0.4
                else "weak positive" if val > 0
                else "weak negative" if val > -0.4
                else "moderate negative" if val > -0.7
                else "strong negative"
            )
            for col, val in correlations.items()
        },
        "correlation_matrix": corr_matrix,
        "column_stats": stats,
    }

    return json.dumps(result, indent=2)

@mcp.tool()
def generate_chart(file_path: str, chart_type: str, x_col: str, y_col: str) -> str:
    """Generate and return a chart as base64 encoded image."""
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import json
    import base64
    import io

    df = pd.read_csv(file_path)

    if chart_type == "correlation_heatmap":
        output_path = "outputs/correlation_heatmap.png"
    else:
        output_path = f"outputs/{chart_type}_{x_col}_vs_{y_col}.png"

    plt.figure(figsize=(10, 6))

    if chart_type == "scatter":
        sns.scatterplot(data=df, x=x_col, y=y_col, color="steelblue", s=100)
        plt.title(f"{x_col} vs {y_col}", fontsize=14)
        plt.xlabel(x_col)
        plt.ylabel(y_col)

    elif chart_type == "bar":
        sns.barplot(data=df, x=x_col, y=y_col, color="steelblue")
        plt.title(f"{y_col} by {x_col}", fontsize=14)
        plt.xticks(rotation=45)
        plt.xlabel(x_col)
        plt.ylabel(y_col)

    elif chart_type == "correlation_heatmap":
        numeric_df = df.select_dtypes(include="number")
        corr = numeric_df.corr().round(2)
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Correlation Heatmap", fontsize=14)

    else:
        return json.dumps({"error": f"Unknown chart type: {chart_type}"})

    plt.tight_layout()

    # Save to bytes instead of disk
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    plt.close()
    buf.seek(0)
    img_bytes = buf.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    return json.dumps({
        "status": "Chart generated successfully",
        "chart_type": chart_type,
        "output_path": output_path,
        "image_b64": img_b64
    })

@mcp.tool()
def train_ml_model(file_path: str, target_column: str) -> str:
    """Train a Random Forest model and return performance metrics."""
    import pandas as pd
    import json
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_csv(file_path)

    # Encode any text columns automatically
    for col in df.select_dtypes(include="object").columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    if target_column not in df.columns:
        return json.dumps({"error": f"Column '{target_column}' not found."})

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)
    r2 = round(r2_score(y_test, predictions), 4)
    mae = round(mean_absolute_error(y_test, predictions), 2)

    # Interpret R2 score
    if r2 >= 0.9:
        quality = "Excellent"
    elif r2 >= 0.75:
        quality = "Good"
    elif r2 >= 0.5:
        quality = "Fair"
    else:
        quality = "Poor - more data may be needed"

    result = {
        "model": "Random Forest Regressor",
        "target_column": target_column,
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "features_used": list(X.columns),
        "metrics": {
            "r2_score": r2,
            "mean_absolute_error": mae,
            "model_quality": quality
        },
        "interpretation": f"Model explains {round(r2 * 100, 1)}% of variance in {target_column}"
    }

    return json.dumps(result, indent=2)

@mcp.tool()
def feature_importance(file_path: str, target_column: str) -> str:
    """Return ranked feature importances driving the target variable."""
    import pandas as pd
    import matplotlib.pyplot as plt
    import json
    import os
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_csv(file_path)

    # Encode text columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    if target_column not in df.columns:
        return json.dumps({"error": f"Column '{target_column}' not found."})

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Get feature importances
    importances = pd.Series(model.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False).round(4)

    # Convert to percentage
    importance_pct = (importances * 100).round(2).to_dict()

    # Rank features
    ranked = [
        {
            "rank": i + 1,
            "feature": col,
            "importance_percent": pct,
            "impact": (
                "High impact" if pct >= 30
                else "Medium impact" if pct >= 15
                else "Low impact"
            )
        }
        for i, (col, pct) in enumerate(importance_pct.items())
    ]

    # Save feature importance chart
    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(10, 6))
    colors = ["#2ecc71" if v >= 30 else "#f39c12" if v >= 15 else "#e74c3c"
              for v in importance_pct.values()]
    plt.barh(list(importance_pct.keys())[::-1],
             list(importance_pct.values())[::-1],
             color=colors[::-1])
    plt.xlabel("Importance (%)")
    plt.title(f"Feature Importance for '{target_column}'", fontsize=14)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    plt.close()
    buf.seek(0)
    img_bytes = buf.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    result = {
        "target_column": target_column,
        "ranked_features": ranked,
        "top_driver": ranked[0]["feature"],
        "chart_saved": f"outputs/feature_importance_{target_column}.png",
        "image_b64": img_b64,
        "insight": f"'{ranked[0]['feature']}' is the #1 driver of {target_column} "
                   f"with {ranked[0]['importance_percent']}% importance"
    }

    return json.dumps(result, indent=2)

# ── Run the server ─────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()