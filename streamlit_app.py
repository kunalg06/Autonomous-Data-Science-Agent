import re
import os
import sys
import json
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import anthropic

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "mcp_server"))
from mcp_server.server import (
    describe_dataset,
    run_python_analysis,
    generate_chart,
    train_ml_model,
    feature_importance
)

load_dotenv()

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Autonomous Data Science Agent",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Autonomous Data Science Agent")
st.markdown("*Upload any CSV and let AI analyze it automatically*")
st.divider()

# ── Tool definitions for Claude ───────────────────────────────
tools = [
    {
        "name": "describe_dataset",
        "description": "Inspect a CSV dataset and return summary statistics, column info, and sample rows.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to the CSV file"}
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "run_python_analysis",
        "description": "Run statistical analysis and correlations on a dataset to find relationships.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to the CSV file"},
                "target_column": {"type": "string", "description": "The column to analyze"}
            },
            "required": ["file_path", "target_column"]
        }
    },
    {
        "name": "generate_chart",
        "description": "Generate and save a chart from the dataset.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to the CSV file"},
                "chart_type": {"type": "string", "description": "scatter, bar, or correlation_heatmap"},
                "x_col": {"type": "string", "description": "Column for x axis"},
                "y_col": {"type": "string", "description": "Column for y axis"}
            },
            "required": ["file_path", "chart_type", "x_col", "y_col"]
        }
    },
    {
        "name": "train_ml_model",
        "description": "Train a Random Forest ML model and return performance metrics.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to the CSV file"},
                "target_column": {"type": "string", "description": "The target column to predict"}
            },
            "required": ["file_path", "target_column"]
        }
    },
    {
        "name": "feature_importance",
        "description": "Return ranked feature importances showing what drives the target variable.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to the CSV file"},
                "target_column": {"type": "string", "description": "The target column to analyze"}
            },
            "required": ["file_path", "target_column"]
        }
    }
]


# ── Tool executor ─────────────────────────────────────────────
def execute_tool(tool_name, tool_input):
    if tool_name == "describe_dataset":
        return describe_dataset(**tool_input)
    elif tool_name == "run_python_analysis":
        return run_python_analysis(**tool_input)
    elif tool_name == "generate_chart":
        return generate_chart(**tool_input)
    elif tool_name == "train_ml_model":
        return train_ml_model(**tool_input)
    elif tool_name == "feature_importance":
        return feature_importance(**tool_input)
    return "Tool not found"


# ── Chart display helper ──────────────────────────────────────
def show_chart_with_actions(img_b64, chart_label, file_name):
    """Display chart inline in chat."""
    import base64 as b64lib
    img_bytes = b64lib.b64decode(img_b64)
    st.markdown(f"**📊 {chart_label}**")
    st.image(img_bytes, use_container_width=True)

def is_rate_limit_error(e):
    """Check if the error is a rate limit error."""
    return "rate_limit" in str(e).lower() or "529" in str(e) or "overloaded" in str(e).lower()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("🔑 API Configuration")
    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Get your key from https://console.anthropic.com"
    )
    if api_key:
        st.success("✅ API Key set!")
    else:
        st.warning("⚠️ Enter your API key to begin")
        st.markdown("👉 [Get API Key](https://console.anthropic.com)")

    st.divider()
    st.header("📁 Upload Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    file_path = None

    if uploaded_file:
        os.makedirs("uploads", exist_ok=True)
        file_path = f"uploads/{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"✅ Uploaded: {uploaded_file.name}")

        df_preview = pd.read_csv(file_path)
        st.markdown("**Preview:**")
        st.dataframe(df_preview.head(5), use_container_width=True)
        st.markdown(f"**Shape:** {df_preview.shape[0]} rows × {df_preview.shape[1]} cols")

    st.divider()
    st.markdown("### 💡 Example Questions")
    st.markdown("""
- Find key drivers of revenue
- Which features matter most?
- Show correlation heatmap
- Train a model to predict revenue
- Give me full analysis
    """)

    st.divider()
    if st.button("🔄 New Session", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main chat area ────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

col1, col2 = st.columns([8, 1])
with col2:
    if st.button("🗑️ Clear", help="Clear conversation history"):
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask anything about your dataset...")

if question:
    if not api_key and not os.getenv("ANTHROPIC_API_KEY"):
        st.warning("⚠️ Please enter your Anthropic API key in the sidebar!")
    elif not uploaded_file:
        st.warning("⚠️ Please upload a CSV file first!")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        user_asked_for_chart = any(word in question.lower() for word in [
            "chart", "plot", "graph", "visualize", "show", "heatmap", "visual"
        ])
        user_wants_full = any(word in question.lower() for word in [
            "full analysis", "analyze everything", "full report", "complete analysis"
        ])

        with st.chat_message("assistant"):
            client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

            system_prompt = f"""You are an expert data science agent.
You have access to tools to analyze datasets.
The uploaded CSV file is at: {file_path}

## Tool Usage Rules — follow strictly:

1. describe_dataset → use ONLY when user asks about dataset overview, columns, shape, or data types. Also use it silently first if you need context before answering.

2. run_python_analysis → use ONLY when user asks about correlations, statistics, relationships between variables, or key drivers.

3. generate_chart → use ONLY when user EXPLICITLY mentions chart, plot, graph, visualize, show me, or heatmap, OR when doing full analysis. For correlation_heatmap always pass empty string "" for x_col and y_col. NEVER generate charts unless asked or doing full analysis.

4. train_ml_model → use ONLY when user asks about model, prediction, accuracy, training, or ML.

5. feature_importance → use ONLY when user asks about important features, key drivers, what matters most, or feature ranking.

## Response Rules:
- Use the MINIMUM number of tools needed to answer the question
- NEVER generate charts unless the user explicitly asks for one
- NEVER run all tools together unless user says "full analysis" or "analyze everything"
- Be concise and explain findings in simple business terms
- If a tool result is just for your context, do not explain it step by step to the user
- NEVER include markdown image syntax like ![...](...) in your responses"""

            messages = [{"role": "user", "content": question}]

            with st.spinner("🤖 Agent is thinking..."):
                try:
                  while True:
                    response = client.messages.create(
                        model="claude-opus-4-5",
                        max_tokens=4096,
                        system=system_prompt,
                        tools=tools,
                        messages=messages
                    )

                    for block in response.content:
                        if hasattr(block, "text"):
                            clean_text = re.sub(r'!\[.*?\]\(.*?\)', '', block.text)
                            st.markdown(clean_text)

                        elif block.type == "tool_use":
                            tool_name = block.name
                            tool_input = block.input

                            with st.status(f"🔧 Running `{tool_name}`...", expanded=False) as status:
                                st.json(tool_input)
                                result = execute_tool(tool_name, tool_input)
                                status.update(label=f"✅ `{tool_name}` complete", state="complete")

                            if tool_name == "generate_chart" and (user_asked_for_chart or user_wants_full):
                                result_data = json.loads(result)
                                img_b64 = result_data.get("image_b64")
                                if img_b64:
                                    chart_type = result_data.get("chart_type", "chart")
                                    label = "Correlation Heatmap" if chart_type == "correlation_heatmap" else f"{chart_type.title()} Chart"
                                    file_name = os.path.basename(result_data.get("output_path", "chart.png"))
                                    show_chart_with_actions(img_b64, label, file_name)

                            elif tool_name == "feature_importance" and (user_asked_for_chart or user_wants_full):
                                result_data = json.loads(result)
                                img_b64 = result_data.get("image_b64")
                                if img_b64:
                                    file_name = os.path.basename(result_data.get("chart_saved", "feature_importance.png"))
                                    show_chart_with_actions(img_b64, "Feature Importance Chart", file_name)

                    if response.stop_reason == "end_turn":
                        final_text = " ".join(
                            block.text for block in response.content if hasattr(block, "text")
                        )
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": final_text
                        })
                        break

                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": execute_tool(block.name, block.input)
                            }
                            for block in response.content
                            if block.type == "tool_use"
                        ]
                    })

                except Exception as e:
                    if is_rate_limit_error(e):
                        st.info(
                            "⏳ **I'm getting a lot of requests right now!**\n\n"
                            "The AI service is temporarily at capacity. "
                            "Please wait **30–60 seconds** and try your question again. "
                            "This is temporary and will resolve shortly! 🙏"
                        )
                    else:
                        st.info(
                            "⚠️ **Something went wrong on my end.**\n\n"
                            f"Error details: `{str(e)}`\n\n"
                            "Please try again or rephrase your question."
                        )
