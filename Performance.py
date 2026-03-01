import streamlit as st
import plotly.express as px
import pandas as pd
from utils.metrics import load_classification_metrics, load_yolo_metrics

st.set_page_config(page_title="Model Performance", layout="wide")

st.title("📊 Model Performance Dashboard")

# ===============================
# Classification Metrics
# ===============================

st.subheader("🔍 Classification Models Comparison")

df = load_classification_metrics()

if df.empty:
    st.warning("No classification metrics found.")
else:
    col1, col2 = st.columns(2)

    # Show metrics table
    with col1:
        st.markdown("### 📋 Metrics Table")
        st.dataframe(df, use_container_width=True)

    # Accuracy Bar Chart
    with col2:
        st.markdown("### 📈 Accuracy Comparison")
        fig = px.bar(
            df,
            x="Model",
            y="Accuracy",
            color="Model",
            text="Accuracy"
        )
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Precision, Recall, F1 Comparison
    st.markdown("### 📊 Precision / Recall / F1 Score")

    metric_df = df.melt(
        id_vars="Model",
        value_vars=["Precision", "Recall", "F1-Score"],
        var_name="Metric",
        value_name="Score"
    )

    fig2 = px.bar(
        metric_df,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Show best model
    best_model = df.sort_values("Accuracy", ascending=False).iloc[0]
    st.success(
        f"🏆 Best Classification Model: {best_model['Model']} "
        f"(Accuracy: {best_model['Accuracy']:.4f})"
    )

# ===============================
# YOLOv8 Metrics
# ===============================

st.subheader("🎯 YOLOv8 Object Detection Metrics")

yolo_metrics = load_yolo_metrics()

if not yolo_metrics:
    st.warning("No YOLO metrics found.")
else:
    yolo_df = pd.DataFrame(
        list(yolo_metrics.items()),
        columns=["Metric", "Value"]
    )

    st.dataframe(yolo_df, use_container_width=True)

    fig3 = px.bar(
        yolo_df,
        x="Metric",
        y="Value",
        color="Metric",
        text="Value"
    )
    fig3.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig3.update_layout(showlegend=False)

    st.plotly_chart(fig3, use_container_width=True)