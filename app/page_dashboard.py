import streamlit as st
import pandas as pd
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import plotly.express as px


def page_dashboard():
    st.title("Dashboard Page")

    if "user_file_dir" in st.session_state:
        eval_data_path = f"{st.session_state.user_file_dir}eval_data.json"
        hp_runs_data_path = f"{st.session_state.user_file_dir}hp_runs_data.csv"
        eval_results_path = f"{st.session_state.user_file_dir}eval_results.json"

    if "user_id" not in st.session_state or not st.session_state.user_id:
        st.warning("Warning: Authenticate before uploading data.")

    elif not os.path.exists(eval_data_path):
        st.warning("No evaluation dataset found, generate it first.")

    elif not os.path.exists(hp_runs_data_path):
        st.warning("No hyperparameter evaluation data found, start some hp evaluation.")

    else:
        if os.path.exists(eval_data_path) and os.path.exists(hp_runs_data_path):
            # load excel file
            df_hp_runs = pd.read_csv(hp_runs_data_path)

            df_eval_set = pd.read_json(eval_data_path)
            df_eval_set = pd.concat(
                [df_eval_set, pd.json_normalize(df_eval_set["metadata"])], axis=1
            )
            df_eval_set = df_eval_set.drop(columns=["metadata"])

            merged_df = df_eval_set.merge(df_hp_runs, left_on="id", right_on="qa_id")
            merged_df.drop(columns="id", inplace=True)

            tab1, tab2, tab3, tab4 = st.tabs(
                [
                    "Charts",
                    "Evaluation dataset",
                    "Generated predictions",
                    "Combined dataset",
                ]
            )

            with tab1:
                plot_eval_results(eval_results_path)

            with tab2:
                showData = st.multiselect(
                    "Filter: ",
                    df_eval_set.columns,
                    default=["question", "answer", "source", "id"],
                )
                st.dataframe(df_eval_set[showData], use_container_width=True)

            with tab3:
                showData = st.multiselect(
                    "Filter: ",
                    df_hp_runs.columns,
                    default=["hp_id", "predicted_answer", "retrieved_docs", "qa_id"],
                )
                st.dataframe(df_hp_runs[showData], use_container_width=True)

            with tab4:
                showData = st.multiselect(
                    "Filter: ",
                    merged_df.columns,
                    default=[
                        "question",
                        "answer",
                        "predicted_answer",
                        "retrieved_docs",
                        "hp_id",
                        "qa_id",
                        "source",
                    ],
                )
                st.dataframe(merged_df[showData], use_container_width=True)


def plot_eval_results(eval_results_path: str):
    with open(eval_results_path, "r", encoding="utf-8") as file:
        eval_results = json.load(file)

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(eval_results)

    # Extract scores and timestamps into separate DataFrames
    scores_df = df["scores"].apply(pd.Series)
    scores_df["timestamp"] = pd.to_datetime(scores_df["timestamp"])
    # Sort DataFrame based on timestamps for chronological order
    scores_df = scores_df.sort_values(by="timestamp")
    df = df.loc[scores_df.index]
    # Create a combined x-axis with incrementing number + timestamp string
    df["x_values"] = range(1, len(scores_df) + 1)
    df["x_ticks"] = [f"{i}" for i, ts in enumerate(scores_df["timestamp"])]
    df["timestamp"] = scores_df["timestamp"]
    # Melt the scores_df DataFrame for Plotly plotting
    df_melted = pd.melt(
        scores_df.reset_index(),
        id_vars=["index"],
        value_vars=[
            "embedding_cosine_sim",
            "correct_ans",
            "comprehensive_ans",
            "readable_ans",
            "retriever_score",
            "rouge1",
            "rouge2",
        ],
    )
    # Merge on 'index' to get the correct x_values and x_ticks
    df_melted = df_melted.merge(
        df[["x_values", "x_ticks", "timestamp"]],
        left_on="index",
        right_index=True,
        how="left",
    )
    df_melted = df_melted[df_melted.value >= 0]

    # Plot using Plotly Express
    fig = px.scatter(
        df_melted,
        x="x_values",
        y="value",
        color="variable",
        hover_data=["x_ticks", "variable", "value", "timestamp"],
        # color_discrete_sequence=px.colors.sequential.Viridis,
        labels={"x_values": "Hyperparameter run id", "value": "Scores"},
        title="Hyperparameter chart",
    )

    fig.update_layout(
        # xaxis_tickangle=-45,
        xaxis=dict(tickvals=df["x_values"], ticktext=df["x_ticks"]),
        yaxis=dict(tickvals=[i / 10.0 for i in range(11)]),
        plot_bgcolor="#F5F5DC",
        paper_bgcolor="#121212",
        height=600,  # Set the height of the plot
        width=1000,  # Set the width of the plot
    )
    st.plotly_chart(fig)
