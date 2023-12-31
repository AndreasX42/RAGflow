import streamlit as st
import pandas as pd
import os
import pandas as pd
import json
import plotly.express as px
from utils import *
from utils import display_user_login_warning


def page_dashboard():
    st.title("Dashboard Page")
    st.subheader("Analyse hyperparameter metrics and evaluation dataset.")

    if display_user_login_warning():
        return

    else:
        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "Charts",
                "Evaluation dataset",
                "Generated predictions",
                "Combined dataset",
            ]
        )

        with tab1:
            if os.path.exists(get_hyperparameters_results_path()):
                plot_hyperparameters_results(get_hyperparameters_results_path())
            else:
                st.warning("No hyperparameter results available. Run some evaluation.")

        with tab2:
            if os.path.exists(get_label_dataset_path()):
                df_label_dataset = get_df_label_dataset()

                showData = st.multiselect(
                    "Filter: ",
                    df_label_dataset.columns,
                    default=["question", "answer", "context", "source", "id"],
                )
                st.dataframe(df_label_dataset[showData], use_container_width=True)
            else:
                st.warning("No evaluation data available. Generate it.")

        with tab3:
            if os.path.exists(get_hyperparameters_results_data_path()):
                df_hp_runs = pd.read_csv(get_hyperparameters_results_data_path())

                showData = st.multiselect(
                    "Filter: ",
                    df_hp_runs.columns,
                    default=[
                        "hp_id",
                        "predicted_answer",
                        "retrieved_docs",
                        "qa_id",
                    ],
                )
                st.dataframe(df_hp_runs[showData], use_container_width=True)
            else:
                st.warning("No generated dataset from hyperparameter runs available.")

        with tab4:
            if os.path.exists(get_label_dataset_path()) and os.path.exists(
                get_hyperparameters_results_data_path()
            ):
                df_label_dataset4 = get_df_label_dataset()
                df_hp_runs4 = get_df_hp_runs()

                merged_df = df_label_dataset4.merge(
                    df_hp_runs4, left_on="id", right_on="qa_id"
                )
                merged_df.drop(columns="id", inplace=True)

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
            else:
                st.warning("Not sufficient data available.")


def get_df_label_dataset() -> pd.DataFrame:
    df_label_dataset = pd.read_json(get_label_dataset_path())
    df_label_dataset = pd.concat(
        [df_label_dataset, pd.json_normalize(df_label_dataset["metadata"])],
        axis=1,
    )
    df_label_dataset = df_label_dataset.drop(columns=["metadata"])

    return df_label_dataset


def get_df_hp_runs() -> pd.DataFrame():
    return pd.read_csv(get_hyperparameters_results_data_path())


def plot_hyperparameters_results(hyperparameters_results_path: str):
    with open(hyperparameters_results_path, "r", encoding="utf-8") as file:
        hyperparameters_results = json.load(file)

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(hyperparameters_results)

    # Extract scores and timestamps into separate DataFrames
    scores_df = df["scores"].apply(pd.Series)
    scores_df["timestamp"] = pd.to_datetime(df["timestamp"])
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
            "answer_similarity_score",
            "retriever_mrr@3",
            "retriever_mrr@5",
            "retriever_mrr@10",
            "rouge1",
            "rouge2",
            "rougeLCS",
            "correctness_score",
            "comprehensiveness_score",
            "readability_score",
            "retriever_semantic_accuracy",
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
        width=950,  # Set the width of the plot
    )
    st.plotly_chart(fig)
