import streamlit as st
import pandas as pd


def page_dashboard():
    st.title("Dashboard Page")

    # load excel file
    df_hp_runs = pd.read_csv("./tmp/hp_runs_data.csv")

    df_eval_set = pd.read_json("./tmp/eval_data.json")
    df_eval_set = pd.concat(
        [df_eval_set, pd.json_normalize(df_eval_set["metadata"])], axis=1
    )
    df_eval_set = df_eval_set.drop(columns=["metadata"])

    merged_df = df_eval_set.merge(df_hp_runs, left_on="id", right_on="qa_id")
    merged_df.drop(columns="id", inplace=True)

    with st.expander("Generated evaluation dataset"):
        showData = st.multiselect(
            "Filter: ",
            df_eval_set.columns,
            default=["question", "answer", "source", "id"],
        )
        st.dataframe(df_eval_set[showData], use_container_width=True)

    with st.expander("Predictions for each set of hyperparameters"):
        showData = st.multiselect(
            "Filter: ",
            df_hp_runs.columns,
            default=["hp_id", "predicted_answer", "retrieved_docs", "qa_id"],
        )
        st.dataframe(df_hp_runs[showData], use_container_width=True)

    with st.expander("Combined evaluation and prediction sets"):
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
