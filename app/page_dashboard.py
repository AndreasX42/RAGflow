import streamlit as st
import pandas as pd
import os


def page_dashboard():
    st.title("Dashboard Page")

    if "user_id" not in st.session_state or not st.session_state.user_id:
        st.warning("WARNING: Authenticate before uploading data.")

    else:
        eval_set_path = f"./tmp/{st.session_state.user_id}/eval_data.csv"
        hp_runs_path = f"./tmp/{st.session_state.user_id}/hp_runs_data.json"

        if os.path.exists(eval_set_path) and os.path.exists(hp_runs_path):
            # load excel file
            df_hp_runs = pd.read_csv(
                f"./tmp/{st.session_state.user_id}/hp_runs_data.csv"
            )

            df_eval_set = pd.read_json(
                f"./tmp/{st.session_state.user_id}/eval_data.json"
            )
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
