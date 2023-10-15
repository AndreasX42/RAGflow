import streamlit as st
import uuid
import os
import re


def page_login():
    st.title("Authentication Page")
    st.subheader("Enter your UUID to authenticate.")

    tab1, tab2 = st.tabs(["Enter existing key", "Generate new key"])

    with tab1:
        if "user_id" not in st.session_state:
            st.session_state.user_id = ""

        # Check if a UUID is already set in session state or not

        # Input for existing or newly generated UUID
        existing_uuid = st.text_input("Enter your UUID (if you have one):")
        if (
            existing_uuid
            and not is_valid_uuid(existing_uuid)
            or not os.path.exists(f"./tmp/{existing_uuid}/")
        ):
            existing_uuid = ""
            st.error("Please provide a valid UUID format.")

        else:
            if existing_uuid:
                set_state(existing_uuid)
                st.success(f"Authenticated your UUID: {st.session_state.user_id}")

    with tab2:
        # If no UUID in session, show button to generate a new UUID
        if st.button("Generate new UUID"):
            new_uuid = str(uuid.uuid4())
            set_state(new_uuid)
            st.success(
                f"Your new UUID (save it somewhere safe!): {st.session_state.user_id}"
            )


def set_state(user_id: str) -> None:
    st.session_state.user_id = user_id
    st.session_state.user_file_dir = f"./tmp/{st.session_state.user_id}/"
    os.makedirs(f"./tmp/{st.session_state.user_id}/", exist_ok=True)


def is_valid_uuid(val):
    regex = re.compile(
        r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}\Z", re.I
    )
    match = regex.match(val)
    return bool(match)
