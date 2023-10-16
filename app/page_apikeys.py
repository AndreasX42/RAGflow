import streamlit as st
from utils import display_user_login_warning


def page_apikeys():
    st.title("Provide API Keys")
    st.subheader("Enter the API Keys for all services you want to use.")

    if display_user_login_warning():
        return

    if "api_keys" not in st.session_state:
        st.session_state.api_keys = {}

    # Create text area to input API keys
    input_text = st.text_area(
        "Enter the required API keys for your intended use. The keys will not get stored outside of Streamlits session state.\n\nProvide one 'name=key' pair per line, for example:",
        "OPENAI_API_KEY = your_openai_key\nANYSCALE_API_KEY = your_anyscale_key",
    )

    # Create a submit button
    if st.button("Submit"):
        store_in_cache(input_text)
        st.success("API keys stored successfully!")

    # Display stored API keys for testing purposes (you might want to remove this in a real application)
    if st.session_state.api_keys:
        keys = ""
        for key, value in st.session_state.api_keys.items():
            keys += f"{key}: {value}\n"

        st.markdown("<br>" * 1, unsafe_allow_html=True)
        st.code(keys)


def store_in_cache(api_keys: str) -> None:
    """Writes api keys to streamlit session state."""
    st.session_state.api_keys = {}

    for line in api_keys.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            st.session_state.api_keys[key.strip()] = value.strip()
