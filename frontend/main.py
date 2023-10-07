import streamlit as st
from streamlit_option_menu import option_menu

from page_params import page_params
from page_home import page_home
from page_documents import page_documents
from page_dashboard import page_dashboard
from page_login import page_login


def main():
    # Display selected page with the respective function
    def sideBar():
        with st.sidebar:
            selected = option_menu(
                menu_title="Main Menu",
                options=[
                    "Home",
                    "Dashboard",
                    "Document Store",
                    "Parameters",
                    "Authentication",
                ],
                icons=[
                    "house-fill",
                    "clipboard2-pulse-fill",
                    "cloud-upload-fill",
                    "toggles",
                    "file-earmark-check-fill",
                ],
                menu_icon="cast",
                default_index=0,
                styles={
                    "nav-link": {
                        "--hover-color": "#1E1E1E",
                    },
                    "nav-link-selected": {"background-color": "#880808"},
                },
            )

        if selected == "Home":
            page_home()
        if selected == "Dashboard":
            page_dashboard()
        if selected == "Document Store":
            page_documents()
        if selected == "Parameters":
            page_params()
        if selected == "Authentication":
            page_login()

    sideBar()


if __name__ == "__main__":
    st.set_page_config(
        page_title="RAGflow",
        page_icon="🧊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    main()

    with st.sidebar:
        st.markdown("<br>" * 3, unsafe_allow_html=True)
        st.markdown("---")
        st.markdown(
            '<div style="text-align: center;"><h6>Made in &nbsp;<a href="https://streamlit.io" target="_blank"><img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="16"></a>&nbsp by <a href="https://github.com/AndreasX42">@AndreasX42</a></h6></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="text-align: center; margin-top: 0.75em;"><a href="https://github.com/AndreasX42/RAGflow" target="_blank"><img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Logo.png" alt="Buy Me A Coffee" height="41" width="174"></a></div>',
            unsafe_allow_html=True,
        )
