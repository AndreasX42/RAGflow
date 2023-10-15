import streamlit as st
from streamlit_option_menu import option_menu

from page_params import page_params
from page_home import page_home
from page_documents import page_documents
from page_dashboard import page_dashboard
from page_login import page_login
from page_downloads import page_downloads
from page_apis import page_apis


def main():
    # Display selected page with the respective function
    def sideBar():
        with st.sidebar:
            selected = option_menu(
                menu_title="Main Menu",
                options=[
                    "Home",
                    "Dashboard",
                    "Parameters",
                    "Documents",
                    "File Manager",
                    "API Keys",
                    "Login",
                ],
                icons=[
                    "house-fill",
                    "clipboard2-pulse-fill",
                    "toggles",
                    "cloud-upload-fill",
                    "file-earmark-text-fill",
                    "safe-fill",
                    "door-open-fill",
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
        if selected == "Parameters":
            page_params()
        if selected == "Documents":
            page_documents()
        if selected == "File Manager":
            page_downloads()
        if selected == "API Keys":
            page_apis()
        if selected == "Login":
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
        if "user_id" in st.session_state and st.session_state.user_id:
            auth_button = '<div style="display: flex; justify-content: center; align-items: center; height: 5vh;"><span style="display: inline-flex; align-items: center; justify-content: center; color:white; background-color:green; padding:8px; border-radius:5px;">Authenticated</span></div>'

        else:
            auth_button = '<div style="display: flex; justify-content: center; align-items: center; height: 5vh;"><span style="display: inline-flex; align-items: center; justify-content: center; color:white; background-color:#880808; padding:8px; border-radius:5px;">Not Authenticated</span></div>'

        st.markdown(auth_button, unsafe_allow_html=True)

        st.markdown("<br>" * 0, unsafe_allow_html=True)
        st.markdown("---")
        st.markdown(
            '<div style="text-align: center;"><h6>Made in &nbsp;<a href="https://streamlit.io" target="_blank"><img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="16"></a>&nbsp by <a href="https://github.com/AndreasX42">@AndreasX42</a></h6></div>',
            unsafe_allow_html=True,
        )

        version_info = """
        <div style="text-align: center; margin-top: 0em; font-size: 0.8em">
            Build Number: $BUILD_NUMBER<br>
            Build Date: $BUILD_DATE<br>
            Git Commit SHA: <a href="https://github.com/AndreasX42/RAGflow/commit/$GIT_SHA" target="_blank">$GIT_SHA</a><br>
        </div>
        """

        st.markdown(version_info, unsafe_allow_html=True)
