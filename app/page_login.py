import streamlit as st
import time

from utils import *


def page_login():
    # Session State to track if user is logged in
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if "user_id" not in st.session_state:
        st.session_state.user_id = ""

    # Tabs for Login and Registration
    tab1, tab2 = st.tabs(["Login", "Register"])

    # Login Tab
    with tab1:
        # Check if user is already logged in
        response, success = get_auth_user()
        if success:
            st.session_state.user_id = str(response.get("id"))
            st.success(f"Already signed in as '{response.get('username')}'.")

        elif not st.session_state.logged_in:
            with st.form("login_form"):
                st.subheader("Login")
                # Input fields for username and password
                login_username = st.text_input("Username", key="login_username")
                login_password = st.text_input(
                    "Password", type="password", key="login_password"
                )

                # Submit button for the form
                submit_button = st.form_submit_button("Login")

                if submit_button:
                    # Attempt to login
                    response, success = user_login(login_username, login_password)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.user_id = str(response.get("id"))
                        st.success(
                            f"Logged in successfully as {response.get('username')}."
                        )
                    else:
                        st.error(
                            f"Login failed: {response.get('detail', 'Unknown error')}"
                        )

        if get_cookie_value() or st.session_state.user_id:
            with st.form("logout_form"):
                st.subheader("Logout")
                logout_button = st.form_submit_button("Logout")
                if logout_button:
                    success = user_logout()

                    if success:
                        st.session_state.logged_in = False
                        st.session_state.user_id = ""
                        st.success("You are logged out!")

                    else:
                        st.error("Error logging out")

                    time.sleep(1)
                    st.rerun()

    # Registration Tab
    with tab2:
        with st.form("register_form"):
            st.subheader("Register")
            reg_username = st.text_input(
                "Username",
                key="reg_username",
                help="Username must be between 4 and 64 characters.",
            )

            reg_email = st.text_input("Email", key="reg_email")

            reg_password = st.text_input(
                "Password",
                type="password",
                key="reg_password",
                help="Password must be between 8 and 128 characters.",
            )

            if st.form_submit_button("Register"):
                reg_response, success = user_register(
                    reg_username, reg_email, reg_password
                )

                if success:
                    st.success(
                        f"Registered successfully! Please log in, {reg_response.get('username')}."
                    )

                else:
                    st.error(f"Registration failed! Info: {reg_response['detail']}")
