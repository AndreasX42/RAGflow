import streamlit as st
import os
import zipfile
import io
from utils import (
    list_files_in_directory,
    get_user_directory,
    ptree,
    display_user_login_warning,
)

import time


def page_filemanager():
    st.title("File Manager")
    st.subheader("Manager the files in your directory.")

    if display_user_login_warning():
        return

    else:
        tab1, tab2 = st.tabs(["Download files", "Delete files"])

        with tab1:
            files = list_files_in_directory(get_user_directory())
            selected_files = st.multiselect("Select files to download:", files)

            if len(selected_files) > 0 and st.button("Select Files"):
                # Create a zip archive in-memory
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(
                    zip_buffer, "a", zipfile.ZIP_DEFLATED, False
                ) as zip_file:
                    for selected in selected_files:
                        file_path = os.path.join(get_user_directory(), selected)
                        zip_file.write(file_path, selected)

                zip_buffer.seek(0)
                st.download_button(
                    label="Download Files",
                    data=zip_buffer,
                    file_name="files.zip",
                    mime="application/zip",
                )

        with tab2:
            files = list_files_in_directory(get_user_directory())
            selected_files = st.multiselect("Select files to delete:", files)

            if len(selected_files) > 0 and st.button("Delete Files"):
                action_success = True
                for selected in selected_files:
                    file_path = os.path.join(get_user_directory(), selected)
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        st.error(f"Error deleting {selected}: {e}")
                        action_success = False

                if action_success:
                    st.success(f"Deleted selected files successfully!")

        st.markdown("<br>" * 1, unsafe_allow_html=True)

        # List all files in the directory
        st.subheader("User directory")
        path = get_user_directory()
        if path or st.session_state.refresh_trigger:
            structure = ptree(path)
            st.code(structure, language="plaintext")
