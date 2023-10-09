import streamlit as st
import os
import zipfile
import io
from utils import list_files_in_directory, get_user_directory, ptree


def page_downloads():
    st.title("File Manager")

    if "user_id" not in st.session_state or not st.session_state.user_id:
        st.warning("Warning: Authenticate before downloading data.")
        return

    else:
        # Assuming the directory is named after the user's UUID

        # List all files in the directory
        st.subheader("Directory structure")
        path = get_user_directory()
        if path:
            structure = ptree(path)
            st.code(structure, language="plaintext")

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

        st.markdown("<br>" * 1, unsafe_allow_html=True)

        files = list_files_in_directory(get_user_directory())
        selected_files = st.multiselect("Select files to delete:", files)

        if len(selected_files) > 0 and st.button("Delete Files"):
            for selected in selected_files:
                file_path = os.path.join(get_user_directory(), selected)
                try:
                    os.remove(file_path)
                    st.success(f"Deleted {selected}")
                except Exception as e:
                    st.error(f"Error deleting {selected}: {e}")
