import streamlit as st
import os
import zipfile
import io


def list_files_in_directory(path):
    """List all files in the specified directory."""
    with os.scandir(path) as directory:
        return [entry.name for entry in directory if entry.is_file()]


def page_downloads():
    st.title("Download Files")

    if "user_id" not in st.session_state or not st.session_state.user_id:
        st.error("WARNING: Authenticate before downloading data.")
    else:
        user_id = st.session_state.user_id

        # Assuming the directory is named after the user's UUID
        user_directory = f"./tmp/{user_id}/"

        # List all files in the directory
        files = list_files_in_directory(user_directory)

        if not files:
            st.write("No files found!")
            return

        selected_files = st.multiselect("Select files to download:", files)

        if len(selected_files) > 0 and st.button("Select Files"):
            # Create a zip archive in-memory
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(
                zip_buffer, "a", zipfile.ZIP_DEFLATED, False
            ) as zip_file:
                for selected in selected_files:
                    file_path = os.path.join(user_directory, selected)
                    zip_file.write(file_path, selected)

            zip_buffer.seek(0)
            st.download_button(
                label="Download Files",
                data=zip_buffer,
                file_name="files.zip",
                mime="application/zip",
            )
