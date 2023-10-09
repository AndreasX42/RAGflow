import streamlit as st
from utils import *


def page_documents():
    # Title of the page
    st.title("Document Store Page")
    st.subheader("Provide the documents that the application should use.")

    if "user_id" not in st.session_state or not st.session_state.user_id:
        st.warning("Warning: Authenticate before uploading data.")
        return

    else:
        tab1, tab2 = st.tabs(["Upload JSON file", "Provide cloud storage link"])

        doc_save_path = f"{st.session_state.user_file_dir}/document_store/"

        with tab1:
            # Upload local files
            upload_files(
                context="docs",
                ext_list=["pdf", "txt", "docx"],
                file_path=get_document_store_path(),
                allow_multiple_files=True,
            )

        with tab2:
            st.write("Not implemented yet.")
            # Provide link to cloud resource
            cloud_link = st.text_input("Provide a link to your cloud resource:")

            if cloud_link:
                st.write(f"You provided the link: {cloud_link}")

        # List all files in the directory
        st.subheader("Files uploaded")
        path = get_document_store_path()
        if path:
            structure = ptree(path)
            st.code(structure, language="plaintext")
