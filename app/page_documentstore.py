import streamlit as st
from utils import *
from utils import display_user_login_warning


def page_documentstore():
    # Title of the page
    st.title("Document Store Page")
    st.subheader("Provide the documents that the application should use.")

    if display_user_login_warning():
        return

    else:
        tab1, tab2 = st.tabs(["Upload Documents", "Provide cloud storage"])

        with tab1:
            upload_files(
                context="docs",
                dropdown_msg="Upload your documents",
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

        st.markdown("<br>" * 1, unsafe_allow_html=True)

        # List all files in the directory
        st.subheader("Your Document Store")
        path = get_document_store_path()
        if path:
            structure = ptree(path)
            st.code(structure, language="plaintext")
