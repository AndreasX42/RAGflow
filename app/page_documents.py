import streamlit as st


def page_documents():
    # Title of the page
    st.title("Document Store Page")
    st.subheader("Provide the documents that the application should use.")

    if "user_id" not in st.session_state or not st.session_state.user_id:
        st.warning("WARNING: Authenticate before uploading data.")

    tab1, tab2 = st.tabs(["Upload JSON file", "Provide cloud storage link"])

    with tab1:
        # Upload local files
        uploaded_files = st.file_uploader(
            "Choose a set of PDF, TXT, or DOCX files",
            accept_multiple_files=True,
            type=["pdf", "txt", "docx"],
        )

        for uploaded_file in uploaded_files:
            st.write(f"You uploaded {uploaded_file.name}")

    with tab2:
        # Provide link to cloud resource
        cloud_link = st.text_input("Provide a link to your cloud resource:")

        if cloud_link:
            st.write(f"You provided the link: {cloud_link}")
