from autoop.core.ml.artifact import Artifact
import streamlit as st

def main():
    """
    Main function to set up the Streamlit page configuration and display content.
    """
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )
    st.sidebar.success("Select a page above.")
    with open("README.md") as readme_file:
        st.markdown(readme_file.read())

if __name__ == "__main__":
    main()