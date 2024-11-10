import streamlit as st
import os


def main() -> None:
    """
    Main function to set up the Streamlit page
    configuration and display content.
    """
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )
    st.sidebar.success("Select a page above.")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    readme_path = os.path.join(project_root, "README.md")
    with open(readme_path) as readme_file:
        st.markdown(readme_file.read())


if __name__ == "__main__":
    main()
