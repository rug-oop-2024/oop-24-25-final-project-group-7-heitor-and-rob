import streamlit as st


def main() -> None:
    """
    Set up the Streamlit page configuration and display the instructions.
    """
    st.set_page_config(
        page_title="Instructions",
        page_icon="👋",
    )

    with open("INSTRUCTIONS.md") as file:
        instructions = file.read()
    st.markdown(instructions)


if __name__ == "__main__":
    main()
