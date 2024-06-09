import streamlit as st

from .search import display_search
from .utils import clear_chat, init_env, sessionKeys


def side_bar():
    with st.sidebar:
        st.write("# Welcome to Avalok AI")
        st.button(label="Clear search", on_click=clear_chat)


def display(dbname: str, config_file: str):
    if sessionKeys.DB_NAME not in st.session_state:
        init_env(dbname, config_file)
    side_bar()
    display_search()
