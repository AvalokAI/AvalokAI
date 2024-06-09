from enum import Enum

import streamlit as st
from avalokai import Searcher


class sessionKeys(Enum):
    DB_NAME = "dbname"
    SEARCHER = "searcher"


def init_env(dbname: str, config_file: str):
    st.session_state[sessionKeys.DB_NAME] = dbname
    st.session_state[sessionKeys.SEARCHER] = Searcher(dbname, config_file)
    st.set_page_config(
        layout="wide",
        page_title="Search documents",
        page_icon="🔍",
        menu_items=None,
    )


def clear_chat():
    pass
