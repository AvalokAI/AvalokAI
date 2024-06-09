import streamlit as st
from avalokai import Searcher

from .utils import sessionKeys


def display_search():
    dbname: str = st.session_state[sessionKeys.DB_NAME]
    st.write(f"# Search documents in {dbname} database")

    if query := st.chat_input("Search a query"):
        searcher: Searcher = st.session_state[sessionKeys.SEARCHER]
        query = query.lower()
        matches = searcher.search(query)
        for _, match in enumerate(matches):
            doc_id = match["id"]
            st.text_area(
                f"Document id {doc_id},  document match score {match['score']}",
                match["content"],
            )
