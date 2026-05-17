from rag import app, vectordb

from langchain_core.messages import HumanMessage

import pandas as pd
# =========================
# STREAMLIT UI SETUP
# =========================

import streamlit as st

st.set_page_config(
    page_title="First Rag",
    layout="centered"
)

st.title("💬 Academic QA")

# =========================
# SESSION STATE
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "streamlit_chat_session"

# =========================
# DISPLAY OLD MESSAGES
# =========================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =========================
# CHAT INPUT
# =========================

if prompt := st.chat_input("Ask any question..."):

    # Store user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant response area
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                # =========================
                # RETRIEVAL
                # =========================

                docs = vectordb.similarity_search_with_score(
                    prompt,
                    k=3
                )

                docs_df = pd.DataFrame(
                    [
                        (
                            prompt,
                            doc[0].page_content,
                            doc[0].metadata.get("source"),
                            doc[0].metadata.get("page"),
                            doc[1]
                        )
                        for doc in docs
                    ],
                    columns=[
                        "query",
                        "paragraph",
                        "document",
                        "page_number",
                        "relevant_score"
                    ]
                )

                # =========================
                # CREATE CONTEXT
                # =========================

                context = "\n\n".join(
                    docs_df["paragraph"]
                )

                # =========================
                # LANGGRAPH INVOKE
                # =========================

                result = app.invoke(
                    {
                        "messages": [
                            HumanMessage(
                                content=f"""
                                Context:
                                {context}

                                Question:
                                {prompt}
                                """
                            )
                        ]
                    },
                    config={
                        "configurable": {
                            "thread_id":
                            st.session_state.thread_id
                        }
                    }
                )

                # =========================
                # FINAL RESPONSE
                # =========================

                ai_response = (
                    result["messages"][-1].content
                )

                source_document = (
                    docs_df["document"][0]
                    if not docs_df.empty
                    else "N/A"
                )

                page_numbers = (
                    docs_df["page_number"]
                    .drop_duplicates()
                    .astype(str)
                    .tolist()
                )

                page_numbers_str = ", ".join(page_numbers)

                final_response = f"""
{ai_response}

**Source Document:** {source_document}

**Reference Pages:** {page_numbers_str}
"""

                st.markdown(final_response)

                # Save assistant response
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": final_response
                    }
                )

            except Exception as e:

                st.error(f"Error: {e}")

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "An error occurred."
                    }
                )