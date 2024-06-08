import streamlit as st

from agent_chain import chat_with_user, get_llm_chat_instance


## Streamlit related functions ##
def get_custom_css_modifier():
    css_modifier = """
<style>
/* remove Streamlit default menu and footer */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}
</style>
    """
    return css_modifier


def load_resources():
    if "starting_prompts" not in st.session_state:
        st.session_state["starting_prompts_shown"] = False
    if "explore_further_enabled" not in st.session_state:
        st.session_state["explore_further_enabled"] = False
    if "conversation_labels" not in st.session_state:
        st.session_state["conversation_labels"] = None

    return


def enable_explore_further():
    st.session_state["explore_further_enabled"] = True

    initial_entry = st.session_state["new_entry_text"]

    chat_model = get_llm_chat_instance()
    st.session_state["chat_model"] = chat_model
    starting_message, conversation_labels = chat_with_user(initial_entry, chat_model)

    st.session_state["conversation_labels"] = conversation_labels

    # pass the entry to start the conversation

    st.session_state["messages"] = [
        {"role": "user", "content": initial_entry},
        {"role": "assistant", "content": starting_message},
    ]

    return
