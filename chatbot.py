from utils import *

st.set_page_config(page_title='Template Chatbot', page_icon='💬', layout="wide", initial_sidebar_state="expanded")
st.markdown(get_custom_css_modifier(), unsafe_allow_html=True)
st.markdown("<h5 style='text-align: Left;'>💬 Template Chatbot</h5>", unsafe_allow_html=True)
st.title("Sophie's diary")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

chat_history_box = st.container()
with chat_history_box:
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])


prompt = st.chat_input()
if prompt:

    st.chat_message("user").write(prompt)
    
    response = f'I hear you said {prompt}'

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": response})

    st.chat_message("assistant").write(response)
    st.rerun()

