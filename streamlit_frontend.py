import streamlit as st
from backend import chatbot
from langchain_core.messages import BaseMessage,HumanMessage

st.title("LangGraph Chatbot")
## st.session_state is a dictionary-like object that allows you to store information across different runs of the Streamlit app. In this case, we are using it to store the chat history, which is a list of messages that have been exchanged between the user and the chatbot. If the chat history does not exist in the session state, we initialize it as an empty list.
CONFIG={'configurable':{'thread_id':"thread-1"}}
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []
    

    
    
    
## loading the conversation history
# ******************************main ui  ****************

for message in st.session_state["message_history"]:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
## getting user input
## {role: "user", content: "Hello, how are you?"}
# {role : "assistant", content: "I'm good, thank you! How can I help you today?"}
user_input=st.chat_input("Type your message here...")
if user_input:
    ## first add the message to message_history
    st.session_state['message_history'].append({'role':"user", "content":user_input})
    with st.chat_message("user"):
        st.text(user_input)
    
    with st.chat_message("assistant"):
        ai_message=st.write_stream(
            message_chunk.content for message_chunk , metadata in chatbot.stream(
               {"messages": [HumanMessage(content=user_input)]},
                config= {'configurable': {'thread_id': 'thread-1'}},
                stream_mode="messages"
            )
        )
    st.session_state['message_history'].append({'role':"assistant", "content":ai_message})