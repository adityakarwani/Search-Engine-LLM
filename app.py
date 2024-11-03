import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain.agents import create_react_agent, AgentType
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Arxiv and Wikipedia Tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

search = DuckDuckGoSearchRun(name="Search")

# Streamlit UI
st.title("🔎 LangChain - Chat with Search")
"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

# Sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# Initialize chat messages in session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

# Display chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

# Input from user
if prompt := st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Initialize the ChatGroq model
    llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)
    tools = [search, arxiv, wiki]

    # Create the search agent
    search_agent = create_react_agent(llm, tools, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        try:
            # Run the agent with the user's query
            response = search_agent.run({"input": prompt}, callbacks=[st_cb])  # Adjusted for run method
            st.session_state.messages.append({'role': 'assistant', "content": response})
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
