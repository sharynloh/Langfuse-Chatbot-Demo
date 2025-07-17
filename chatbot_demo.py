import streamlit as st
import time
import uuid
from langchain_ollama import ChatOllama
from langfuse import observe, get_client
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Optional: flush Langfuse events at the end
import atexit

load_dotenv()
langfuse = get_client()

st.title("Chatbot")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "trace_id" not in st.session_state:
    st.session_state.trace_id = None

# Langfuse tracing
@observe(name = 'preprocessing', as_type = 'span')
def preprocess_input(user_input: str):
    time.sleep(0.5)
    return user_input.strip()

@observe(name="llm-call", as_type="span")
def call_model(messages):
    llm = ChatOllama(model="llama3", temperature=1.0)
    return llm.invoke(messages)

@observe(name = 'chatbot-generation', as_type = 'generation')
def invoke_llm(messages, session_id):
    langfuse.update_current_trace(session_id=session_id)

    user_input = messages[-1].content
    cleaned_input = preprocess_input(user_input)

    # Step 2: Call the LLM
    response = call_model(messages)

    # Step 3: Token + cost calculation
    prompt_tokens = sum(len(m.content.split()) for m in messages)
    completion_tokens = len(response.content.split())
    total_tokens = prompt_tokens + completion_tokens
    cost = round((prompt_tokens / 1_000_000 * 0.73) + (completion_tokens / 1_000_000 * 0.84), 6)

    # Step 4: Update the current generation span
    langfuse.update_current_generation(
        model = 'llama3',
        input=messages[-1].content,
        metadata={"prompt_tokens": int(prompt_tokens), 
                  "completion_tokens": int(completion_tokens),
                  "total_tokens": int(total_tokens),
                  "cost_usd": cost
            }
        )
    
    langfuse.update_current_generation(
        usage_details={"input": int(prompt_tokens), 
                  "output": int(completion_tokens),
                  "total": int(total_tokens),
            }
        )

    # Step 5: Save trace_id for feedback
    st.session_state.trace_id = langfuse.get_current_trace_id()

    return response.content

# display chat messages from history on app rerun
for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# create the bar where we can type messages
prompt = st.chat_input("How are you?")

# did the user submit a prompt?
if prompt:
    # display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append(HumanMessage(prompt))

    # invoke model with tracing
    result = invoke_llm(st.session_state.messages, st.session_state.session_id)

    # display assistant message
    with st.chat_message("assistant"):
        st.markdown(result)
    st.session_state.messages.append(AIMessage(result))

    st.markdown("---")
st.subheader("💬 Feedback on the last response")

col1, col2 = st.columns([1, 4])

with col1:
    feedback_choice = st.radio("Was it helpful?", ["👍", "👎"], horizontal=True, key="feedback_radio")
    score_value = 1 if feedback_choice == "👍" else 0

with col2:
    feedback_comment = st.text_area("Leave an optional comment", key="feedback_comment")

if st.button("Submit Feedback", key="feedback_submit"):
    trace_id = st.session_state.get("trace_id")

    if trace_id:
        langfuse.create_score(
            name="user-feedback",
            value=score_value,
            trace_id=trace_id,
            comment=feedback_comment.strip() if feedback_comment else None
        )
        langfuse.flush()
        st.success("✅ Feedback submitted. Thank you!")
    else:
        st.error("❌ Could not submit feedback: trace ID not available.")

# flush Langfuse traces at the end (important for short-lived apps like Streamlit)
langfuse = get_client()
atexit.register(lambda: langfuse.flush())