"""
app.py
------
Streamlit chat interface for the ML + RAG Chatbot.
Connects the UI to the LangGraph backend.

Run: streamlit run app.py
"""

import html
import os
import re

from dotenv import load_dotenv
import streamlit as st

load_dotenv()
for key, value in st.secrets.items():
    if value and key not in os.environ:
        os.environ[key] = value

from graph import run_chatbot

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Car Price Chatbot",
    page_icon="🚗",
    layout="centered",
)

# ── Custom CSS – clean, modern look ──────────────────────────────────────────
st.markdown("""
<style>
  /* ---- global ---- */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
  html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

  /* ---- header ---- */
  .hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
    padding: 2rem 1.5rem 1.5rem;
    border-radius: 16px;
    text-align: center;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
  }
  .hero h1  { color: #e0e0ff; font-size: 2rem; margin: 0; letter-spacing: -0.5px; }
  .hero p   { color: #a0a8d0; margin: 0.4rem 0 0; font-size: 0.95rem; }

  /* ---- chat bubbles ---- */
  .bubble-user {
    background: #0f3460;
    color: #e8eaff;
    padding: 0.65rem 0.9rem;
    border-radius: 16px 16px 4px 16px;
    margin: 0.35rem 0 0.35rem 3rem;
    line-height: 1.4;
  }
  .bubble-bot {
    background: #1e1e3a;
    color: #d0d4f0;
    padding: 0.65rem 0.9rem;
    border-radius: 16px 16px 16px 4px;
    margin: 0.35rem 3rem 0.35rem 0;
    line-height: 1.45;
    border-left: 3px solid #e94560;
  }
  .bubble-content {
    white-space: pre-wrap;
  }
  .bubble-label {
    display: inline-block;
    margin-bottom: 0.2rem;
    font-size: 0.82rem;
    font-weight: 700;
    opacity: 0.9;
  }

  /* ---- sample queries chip row ---- */
  .chips { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 1rem; }
  .chip  {
    background: #1e1e3a; color: #a0a8d0;
    padding: 0.3rem 0.8rem; border-radius: 999px;
    font-size: 0.8rem; border: 1px solid #2a2a55;
    cursor: pointer;
  }

  /* ---- input area ---- */
  .stTextInput > div > div > input {
    background: #1e1e3a !important;
    color: #e0e0ff !important;
    border: 1px solid #2a2a55 !important;
    border-radius: 10px !important;
  }
  .stButton button {
    background: #e94560 !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
  }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🚗 Car Price AI Chatbot</h1>
  <p>Predict prices with ML · Explore car knowledge with RAG · Powered by LangGraph</p>
  <p>ML prediction supports core dataset brands, while RAG provides broader car-domain guidance.</p>
</div>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "prefill"  not in st.session_state:
    st.session_state.prefill  = ""


def render_bubble(message: str) -> str:
    compact = re.sub(r"\n\s*\n\s*\n+", "\n\n", message.strip())
    compact = re.sub(r"\n\s*[-*]\s+", "\n• ", compact)
    safe = html.escape(compact)
    return f'<div class="bubble-content">{safe}</div>'

# ── Sample queries ────────────────────────────────────────────────────────────
SAMPLES = [
    "Predict price for 2022 BMW Petrol 2000cc 190hp 20000km",
    "How much is a 2019 Toyota Diesel 30000km?",
    "What factors affect car resale value?",
    "Why are electric cars more expensive?",
    "Tips for buying a used car in India?",
]

st.markdown("**💡 Try a sample query:**")
cols = st.columns(len(SAMPLES))
for col, sample in zip(cols, SAMPLES):
    if col.button(sample[:30] + "…", key=sample):
        st.session_state.prefill = sample
        st.rerun()

st.divider()

# ── Chat history display ──────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="bubble-user"><div class="bubble-label">👤 You</div>{render_bubble(msg["content"])}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="bubble-bot"><div class="bubble-label">🤖 AutoMate</div>{render_bubble(msg["content"])}</div>',
            unsafe_allow_html=True,
        )

# ── Input form ────────────────────────────────────────────────────────────────
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input(
        "Your message",
        value=st.session_state.prefill,
        placeholder="e.g.  Predict price for a 2021 Honda Petrol 1500cc 40000km …",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("Send 🚀")

# Reset prefill after use
if st.session_state.prefill and not submitted:
    st.session_state.prefill = ""

# ── Handle submission ─────────────────────────────────────────────────────────
if submitted and user_input.strip():
    query = user_input.strip()

    # Append user message
    st.session_state.messages.append({"role": "user", "content": query})

    # Show a spinner while the graph runs
    with st.spinner("Thinking…"):
        answer = run_chatbot(query, history=st.session_state.messages[:-1])

    # Append bot answer
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Clear prefill and rerun to refresh the chat
    st.session_state.prefill = ""
    st.rerun()

# ── Sidebar – info panel ──────────────────────────────────────────────────────
with st.sidebar:
    st.title("ℹ️ About")
    st.markdown("""
**ML + RAG Chatbot**

This chatbot specializes in Indian car pricing, resale, and used-car decision support.

**Scope note:**  
`ML price prediction` is tuned for core dataset brands:  
`Toyota`, `Honda`, `Ford`, `BMW`, `Hyundai`, `Maruti`

`RAG knowledge responses` cover broader car-domain guidance such as brands, resale, fuel type, mileage, depreciation, and used-car buying in India.

| Component | Technology |
|-----------|-----------|
| ML model  | Linear Regression (sklearn) |
| Embeddings | MiniLM-L6-v2 (HuggingFace) |
| Vector DB  | FAISS |
| Orchestration | LangGraph |
| UI | Streamlit |

---

**How it works:**

1. You type a query
2. Router checks intent
3. **Price query?** → ML prediction
4. **Info query?** → FAISS RAG
5. Response displayed & stored

---

**Prediction keywords:**  
`price`, `predict`, `cost`, `worth`, `value`, `estimate`, `how much`

**Features you can specify:**  
brand, year, km driven, fuel type, engine cc, horsepower, seats
    """)

    if st.button("🗑️  Clear chat"):
        st.session_state.messages = []
        st.rerun()
