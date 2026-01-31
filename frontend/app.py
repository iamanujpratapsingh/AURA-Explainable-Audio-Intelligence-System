import streamlit as st
import plotly.express as px
from pyvis.network import Network
import tempfile
import os
import sys
import pandas as pd
import requests
from streamlit_lottie import st_lottie
import streamlit.components.v1 as components

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try: from backend.aura_engine import AuraEngine
except: st.error("Backend Missing"); st.stop()

st.set_page_config(page_title="NovaTech AURA: Gemini Online Edition", layout="wide", page_icon="⚡")

def load_lottie(url):
    try: return requests.get(url, timeout=2).json()
    except: return None

@st.cache_resource
def load_engine(): return AuraEngine()
engine = load_engine()

if "processed" not in st.session_state: st.session_state.processed = False
if "chat_history" not in st.session_state: st.session_state.chat_history = []

with st.sidebar:
    st.image("https://www.gstatic.com/lamda/images/gemini_sparkle_v002_d4735304ff6292a690345.svg", width=50)
    st.title("Multimodal Engine")
    api_key = st.text_input("🔑 Google API Key", type="password")
    if api_key: engine.set_api_key(api_key)
    st.markdown("---")
    lang = st.selectbox("Language", ["English", "Hindi", "Mandarin", "Urdu", "Tamil", "Spanish", "French"])
    mode = st.radio("Input", ["📁 Upload", "🎤 Record"])

col1, col2 = st.columns([4, 1])
with col1: 
    st.title("⚡ Novatech AURA - Audio Understanding and Reasoning Agent")
    st.caption("TRL-4 Functional prototype**")
with col2:
    anim = load_lottie("https://assets5.lottiefiles.com/packages/lf20_w51pcehl.json")
    if anim: st_lottie(anim, height=80)

audio_path = None
if mode == "📁 Upload":
    up = st.file_uploader("Audio", type=['wav', 'mp3', 'm4a'])
    if up:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as t:
            t.write(up.getvalue()); audio_path = t.name
        st.audio(audio_path)
elif mode == "🎤 Record":
    rec = st.audio_input("Voice Input")
    if rec:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as t:
            t.write(rec.getvalue()); audio_path = t.name

if audio_path and st.button("🚀 Analyze with Cloud", type="primary"):
    if not api_key:
        st.error("Please enter your Google API Key in the sidebar first for testing of some online features.")
    else:
        with st.spinner("🚀 Uploading to Studio & Processing..."):
            try:
                t, evt, emo, summ, aud_file = engine.process_audio(audio_path, language=lang)
                G = Network(directed=True); G.add_node("ROOT", label=emo, color="#FF4B4B")
                for x in t: G.add_node(x['speaker'], label=x['speaker'], color="#00ADB5"); G.add_edge("ROOT", x['speaker'])

                st.session_state.update({
                    "transcript": t, "event": evt, "emotion": emo, 
                    "summary": summ, "audio_reply": aud_file, "G": G,
                    "processed": True, "chat_history": []
                })
                st.rerun()
            except Exception as e: st.error(f"Error: {e}")

if st.session_state.processed:
    st.success("✅ Analysis Complete")
    col_sum, col_play = st.columns([3, 1])
    with col_sum: st.info(f"**AI Summary:** {st.session_state.summary}")
    with col_play:
        if st.session_state.audio_reply:
            st.markdown("**🔊 Listen to AI:**")
            st.audio(st.session_state.audio_reply)

    t_data = st.session_state.transcript
    tab1, tab2, tab3 = st.tabs(["📊 Live Timeline", "📝 ASG/Data Grid", "🤖 Gemini QnA"])
    
    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Global Emotion", st.session_state.emotion)
        c2.metric("Key Event", st.session_state.event)
        c3.metric("Segments", len(t_data))
        if t_data:
            df = pd.DataFrame(t_data)
            color_map = {"Panic":"#FF0000", "Joy":"#00FF00", "Neutral":"#808080", "Hostile":"#8B0000"}
            fig = px.timeline(df, x_start="start", x_end="end", y="speaker", color="tone", color_discrete_map=color_map, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if t_data: st.data_editor(pd.DataFrame(t_data), use_container_width=True)

    with tab3:
        for m in st.session_state.chat_history: st.chat_message(m["role"]).write(m["content"])
        if q := st.chat_input("Ask about specific details..."):
            st.session_state.chat_history.append({"role":"user", "content":q})
            st.chat_message("user").write(q)
            with st.spinner("Thinking..."):
                ans = engine.answer_question(t_data, q)
                st.session_state.chat_history.append({"role":"assistant", "content":ans})
                st.chat_message("assistant").write(ans)