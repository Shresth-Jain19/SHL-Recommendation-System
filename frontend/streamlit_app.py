import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="SHL Assessment Recommendation", page_icon="🎯", layout="wide")

API_URL = "http://13.233.216.63"

# Custom CSS for modern look
st.markdown("""
    <style>
    .main {
        background-color: #f7f9fa;
    }
    .stButton>button {
        background-color: #4F8BF9;
        color: white;
        border-radius: 8px;
        padding: 0.5em 2em;
        font-size: 1.1em;
        margin-top: 1em;
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 1px solid #4F8BF9;
        padding: 0.5em;
    }
    .stTable {
        background-color: #fff;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 SHL Assessment Recommendation System")
st.write("""
Welcome!  
Enter a **job description** or **natural language query** below, or paste a **public JD URL**.  
You'll get up to 10 recommended SHL assessments tailored to your needs.

**Instructions:**
- For best results, provide a detailed job description or query.
- If using a URL, ensure it is publicly accessible (not behind login).
- Click "Get Recommendations" to see results.
""")

query = st.text_area("Paste your job description, query, or JD URL here:", height=150)
top_k = st.slider("Number of recommendations", min_value=1, max_value=10, value=5)

if st.button("Get Recommendations"):
    if not query.strip():
        st.warning("Please enter a job description, query, or URL.")
    else:
        with st.spinner("Fetching recommendations..."):
            payload = {"query": query}
            try:
                response = requests.post(f"{API_URL}/recommend", json=payload, params={"top_k": top_k}, timeout=60)
                if response.status_code == 200:
                    results = response.json()
                    if results:
                        df = pd.DataFrame(results)
                        # Make assessment names clickable
                        df["name"] = df.apply(lambda row: f'<a href="{row["url"]}" target="_blank">{row["name"]}</a>', axis=1)
                        df = df.rename(columns={
                            "name": "Assessment Name",
                            "remote_testing": "Remote Testing",
                            "adaptive_irt_support": "Adaptive/IRT Support",
                            "duration": "Duration",
                            "test_type": "Test Type"
                        })
                        st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)
                    else:
                        st.info("No recommendations found for your input.")
                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Error contacting API: {e}")

st.markdown("---")
st.caption("Built with ❤️ by Shresth Jain.")
