import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional
from pydantic import BaseModel, Field
import os
from datetime import datetime

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.document_loaders import PyPDFLoader

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

from config import SUPPORTED_PROVIDERS, PAGE_CONFIG
from utils import get_llm, extract_data, calculate_ratios, generate_peer_review


# Page configuration
st.set_page_config(**PAGE_CONFIG)

# Custom CSS for better styling
st.markdown("""
<style>
    :root {
        --primary-color: #1E88E5;
        --secondary-color: #26A69A;
        --success-color: #43A047;
        --danger-color: #E53935;
    }
    
    .main {
        background-color: #F5F7FA;
    }
    
    .stTabs [data-baseweb="tab-list"] button {
        background-color: #E8EDF2;
        border-radius: 8px;
        padding: 10px 20px;
        margin-right: 5px;
    }
    
    .stTabs [data-baseweb="tab-list"] button:hover {
        background-color: #1E88E5;
        color: white;
    }
    
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    
    .stButton > button {
        background-color: #1E88E5;
        color: white;
        border-radius: 8px;
        padding: 10px 30px;
        font-weight: bold;
        border: none;
    }
    
    .stButton > button:hover {
        background-color: #1565C0;
    }
    
    h1 {
        color: #1E88E5;
        text-align: center;
        margin-bottom: 30px;
    }
    
    h2 {
        color: #26A69A;
        border-bottom: 2px solid #26A69A;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


class LatvianFinancials(BaseModel):
    """Schema for extracting financial data from Latvian Annual Reports."""
    
    uznemuma_nosaukums: str = Field(description="Name of the company (Uzņēmuma nosaukums).")
    parskata_gads: int = Field(description="The reporting year (Pārskata gads/periods).")
    
    neto_apgrozijums: Optional[float] = Field(description="Net turnover (Neto apgrozījums). Remove spaces.")
    bruto_pelna: Optional[float] = Field(description="Gross profit (Bruto peļņa). Remove spaces.")
    parskata_gada_pelna: Optional[float] = Field(description="Profit/loss for the year (Pārskata gada peļņa). Remove spaces.")
    
    ilgtermina_ieguldijumi: Optional[float] = Field(description="Total Long-term investments (Ilgtermiņa ieguldījumi).")
    apgrozamie_lidzekli: Optional[float] = Field(description="Total Current assets (Apgrozāmie līdzekļi).")
    nauda: Optional[float] = Field(description="Cash (Nauda).")
    pasu_kapitals: Optional[float] = Field(description="Total Equity (Pašu kapitāls).")
    istermina_kreditori: Optional[float] = Field(description="Total Current liabilities (Īstermiņa kreditori).")
    ilgtermina_kreditori: Optional[float] = Field(description="Total Long-term liabilities (Ilgtermiņa kreditori).")


def process_reports(files, provider, api_key, language):
    """Process PDF files and generate analysis."""
    if not files:
        st.error("❌ No files uploaded. Please upload at least one PDF.")
        return None, None, None
    
    if not api_key:
        st.error("❌ Please enter an API Key.")
        return None, None, None
    
    try:
        llm = get_llm(provider, api_key)
    except Exception as e:
        st.error(f"❌ API Error: {e}")
        return None, None, None
    
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, file in enumerate(files):
        status_text.text(f"Processing file {idx + 1}/{len(files)}: {file.name}...")
        
        extracted = extract_data(file, llm, LatvianFinancials)
        
        if isinstance(extracted, str):
            if extracted.startswith("Error extracting"):
                st.warning(f"⚠️ {extracted}")
                continue
        
        ratios = calculate_ratios(extracted)
        results.append(ratios)
        progress_bar.progress((idx + 1) / len(files))
    
    if not results:
        st.error("❌ Could not extract data. Check if PDFs are readable or if API Key is valid.")
        return None, None, None
    
    df = pd.DataFrame(results)
    
    # Create visualizations
    df_melted = df.melt(
        id_vars=["Company", "Year"],
        value_vars=["Current Ratio", "Net Margin (%)", "Debt/Equity"],
        var_name="Ratio", 
        value_name="Value"
    )
    
    fig = px.bar(
        df_melted, 
        x="Company", 
        y="Value", 
        color="Ratio", 
        barmode="group",
        title="📊 Peer Comparison: Key Financial Ratios",
        labels={"Value": "Ratio Value", "Company": "Company Name"},
        template="plotly_white",
        height=500
    )
    
    fig.update_layout(
        font=dict(size=12),
        hovermode='x unified',
        plot_bgcolor="rgba(240, 240, 240, 0.5)",
        paper_bgcolor="white"
    )
    
    analysis_text = generate_peer_review(df, language, llm)
    
    status_text.text("✅ Processing complete!")
    progress_bar.empty()
    
    return analysis_text, fig, df


# Header Section
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    # 🇱🇻 Baltic Financial AI Agent
    **Automated Financial Analysis for Latvian Annual Reports**
    """)

st.markdown("""
---
Upload Latvian Annual Reports (PDF) and let AI analyze them instantly.  
Extract key metrics, calculate ratios, and get peer comparisons.
""")

# Main interface
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    api_key = st.text_input(
        "API Key",
        type="password",
        placeholder="sk-... or your-api-key",
        help="Enter your LLM provider API key"
    )
    
    provider = st.selectbox(
        "LLM Provider",
        SUPPORTED_PROVIDERS,
        help="Choose your preferred language model"
    )
    
    language = st.radio(
        "Output Language",
        ["English", "Latvian"],
        help="Choose language for AI analysis"
    )
    
    st.markdown("---")
    st.markdown("### 📖 Instructions")
    st.markdown("""
    1. Enter your API Key
    2. Select LLM Provider
    3. Choose output language
    4. Upload PDF files (Bilance/PZA)
    5. Click 'Analyze' button
    """)

# Main content area
tab1, tab2, tab3 = st.tabs(["📁 Upload & Analyze", "📊 Results", "📈 Dashboard"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📤 Upload PDF Files")
        uploaded_files = st.file_uploader(
            "Upload Latvian Annual Reports (Bilance/PZA)",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDF files"
        )
        
        if uploaded_files:
            st.info(f"✅ {len(uploaded_files)} file(s) uploaded")
            for file in uploaded_files:
                st.caption(f"📄 {file.name} ({file.size / 1024:.2f} KB)")
    
    with col2:
        st.markdown("### 🚀 Action")
        analyze_button = st.button(
            "🚀 Analyze & Generate Report",
            use_container_width=True,
            type="primary"
        )
    
    if analyze_button:
        analysis_text, fig, df = process_reports(
            uploaded_files, provider, api_key, language
        )
        
        if analysis_text and fig is not None and df is not None:
            st.session_state['analysis_text'] = analysis_text
            st.session_state['fig'] = fig
            st.session_state['df'] = df
            st.success("✅ Analysis complete! Check the 'Results' tab.")

with tab2:
    if 'analysis_text' in st.session_state and 'fig' in st.session_state:
        st.markdown("## 📊 Peer Review Analysis")
        st.markdown(st.session_state['analysis_text'])
        
        st.markdown("---")
        st.markdown("## 📈 Financial Ratios Visualization")
        st.plotly_chart(st.session_state['fig'], use_container_width=True)
        
        st.markdown("---")
        st.markdown("## 📋 Extracted Financial Data")
        st.dataframe(st.session_state['df'], use_container_width=True)
        
        # Download options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = st.session_state['df'].to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv,
                "financial_data.csv",
                "text/csv"
            )
        
        with col2:
            st.download_button(
                "📥 Download Analysis",
                st.session_state['analysis_text'],
                "analysis.txt",
                "text/plain"
            )
    else:
        st.info("👈 Upload files and click 'Analyze' to see results here.")

with tab3:
    if 'df' in st.session_state:
        df = st.session_state['df']
        
        st.markdown("## 💰 Financial Metrics Dashboard")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_revenue = df['Revenue (€)'].mean()
            st.metric("📊 Avg Revenue", f"€{avg_revenue:,.0f}")
        
        with col2:
            avg_profit = df['Net Profit (€)'].mean()
            st.metric("💵 Avg Net Profit", f"€{avg_profit:,.0f}")
        
        with col3:
            avg_margin = df['Net Margin (%)'].mean()
            st.metric("📈 Avg Net Margin", f"{avg_margin:.2f}%")
        
        with col4:
            avg_liquidity = df['Current Ratio'].mean()
            st.metric("💧 Avg Liquidity", f"{avg_liquidity:.2f}")
        
        st.markdown("---")
        
        # Additional visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            fig_revenue = px.bar(
                df,
                x="Company",
                y="Revenue (€)",
                title="Revenue by Company",
                template="plotly_white"
            )
            st.plotly_chart(fig_revenue, use_container_width=True)
        
        with col2:
            fig_profit = px.bar(
                df,
                x="Company",
                y="Net Profit (€)",
                title="Net Profit by Company",
                template="plotly_white"
            )
            st.plotly_chart(fig_profit, use_container_width=True)
    else:
        st.info("👈 Upload files and analyze to see dashboard.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🇱🇻 <strong>Baltic Financial AI Agent</strong> | Powered by LangChain & Streamlit</p>
    <p>Built with ❤️ for financial analysis</p>
</div>
""", unsafe_allow_html=True)
