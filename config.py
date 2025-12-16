"""Configuration settings for the Baltic Financial AI Agent."""

# Page configuration
PAGE_CONFIG = {
    "page_title": "Baltic Financial AI Agent",
    "page_icon": "🇱🇻",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

# Supported LLM providers
SUPPORTED_PROVIDERS = [
    "OpenAI (GPT-4o)",
    "Groq (Llama 3)",
    "Gemini (2.5 Flash)"
]

# LLM Models mapping
LLM_MODELS = {
    "OpenAI (GPT-4o)": "gpt-4o",
    "Groq (Llama 3)": "llama-3.3-70b-versatile",
    "Gemini (2.5 Flash)": "gemini-2.5-flash"
}

# PDF processing settings
PDF_SETTINGS = {
    "max_pages": 8,
    "supported_formats": ["pdf"],
}

# Financial metrics
FINANCIAL_METRICS = [
    "Current Ratio",
    "Net Margin (%)",
    "Debt/Equity"
]
