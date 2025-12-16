"""Utility functions for the Baltic Financial AI Agent."""

import os
import tempfile
from typing import Optional
from pydantic import BaseModel

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.document_loaders import PyPDFLoader

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

import streamlit as st


def get_llm(provider: str, api_key: str):
    """Factory to initialize the selected LLM."""
    if not api_key:
        raise ValueError("Please enter an API Key.")
    
    if provider == "OpenAI (GPT-4o)":
        return ChatOpenAI(model="gpt-4o", api_key=api_key, temperature=0)
    elif provider == "Groq (Llama 3)":
        return ChatGroq(model_name="llama-3.3-70b-versatile", api_key=api_key, temperature=0)
    elif provider == "Gemini (2.5 Flash)":
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0)
    else:
        raise ValueError("Invalid Provider")


def extract_data(file_obj, llm, schema_class):
    """Loads PDF (Streamlit UploadedFile), extracts text, and parses it into JSON."""
    try:
        # Get file name from Streamlit UploadedFile
        file_name = getattr(file_obj, 'name', 'unknown.pdf')
        
        # Create temporary file for PDF processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_obj.read())
            tmp_path = tmp_file.name
        
        try:
            # Load PDF
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            
            # Extract text from first 8 pages
            full_text = "\n".join([p.page_content for p in pages[:8]])
            # Remove non-standard characters
            full_text = "".join(c for c in full_text if ord(c) < 0x10000)
            
            # Parse with LLM
            parser = PydanticOutputParser(pydantic_object=schema_class)
            
            prompt = PromptTemplate(
                template="""
                You are an expert Accountant for Baltic companies.
                Extract the following financial figures from the Latvian Annual Report text below.

                IMPORTANT RULES:
                1. Latvian reports use spaces as thousand separators (e.g., "29 052 268"). You MUST remove spaces and return a pure number (29052268).
                2. If a value is in brackets (), it is negative.
                3. Look for "Rindas kods" to identify correct rows if names are ambiguous.
                4. Extract data for the "Reporting Year" (Pārskata gads), not the previous year.

                RAW TEXT:
                {text}

                {format_instructions}
                """,
                input_variables=["text"],
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )
            
            chain = prompt | llm | parser
            result = chain.invoke({"text": full_text})
            
            return result
        
        finally:
            # Always clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    except Exception as e:
        file_name = getattr(file_obj, 'name', 'unknown.pdf')
        error_msg = f"Error extracting {file_name}: {str(e)}"
        return error_msg


def calculate_ratios(data) -> dict:
    """Calculates key ratios mathematically."""
    ratios = {
        "Company": data.uznemuma_nosaukums,
        "Year": data.parskata_gads,
        "Revenue (€)": data.neto_apgrozijums,
        "Net Profit (€)": data.parskata_gada_pelna,
    }
    
    # Current Ratio (Liquidity)
    if data.apgrozamie_lidzekli and data.istermina_kreditori:
        ratios["Current Ratio"] = round(data.apgrozamie_lidzekli / data.istermina_kreditori, 2)
    else:
        ratios["Current Ratio"] = 0.0
    
    # Net Margin (Profitability)
    if data.parskata_gada_pelna and data.neto_apgrozijums:
        ratios["Net Margin (%)"] = round((data.parskata_gada_pelna / data.neto_apgrozijums) * 100, 2)
    else:
        ratios["Net Margin (%)"] = 0.0
    
    # Debt/Equity (Solvency)
    total_debt = (data.istermina_kreditori or 0) + (data.ilgtermina_kreditori or 0)
    if data.pasu_kapitals:
        ratios["Debt/Equity"] = round(total_debt / data.pasu_kapitals, 2)
    else:
        ratios["Debt/Equity"] = 0.0
    
    return ratios


def generate_peer_review(df, language: str, llm) -> str:
    """Uses LLM to write a comparative analysis of the calculated data."""
    data_summary = df.to_markdown(index=False)
    
    lang_instruction = "Write the response in English." if language == "English" else "Raksti atbildi Latviešu valodā."
    
    prompt_text = f"""
    You are a Senior Financial Analyst.
    Review the following financial data for Baltic companies:

    {data_summary}

    Perform a Peer Review:
    1. Compare the *Profitability* (Net Margin). Which company is more efficient?
    2. Analyze *Liquidity* (Current Ratio). Are any companies at risk (ratio < 1.0)?
    3. Evaluate *Solvency* (Debt/Equity). Who is more leveraged?
    4. Provide a concluding recommendation.

    {lang_instruction}
    """
    
    if isinstance(llm, ChatOpenAI) or isinstance(llm, ChatGroq) or isinstance(llm, ChatGoogleGenerativeAI):
        response = llm.invoke(prompt_text)
        return response.content
    return "LLM Error"
