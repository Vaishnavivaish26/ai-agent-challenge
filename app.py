"""
Streamlit UI for AI Agent Bank Statement Parser

This module provides a web interface for the AI agent that generates
and tests bank statement parsers.

Author: AI Agent
Date: 2025
"""

import streamlit as st
import subprocess
import sys
import os
import time
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI Agent Bank Parser",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables from .env
load_dotenv()

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #2563eb;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-success {
        color: #10b981;
        font-weight: 600;
    }
    .status-error {
        color: #ef4444;
        font-weight: 600;
    }
    .status-warning {
        color: #f59e0b;
        font-weight: 600;
    }
    .log-container {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 0.5rem;
        padding: 1rem;
        font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Roboto Mono', monospace;
        font-size: 0.875rem;
        max-height: 400px;
        overflow-y: auto;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Available banks
AVAILABLE_BANKS = ["icici", "sbi", "hdfc", "axis", "kotak"]

def check_api_keys() -> dict:
    """Check which API keys are available."""
    api_keys = {}
    
    # Check Gemini (primary)
    if os.getenv("GEMINI_API_KEY"):
        api_keys["Gemini"] = "✅ Available"
    else:
        api_keys["Gemini"] = "❌ Not set"
    
    return api_keys

def run_agent_command(bank: str, provider: str, api_key: Optional[str] = None) -> Tuple[bool, str]:
    """
    Run the agent command and capture output.
    
    Args:
        bank (str): Bank name
        provider (str): LLM provider
        api_key (str, optional): API key
        
    Returns:
        Tuple[bool, str]: (success, output)
    """
    cmd = [sys.executable, "agent.py", "--target", bank, "--provider", provider]
    
    # API key is read from environment by the agent; do not pass via CLI
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout
        )
        
        success = result.returncode == 0
        output = result.stdout + result.stderr
        
        return success, output
        
    except subprocess.TimeoutExpired:
        return False, "Command timed out after 5 minutes"
    except Exception as e:
        return False, f"Error running command: {str(e)}"

def load_parsed_data(bank: str, pdf_path: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Load and display parsed data from the generated parser.
    
    Args:
        bank (str): Bank name
        
    Returns:
        Optional[pd.DataFrame]: Parsed data or None if error
    """
    try:
        # Import the generated parser
        sys.path.insert(0, str(Path("custom_parser")))
        
        parser_module = __import__(f"{bank.lower()}_parser")
        parse_func = getattr(parser_module, "parse")
        
        # Determine PDF path (prefer provided/uploaded)
        effective_pdf_path = pdf_path or f"data/{bank.lower()}/{bank.lower()} sample.pdf"
        
        if not os.path.exists(effective_pdf_path):
            st.error(f"PDF not found: {effective_pdf_path}")
            return None
        
        # Parse the PDF
        df = parse_func(effective_pdf_path)
        return df
        
    except Exception as e:
        st.error(f"Error loading parsed data: {str(e)}")
        return None

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🏦 AI Bank Parser</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Automatically generate and test bank statement parsers using AI</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key Status
        st.subheader("API Key Status")
        api_status = check_api_keys()
        for provider, status in api_status.items():
            st.text(f"{provider}: {status}")
        
        # Provider Selection (Default to Gemini)
        st.subheader("AI Provider")
        provider = st.selectbox(
            "Select AI Provider",
            ["gemini"],
            index=0,
            disabled=True
        )
        
        st.caption("Set your key in a .env file as GEMINI_API_KEY and restart the app.")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🎯 Bank Selection")
        selected_bank = st.selectbox(
            "Select Bank to Generate Parser For",
            AVAILABLE_BANKS,
            index=0
        )
        
        # Check if sample data exists
        sample_csv = f"data/{selected_bank}/result.csv"
        sample_pdf = f"data/{selected_bank}/{selected_bank} sample.pdf"
        
        col_csv, col_pdf = st.columns(2)
        with col_csv:
            if os.path.exists(sample_csv):
                st.success(f"✅ Sample CSV found: {sample_csv}")
            else:
                st.error(f"❌ Sample CSV not found: {sample_csv}")
        
        with col_pdf:
            if os.path.exists(sample_pdf):
                st.success(f"✅ Sample PDF found: {sample_pdf}")
            else:
                st.error(f"❌ Sample PDF not found: {sample_pdf}")
    
    with col2:
        st.header("🚀 Actions")
        
        # Run Agent Button
        if st.button("🤖 Generate Parser", type="primary", use_container_width=True):
            if not os.getenv("GEMINI_API_KEY"):
                st.error("Please provide your Gemini API key")
            else:
                with st.spinner("AI Agent is generating your bank parser..."):
                    # Create a placeholder for logs
                    log_placeholder = st.empty()
                    
                    # Run the agent
                    success, output = run_agent_command(selected_bank, provider, None)
                    
                    # Display results
                    if success:
                        st.success("✅ Parser generated successfully!")
                        st.session_state.agent_success = True
                        st.session_state.agent_output = output
                    else:
                        st.error("❌ Parser generation failed")
                        st.session_state.agent_success = False
                        st.session_state.agent_output = output
                    
                    # Show logs
                    with log_placeholder.container():
                        st.subheader("📋 Generation Logs")
                        st.markdown('<div class="log-container">', unsafe_allow_html=True)
                        st.text(output)
                        st.markdown('</div>', unsafe_allow_html=True)
        
        # Test Parser Button
        if st.button("🧪 Test Parser", use_container_width=True):
            with st.spinner("Testing parser..."):
                # Run pytest
                test_file = f"tests/test_{selected_bank}_parser.py"
                if os.path.exists(test_file):
                    result = subprocess.run(
                        [sys.executable, "-m", "pytest", test_file, "-v"],
                        capture_output=True,
                        text=True
                    )
                    
                    if result.returncode == 0:
                        st.success("✅ All tests passed!")
                    else:
                        st.error("❌ Tests failed!")
                    
                    st.subheader("Test Output")
                    st.text(result.stdout + result.stderr)
                else:
                    st.error(f"Test file not found: {test_file}")

        # Upload parsing disabled; app uses bundled sample PDF
    
    # Results Section
    if hasattr(st.session_state, 'agent_success') and st.session_state.agent_success:
        st.header("📊 Parsed Data")
        
        # Load and display parsed data
        parsed_data = load_parsed_data(selected_bank)
        
        if parsed_data is not None:
            st.success(f"Successfully parsed {len(parsed_data)} transactions")
            
            # Display basic statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Transactions", len(parsed_data))
            with col2:
                st.metric("Date Range", f"{parsed_data['Date'].min().strftime('%d-%m-%Y')} to {parsed_data['Date'].max().strftime('%d-%m-%Y')}")
            with col3:
                total_debit = parsed_data['Debit'].sum() if 'Debit' in parsed_data.columns else 0
                st.metric("Total Debit", f"₹{total_debit:,.2f}")
            with col4:
                total_credit = parsed_data['Credit'].sum() if 'Credit' in parsed_data.columns else 0
                st.metric("Total Credit", f"₹{total_credit:,.2f}")
            
            # Display data table
            st.subheader("Transaction Data")
            st.dataframe(parsed_data, use_container_width=True)
            
            # Download button
            csv = parsed_data.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"{selected_bank}_parsed_data.csv",
                mime="text/csv"
            )
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**AI Agent Features:** "
        "Autonomous Parser Generation | Self-Correcting Loop | Automated Testing | Multi-Bank Support"
    )

if __name__ == "__main__":
    main()
