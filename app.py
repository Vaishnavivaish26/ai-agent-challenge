"""
Streamlined Streamlit UI for AI Agent Bank Statement Parser
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
st.set_page_config(page_title="Bank Statement Parser (AI Agent)", page_icon="🏦", layout="wide")

# Load environment variables from .env
load_dotenv()


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

def run_agent_command(bank: str, provider: str, api_key: Optional[str] = None,
                      sample_pdf: Optional[str] = None, sample_csv: Optional[str] = None) -> Tuple[bool, str]:
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
    if sample_pdf:
        cmd += ["--pdf", sample_pdf]
    if sample_csv:
        cmd += ["--csv", sample_csv]
    
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

def load_parsed_data(bank: str, pdf_path: str) -> Optional[pd.DataFrame]:
    """Call generated parse() to get a DataFrame from the uploaded PDF."""
    try:
        sys.path.insert(0, str(Path("custom_parser")))
        parse_func = getattr(__import__(f"{bank.lower()}_parser"), "parse")
        if not os.path.exists(pdf_path):
            st.error(f"PDF not found: {pdf_path}")
            return None
        return parse_func(pdf_path)
    except Exception as e:
        st.error(f"Error parsing PDF: {str(e)}")
        return None

def main():
    """Main Streamlit application."""
    # Header Section
    st.title("🏦 Bank Statement Parser (AI Agent)")
    st.caption("Upload your bank PDF or CSV and view transactions in a structured table.")

    # Sidebar
    with st.sidebar:
        st.header("Options")
        selected_bank = st.selectbox("Select Bank", AVAILABLE_BANKS, index=0)
        up_dir = Path("uploads"); up_dir.mkdir(exist_ok=True)
        uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"], key="pdf")
        uploaded_csv = st.file_uploader("Upload CSV", type=["csv"], key="csv")
        run_clicked = st.button("Run Agent")

    # Main Content
    st.markdown("---")
    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.header("Bank Selection")
        st.write(f"Selected bank: `{selected_bank}`")
        st.write(f"Uploaded PDF: `{uploaded_pdf.name if uploaded_pdf else 'None'}`")
        st.write(f"Uploaded CSV: `{uploaded_csv.name if uploaded_csv else 'None'}`")
    with col_right:
        st.header("Actions")
        generate_clicked_main = st.button("Generate Parser", key="run_main")
        test_clicked_main = st.button("Test Parser", key="test_main")
        # Keep main button behavior in sync with sidebar
        if generate_clicked_main:
            st.session_state["generate_clicked"] = True
        if test_clicked_main:
            st.session_state["test_clicked"] = True

    step1 = st.container()
    step2 = st.container()
    step3 = st.container()
    step4 = st.container()

    with step1:
        st.subheader("Step 1: Inputs")
        pdf_name = uploaded_pdf.name if uploaded_pdf else "None"
        csv_name = uploaded_csv.name if uploaded_csv else "None"
        st.write(f"Selected bank: `{selected_bank}`")
        st.write(f"Uploaded PDF: `{pdf_name}` | Uploaded CSV: `{csv_name}`")

        # Show defaults that will be used if nothing uploaded
        default_pdf = Path(f"data/{selected_bank}/{selected_bank} sample.pdf")
        default_csv = Path(f"data/{selected_bank}/result.csv")
        st.caption("Defaults for selected bank (used when no uploads are provided):")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.text_input("Default PDF", value=str(default_pdf), key="default_pdf_display")
            st.write("✅ Exists" if default_pdf.exists() else "❌ Not found")
        with col_d2:
            st.text_input("Default CSV", value=str(default_csv), key="default_csv_display")
            st.write("✅ Exists" if default_csv.exists() else "❌ Not found")

    parsed_df: Optional[pd.DataFrame] = None

    # If CSV is uploaded, load and display directly
    with step2:
        if uploaded_csv:
            st.subheader("Step 2: Loaded CSV")
            csv_path = Path("uploads") / (uploaded_csv.name if uploaded_csv.name.endswith(".csv") else "result.csv")
            with open(csv_path, "wb") as f:
                f.write(uploaded_csv.getbuffer())
            try:
                parsed_df = pd.read_csv(csv_path)
                st.dataframe(parsed_df, use_container_width=True)
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")

    # If PDF is uploaded, run the agent and parse to DataFrame
    with step3:
        # Determine if generation was requested from sidebar or main
        generate_triggered = (
            run_clicked or st.session_state.get("generate_clicked", False)
        )
        if generate_triggered:
            st.subheader("Step 3: Agent Run & Parsing")
            # Resolve effective inputs: uploads take precedence, else use bundled samples
            default_pdf = Path(f"data/{selected_bank}/{selected_bank} sample.pdf")
            default_csv = Path(f"data/{selected_bank}/result.csv")

            if uploaded_pdf:
                pdf_path = Path("uploads") / (uploaded_pdf.name if uploaded_pdf.name.endswith(".pdf") else "statement.pdf")
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_pdf.getbuffer())
            else:
                pdf_path = default_pdf

            # Only pass overrides if the user uploaded files
            pdf_override = str(pdf_path) if uploaded_pdf else None
            csv_override = str(csv_path) if uploaded_csv else None

            # Effective paths banner (uploaded overrides > defaults)
            effective_pdf = pdf_override or str(default_pdf)
            effective_csv = csv_override or (str(default_csv) if default_csv.exists() else "<none>")
            st.session_state["effective_pdf_path"] = effective_pdf
            st.session_state["effective_csv_path"] = effective_csv
            st.info(f"Using PDF: {effective_pdf}\nUsing Expected CSV: {effective_csv}")

            # If only PDF is uploaded (no CSV), warn that default CSV will be used
            if uploaded_pdf and not uploaded_csv:
                st.warning("No CSV uploaded. The agent will use the default expected CSV for the selected bank, so results may look identical to the sample.")

            if not os.getenv("GEMINI_API_KEY"):
                st.warning("Set GEMINI_API_KEY in your .env for generation.")

            with st.spinner("Running AI agent to (re)generate parser and tests..."):
                ok, logs = run_agent_command(selected_bank, "gemini", None, pdf_override, csv_override)
                st.text_area("Agent Logs", logs, height=200)

            if ok:
                # Let the final 'Parsed Data' section render the table/metrics once
                display_pdf = str(pdf_path) if uploaded_pdf else str(default_pdf)
                df = load_parsed_data(selected_bank, display_pdf)
                if df is not None:
                    parsed_df = df
            else:
                st.error("Agent run failed. See logs above.")

    # Parsed Data section with metrics similar to earlier implementation
    if parsed_df is not None:
        st.markdown("---")
        st.subheader("Parsed Data")
        st.success(f"Successfully parsed {len(parsed_df)} transactions")

        # Metrics row
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Transactions", len(parsed_df))
        with c2:
            try:
                min_d = parsed_df['Date'].min()
                max_d = parsed_df['Date'].max()
                if pd.api.types.is_datetime64_any_dtype(parsed_df['Date']):
                    st.metric("Date Range", f"{min_d.strftime('%d-%m-%Y')} to {max_d.strftime('%d-%m-%Y')}")
                else:
                    st.metric("Date Range", f"{str(min_d)} to {str(max_d)}")
            except Exception:
                st.metric("Date Range", "-")
        with c3:
            total_debit = parsed_df['Debit'].sum() if 'Debit' in parsed_df.columns else 0
            st.metric("Total Debit", f"₹{total_debit:,.2f}")
        with c4:
            total_credit = parsed_df['Credit'].sum() if 'Credit' in parsed_df.columns else 0
            st.metric("Total Credit", f"₹{total_credit:,.2f}")

        st.subheader("Transaction Data")
        st.dataframe(parsed_df, use_container_width=True)

        # Step 4: Download
        with step4:
            st.subheader("Step 4: Download")
            st.download_button("Download CSV", parsed_df.to_csv(index=False), file_name=f"{selected_bank}_parsed.csv", mime="text/csv")

    # Optional: run tests from main Actions
    if st.session_state.get("test_clicked", False):
        test_file = f"tests/test_{selected_bank}_parser.py"
        if os.path.exists(test_file):
            result = subprocess.run([sys.executable, "-m", "pytest", test_file, "-v"], capture_output=True, text=True)
            st.subheader("Test Output")
            st.text(result.stdout + result.stderr)
            if result.returncode == 0:
                st.success("All tests passed!")
            else:
                st.error("Tests failed. See output above.")
        else:
            st.warning(f"Test file not found: {test_file}")
        # reset flag after showing
        st.session_state["test_clicked"] = False
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**AI Agent Features:** "
        "Autonomous Parser Generation | Self-Correcting Loop | Automated Testing | Multi-Bank Support"
    )

if __name__ == "__main__":
    main()
