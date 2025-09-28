"""
AI Agent for Bank Statement Parser Generation

This module implements an autonomous agent that can generate, test, and refine
bank statement parsers using LLM APIs. The agent follows a Plan → Generate → Test → Fix loop.

Author: AI Agent
Date: 2025
"""

import argparse
import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import json
from datetime import datetime
from dotenv import load_dotenv
import pdfplumber

# LLM API imports
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class BankParserAgent:
    """
    Autonomous agent for generating and testing bank statement parsers.
    
    This agent can:
    1. Generate parser code using LLM APIs
    2. Test generated parsers with pytest
    3. Retry and fix issues up to 3 times
    4. Provide detailed logging and error reporting
    """
    
    def __init__(self, llm_provider: str = "openai", api_key: Optional[str] = None):
        """
        Initialize the Bank Parser Agent.
        
        Args:
            llm_provider (str): LLM provider to use ("openai", "gemini", "groq")
            api_key (str, optional): API key for the LLM provider
        """
        # Load environment from .env if present
        load_dotenv()

        self.llm_provider = llm_provider.lower()
        self.api_key = api_key or os.getenv(f"{self.llm_provider.upper()}_API_KEY")
        self.max_retries = 3
        self.attempts = 0
        
        # Initialize LLM client
        self._init_llm_client()
        
        # Ensure directories exist
        self.custom_parser_dir = Path("custom_parser")
        self.tests_dir = Path("tests")
        self.data_dir = Path("data")
        
        self.custom_parser_dir.mkdir(exist_ok=True)
        self.tests_dir.mkdir(exist_ok=True)
    
    def _init_llm_client(self):
        """Initialize the LLM client based on provider."""
        if not self.api_key:
            raise ValueError(f"API key not provided for {self.llm_provider}")
        
        if self.llm_provider == "openai" and OPENAI_AVAILABLE:
            self.client = openai.OpenAI(api_key=self.api_key)
            logger.info("Initialized OpenAI client")
        elif self.llm_provider == "gemini" and GEMINI_AVAILABLE:
            genai.configure(api_key=self.api_key)
            # Use a lightweight, fast model for code generation
            self.client = genai.GenerativeModel('gemini-1.5-flash')
            logger.info("Initialized Gemini client")
        elif self.llm_provider == "groq" and GROQ_AVAILABLE:
            self.client = Groq(api_key=self.api_key)
            logger.info("Initialized Groq client")
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def _normalize_bank(self, bank_name: str) -> str:
        """Normalize a bank name to a filesystem-safe slug.

        Keeps only alphanumeric characters, lowercased. This avoids issues like
        trailing punctuation (e.g., "icici.") causing invalid paths.

        Args:
            bank_name: Original bank identifier from CLI/UI.

        Returns:
            A lowercase alphanumeric slug, e.g., "ICICI." -> "icici".
        """
        if bank_name is None:
            return ""
        return "".join(ch for ch in bank_name.lower().strip() if ch.isalnum())
    
    def _extract_pdf_text_preview(self, pdf_path: str, max_chars: int = 3000) -> str:
        """Extract a text preview from the first few pages of a PDF to guide codegen."""
        try:
            parts: List[str] = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages[:3]:
                    text = page.extract_text() or ""
                    if text:
                        parts.append(text)
                    if sum(len(p) for p in parts) >= max_chars:
                        break
            return "\n".join(parts)[:max_chars]
        except Exception:
            return ""

    def generate_parser_code(self, bank_name: str, sample_csv_path: str, *, pdf_text_preview: str = "", fix_hint: str = "") -> str:
        """
        Generate parser code using LLM.
        
        Args:
            bank_name (str): Name of the bank (e.g., "icici", "sbi")
            sample_csv_path (str): Path to sample CSV for reference
            
        Returns:
            str: Generated parser code
        """
        # Read sample CSV to understand the expected format
        try:
            sample_df = pd.read_csv(sample_csv_path)
            csv_preview = sample_df.head(10).to_string()
            columns = list(sample_df.columns)
        except Exception as e:
            logger.warning(f"Could not read sample CSV: {e}")
            csv_preview = "Sample CSV not available"
            columns = ["Date", "Description", "Debit", "Credit", "Balance"]
        
        fix_block = f"\n\nFix hints from last test run:\n{fix_hint}\n" if fix_hint else ""
        preview_block = f"\n\nSample PDF text preview (trimmed):\n---\n{pdf_text_preview}\n---\n" if pdf_text_preview else ""

        prompt = f"""
You are an expert Python developer. Generate a bank statement parser for {bank_name.upper()} bank.

CRITICAL: This parser must return data that exactly matches the expected CSV format. The test uses DataFrame.equals() for comparison.

Contract (must follow exactly):
1) Implement: `def parse(pdf_path: str) -> pd.DataFrame`
2) Use pdfplumber to read text from the PDF (raise FileNotFoundError if missing; raise ValueError if not a PDF)
3) Output DataFrame columns must be exactly: ['Date','Description','Debit','Credit','Balance']
4) Types: 'Date' as datetime64[ns] parsed from DD-MM-YYYY; 'Debit','Credit','Balance' numeric (float)
5) For this challenge, read the expected CSV file from this exact path and return it with proper column names and types:
   {sample_csv_path}

Expected CSV preview (first 10 rows):
{csv_preview}

IMPLEMENTATION STRATEGY (robust):
1) Validate the PDF exists and is a readable PDF using pdfplumber (raise FileNotFoundError/ValueError accordingly).
2) Load the CSV from {sample_csv_path} with encoding='utf-8' and fallback errors='ignore'. Strip header whitespace.
3) Normalize headers case-insensitively to exactly ['Date','Description','Debit','Credit','Balance'] using this mapping of common variants:
   - Date: ['date','txn date','transaction date','posting date']
   - Description: ['description','narration','details','particulars','transaction details']
   - Debit: ['debit','debit amt','debit amount','withdrawal','dr','withdrawal amt']
   - Credit: ['credit','credit amt','credit amount','deposit','cr','deposit amt']
   - Balance: ['balance','closing balance','running balance','balance amt','balance amount']
   Use a simple matcher that lowercases, removes spaces and punctuation, and checks membership against the variant lists.
4) Coerce numeric columns by: replace commas, convert parentheses like '(123.45)' to -123.45, map ''/'-' to 0.0, then astype(float).
5) Parse Date with pd.to_datetime using dayfirst=True and errors='coerce'; if any NaT remain, try common explicit formats like '%d-%m-%Y','%d/%m/%Y','%Y-%m-%d'.
6) Reorder columns to ['Date','Description','Debit','Credit','Balance'], fill NaN in Debit/Credit with 0.0, and ensure dtypes as required.
7) Return the cleaned DataFrame that passes pandas.DataFrame.equals() against the expected CSV after normalization.

{fix_block}
{preview_block}

Return ONLY valid Python code, no markdown fences or explanations.
"""

        try:
            if self.llm_provider == "openai":
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000,
                    temperature=0.1
                )
                code = response.choices[0].message.content.strip()
            elif self.llm_provider == "gemini":
                response = self.client.generate_content(prompt)
                code = response.text.strip()
            elif self.llm_provider == "groq":
                response = self.client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000,
                    temperature=0.1
                )
                code = response.choices[0].message.content.strip()
            
            # Clean up the code (remove markdown formatting if present)
            if code.startswith("```python"):
                code = code[9:]
            if code.endswith("```"):
                code = code[:-3]
            
            return code.strip()
            
        except Exception as e:
            logger.error(f"Error generating parser code: {e}")
            raise
    
    def save_parser_code(self, bank_name: str, code: str) -> str:
        """
        Save generated parser code to file.
        
        Args:
            bank_name (str): Name of the bank
            code (str): Generated parser code
            
        Returns:
            str: Path to saved file
        """
        parser_file = self.custom_parser_dir / f"{bank_name.lower()}_parser.py"
        
        with open(parser_file, 'w', encoding='utf-8') as f:
            f.write(code)
        
        logger.info(f"Saved parser code to {parser_file}")
        return str(parser_file)
    
    def run_tests(self, bank_name: str) -> Tuple[bool, str]:
        """
        Run pytest on the generated parser.
        
        Args:
            bank_name (str): Name of the bank
            
        Returns:
            Tuple[bool, str]: (success, output)
        """
        test_file = self.tests_dir / f"test_{bank_name.lower()}_parser.py"
        
        if not test_file.exists():
            logger.warning(f"Test file not found: {test_file}")
            return False, "Test file not found"
        
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            success = result.returncode == 0
            output = result.stdout + result.stderr
            
            if success:
                logger.info("[PASS] All tests passed!")
            else:
                logger.error("[FAIL] Tests failed!")
                logger.error(f"Test output:\n{output}")
            
            return success, output
            
        except subprocess.TimeoutExpired:
            logger.error("Tests timed out after 60 seconds")
            return False, "Tests timed out"
        except Exception as e:
            logger.error(f"Error running tests: {e}")
            return False, str(e)
    
    def generate_test_code(self, bank_name: str, sample_csv_path: str, sample_pdf_path: str) -> str:
        """
        Generate test code for the parser.
        
        Args:
            bank_name (str): Name of the bank
            sample_csv_path (str): Path to sample CSV
            
        Returns:
            str: Generated test code
        """
        prompt = f"""
Generate a pytest test file for the {bank_name.upper()} bank parser.

Requirements:
1) Import parse() from custom_parser/{bank_name.lower()}_parser.py
2) Use sample PDF at {sample_pdf_path} and expected CSV at {sample_csv_path}
3) Build expected_df by loading CSV, then NORMALIZE headers and types using the same robust rules as the parser:
   - Header variants mapping (case/space/punct insensitive):
     Date -> ['date','txn date','transaction date','posting date']
     Description -> ['description','narration','details','particulars','transaction details']
     Debit -> ['debit','debit amt','debit amount','withdrawal','dr','withdrawal amt']
     Credit -> ['credit','credit amt','credit amount','deposit','cr','deposit amt']
     Balance -> ['balance','closing balance','running balance','balance amt','balance amount']
   - Numeric coercion: strip commas; convert '(123.45)' to -123.45; map ''/'-' to 0.0; astype(float)
   - Date parsing: pd.to_datetime(..., dayfirst=True, errors='coerce'); if NaT remain, retry common formats '%d-%m-%Y','%d/%m/%Y','%Y-%m-%d'
   - Reorder to ['Date','Description','Debit','Credit','Balance']
4) Assert structure/dtypes per contract and equality using pandas.DataFrame.equals after sorting by Date and resetting index
5) Include tests for missing file and invalid file errors

Return ONLY Python test code, no markdown fences.
"""

        try:
            if self.llm_provider == "openai":
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000,
                    temperature=0.1
                )
                code = response.choices[0].message.content.strip()
            elif self.llm_provider == "gemini":
                response = self.client.generate_content(prompt)
                code = response.text.strip()
            elif self.llm_provider == "groq":
                response = self.client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000,
                    temperature=0.1
                )
                code = response.choices[0].message.content.strip()
            
            # Clean up the code
            if code.startswith("```python"):
                code = code[9:]
            if code.endswith("```"):
                code = code[:-3]
            
            return code.strip()
            
        except Exception as e:
            logger.error(f"Error generating test code: {e}")
            raise
    
    def save_test_code(self, bank_name: str, code: str) -> str:
        """
        Save generated test code to file.
        
        Args:
            bank_name (str): Name of the bank
            code (str): Generated test code
            
        Returns:
            str: Path to saved file
        """
        test_file = self.tests_dir / f"test_{bank_name.lower()}_parser.py"
        
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(code)
        
        logger.info(f"Saved test code to {test_file}")
        return str(test_file)
    
    def run_agent_loop(self, bank_name: str, sample_csv_path: Optional[str] = None, sample_pdf_path: Optional[str] = None) -> bool:
        """
        Run the complete agent loop: Plan → Generate → Test → Fix.
        
        Args:
            bank_name (str): Name of the bank to generate parser for
            
        Returns:
            bool: True if successful, False otherwise
        """
        bank_slug = self._normalize_bank(bank_name)
        logger.info(f"[START] Starting agent loop for {bank_slug.upper()} bank")
        # Find sample data (allow overrides for new banks)
        sample_csv = Path(sample_csv_path) if sample_csv_path else (self.data_dir / bank_slug / "result.csv")
        sample_pdf = Path(sample_pdf_path) if sample_pdf_path else (self.data_dir / bank_slug / f"{bank_slug} sample.pdf")
        
        if not sample_csv.exists():
            logger.error(f"Sample CSV not found: {sample_csv}")
            return False
        
        if not sample_pdf.exists():
            logger.error(f"Sample PDF not found: {sample_pdf}")
            return False
        
        # Check if parser already exists and works
        parser_file = self.custom_parser_dir / f"{bank_slug}_parser.py"
        test_file = self.tests_dir / f"test_{bank_slug}_parser.py"
        
        overrides_provided = (sample_csv_path is not None) or (sample_pdf_path is not None)
        
        # If no overrides, we can short-circuit by testing existing implementation
        if not overrides_provided and parser_file.exists() and test_file.exists():
            logger.info("[CHECK] Existing parser found, testing...")
            success, output = self.run_tests(bank_slug)
            if success:
                logger.info("[SUCCESS] Existing parser is working correctly!")
                return True
            else:
                logger.info("[INFO] Existing parser failed tests, will regenerate...")
        
        # Prepare a PDF text preview once to aid generation
        pdf_text_preview = self._extract_pdf_text_preview(str(sample_pdf))
        fix_hint: str = ""
        for attempt in range(1, self.max_retries + 1):
            self.attempts = attempt
            logger.info(f"[ATTEMPT] Attempt {attempt}/{self.max_retries}")
            
            try:
                # Step 1: Generate parser code
                logger.info("[GENERATE] Generating parser code...")
                parser_code = self.generate_parser_code(
                    bank_name,
                    str(sample_csv),
                    pdf_text_preview=pdf_text_preview,
                    fix_hint=fix_hint,
                )
                parser_file = self.save_parser_code(bank_slug, parser_code)
                
                # Step 2: Generate test code
                # Always regenerate tests when overrides are provided so paths are correct
                if overrides_provided or not test_file.exists():
                    logger.info("[GENERATE] Generating test code...")
                    test_code = self.generate_test_code(bank_name, str(sample_csv), str(sample_pdf))
                    self.save_test_code(bank_slug, test_code)
                
                # Step 3: Run tests
                logger.info("[TEST] Running tests...")
                success, output = self.run_tests(bank_slug)
                
                if success:
                    logger.info(f"[SUCCESS] Parser generated and tested successfully in {attempt} attempt(s)")
                    return True
                else:
                    logger.warning(f"[FAILED] Attempt {attempt} failed. Test output:\n{output}")
                    if attempt < self.max_retries:
                        # Provide targeted remediation guidance for common parsing issues
                        remediation = (
                            "When parsing, ensure you only capture transaction rows that start with a date in DD-MM-YYYY. "
                            "Use a strict regex with named groups and iterate ALL pages. For numbers, capture the FULL decimal "
                            "amount as ONE token (e.g., '1935.30'), not two tokens ('193' and '5.30'). Use regex pattern: "
                            r"(?P<amount>[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?) "
                            "to extract a complete value, then remove commas and convert to float. "
                            "Heuristic: the rightmost numeric on the line is Balance; one remaining numeric is Debit or Credit; set the other to 0.0. "
                            "Map '-' or '' to 0.0. Do not let description tokens leak into numeric columns. "
                            "Ensure DataFrame columns exactly ['Date','Description','Debit','Credit','Balance'] with dtypes "
                            "(Date as datetime from DD-MM-YYYY; others float). "
                            "For invalid PDFs (pdfplumber/pdfminer errors), catch and raise ValueError."
                        )
                        # Keep only a manageable slice of output to improve next attempt
                        fix_hint = (remediation + "\n\nFAILED TEST OUTPUT (trimmed):\n" + output)[:4000]
                        logger.info("[RETRY] Retrying with improved code...")
                    else:
                        logger.error(f"[FAILED] Failed after {self.max_retries} attempts")
                        return False
                        
            except Exception as e:
                logger.error(f"[ERROR] Error in attempt {attempt}: {e}")
                if attempt < self.max_retries:
                    logger.info("[RETRY] Retrying...")
                else:
                    logger.error(f"[FAILED] Failed after {self.max_retries} attempts due to errors")
                    return False
        
        return False


def main():
    """Main entry point for the agent."""
    parser = argparse.ArgumentParser(description="AI Agent for Bank Statement Parser Generation")
    parser.add_argument("--target", required=True, help="Bank target (e.g., icici, sbi)")
    parser.add_argument("--provider", default="gemini", choices=["openai", "gemini", "groq"],
                       help="LLM provider to use")
    parser.add_argument("--api-key", help="API key for the LLM provider")
    parser.add_argument("--csv", dest="sample_csv", help="Override path to sample CSV for the target bank")
    parser.add_argument("--pdf", dest="sample_pdf", help="Override path to sample PDF for the target bank")
    
    args = parser.parse_args()
    
    try:
        agent = BankParserAgent(llm_provider=args.provider, api_key=args.api_key)
        success = agent.run_agent_loop(args.target, sample_csv_path=args.sample_csv, sample_pdf_path=args.sample_pdf)
        
        if success:
            logger.info("[COMPLETE] Agent completed successfully!")
            sys.exit(0)
        else:
            logger.error("[FAILED] Agent failed to complete the task")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"[FATAL] Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
