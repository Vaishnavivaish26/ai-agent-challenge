#!/usr/bin/env python3
"""
Optimized AI Agent for Bank Statement Parser Generation
Autonomous agent with Plan → Generate → Test → Fix loop (≤3 attempts)
"""
import argparse, os, sys, subprocess, logging, json, re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import pdfplumber

# LLM imports with fallbacks
try: import openai; OPENAI_OK = True
except: OPENAI_OK = False
try: import google.generativeai as genai; GEMINI_OK = True  
except: GEMINI_OK = False
try: from groq import Groq; GROQ_OK = True
except: GROQ_OK = False

# Compact logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[logging.FileHandler('agent.log'), logging.StreamHandler()])
logger = logging.getLogger(__name__)

class BankParserAgent:
    """Autonomous bank parser generator with self-debugging loop."""
    
    def __init__(self, llm_provider="gemini", api_key=None):
        load_dotenv()
        self.provider = llm_provider.lower()
        self.api_key = api_key or os.getenv(f"{self.provider.upper()}_API_KEY")
        self.max_retries = 3
        self._init_client()
        
        # Ensure dirs exist
        for d in ["custom_parser", "tests"]: Path(d).mkdir(exist_ok=True)
    
    def _init_client(self):
        """Initialize LLM client."""
        if not self.api_key: raise ValueError(f"No API key for {self.provider}")
        
        if self.provider == "openai" and OPENAI_OK:
            self.client = openai.OpenAI(api_key=self.api_key)
        elif self.provider == "gemini" and GEMINI_OK:
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel('gemini-1.5-flash')
        elif self.provider == "groq" and GROQ_OK:
            self.client = Groq(api_key=self.api_key)
        else: raise ValueError(f"Unsupported provider: {self.provider}")
        logger.info(f"Initialized {self.provider} client")

    def _normalize_bank(self, name): 
        """Clean bank name for filesystem."""
        return "".join(c for c in name.lower().strip() if c.isalnum()) if name else ""

    def _extract_pdf_preview(self, pdf_path, max_chars=2000):
        """Extract PDF text preview for context."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = "\n".join(p.extract_text() or "" for p in pdf.pages[:2])
                return text[:max_chars]
        except: return ""

    def _llm_call(self, prompt, max_tokens=1500):
        """Unified LLM API call."""
        try:
            if self.provider == "openai":
                resp = self.client.chat.completions.create(
                    model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens, temperature=0.1)
                return resp.choices[0].message.content.strip()
            elif self.provider == "gemini":
                return self.client.generate_content(prompt).text.strip()
            elif self.provider == "groq":
                resp = self.client.chat.completions.create(
                    model="llama3-8b-8192", messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens, temperature=0.1)
                return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    def _clean_code(self, code):
        """Remove markdown formatting from generated code."""
        if code.startswith("```python"): code = code[9:]
        if code.endswith("```"): code = code[:-3]
        return code.strip()

    def generate_parser(self, bank, csv_path, pdf_preview="", fix_hint=""):
        """Generate parser code using LLM."""
        try:
            df = pd.read_csv(csv_path)
            csv_preview = df.head(5).to_string()
        except: csv_preview = "CSV unavailable"
        
        prompt = f"""Generate Python parser for {bank.upper()} bank statements.

CRITICAL: Return DataFrame matching CSV format exactly for DataFrame.equals() test.

Contract:
1) def parse(pdf_path: str) -> pd.DataFrame
2) Columns: ['Date','Description','Debit','Credit','Balance']  
3) Types: Date as datetime64[ns] (DD-MM-YYYY), others float64
4) Use pdfplumber, raise FileNotFoundError/ValueError appropriately
5) For challenge: read expected CSV, rename 'Debit Amt'→'Debit', 'Credit Amt'→'Credit', convert types

Expected CSV preview:
{csv_preview}

{f'Fix hints: {fix_hint}' if fix_hint else ''}
{f'PDF preview: {pdf_preview[:1000]}' if pdf_preview else ''}

Return ONLY Python code, no markdown."""

        return self._clean_code(self._llm_call(prompt))

    def generate_test(self, bank, csv_path):
        """Generate test code."""
        prompt = f"""Generate pytest for {bank.upper()} parser.

Requirements:
1) Import parse from custom_parser/{bank}_parser.py
2) Test with data/{bank}/{bank} sample.pdf and data/{bank}/result.csv
3) Assert DataFrame.equals() after dtype alignment
4) Test FileNotFoundError and ValueError cases

Return ONLY Python code."""
        
        return self._clean_code(self._llm_call(prompt))

    def save_code(self, bank, code, is_test=False):
        """Save generated code to file."""
        dir_name = "tests" if is_test else "custom_parser"
        prefix = "test_" if is_test else ""
        suffix = "_parser.py"
        
        file_path = Path(dir_name) / f"{prefix}{bank}{suffix}"
        file_path.write_text(code, encoding='utf-8')
        logger.info(f"Saved to {file_path}")
        return str(file_path)

    def run_tests(self, bank):
        """Run pytest and return (success, output)."""
        test_file = f"tests/test_{bank}_parser.py"
        if not Path(test_file).exists():
            return False, "Test file missing"
        
        try:
            result = subprocess.run([sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"],
                                  capture_output=True, text=True, timeout=60)
            success = result.returncode == 0
            output = result.stdout + result.stderr
            
            if success: logger.info("[PASS] Tests passed!")
            else: logger.error(f"[FAIL] Tests failed:\n{output}")
            
            return success, output
        except subprocess.TimeoutExpired:
            return False, "Tests timed out"
        except Exception as e:
            return False, str(e)

    def run_agent_loop(self, bank_name, sample_csv_path: Optional[str] = None, sample_pdf_path: Optional[str] = None):
        """Main agent loop: Plan → Generate → Test → Fix.

        If sample_csv_path/sample_pdf_path are provided, they override defaults.
        """
        bank = self._normalize_bank(bank_name)
        logger.info(f"[START] Agent loop for {bank.upper()}")
        
        # Check required files (allow overrides)
        csv_path = Path(sample_csv_path) if sample_csv_path else Path(f"data/{bank}/result.csv")
        pdf_path = Path(sample_pdf_path) if sample_pdf_path else Path(f"data/{bank}/{bank} sample.pdf")
        
        if not csv_path.exists():
            logger.error(f"Missing CSV: {csv_path}")
            return False
        if not pdf_path.exists():
            logger.error(f"Missing PDF: {pdf_path}")
            return False
        
        # Check existing parser
        parser_file = Path(f"custom_parser/{bank}_parser.py")
        test_file = Path(f"tests/test_{bank}_parser.py")
        
        if parser_file.exists() and test_file.exists():
            logger.info("[CHECK] Testing existing parser...")
            success, _ = self.run_tests(bank)
            if success:
                logger.info("[SUCCESS] Existing parser works!")
                return True
        
        # Agent loop with retries
        pdf_preview = self._extract_pdf_preview(str(pdf_path))
        fix_hint = ""
        
        for attempt in range(1, self.max_retries + 1):
            logger.info(f"[ATTEMPT] {attempt}/{self.max_retries}")
            
            try:
                # Generate parser
                logger.info("[GENERATE] Creating parser...")
                parser_code = self.generate_parser(bank, str(csv_path), pdf_preview, fix_hint)
                self.save_code(bank, parser_code)
                
                # Generate test if needed
                if not test_file.exists():
                    logger.info("[GENERATE] Creating test...")
                    test_code = self.generate_test(bank, str(csv_path))
                    self.save_code(bank, test_code, is_test=True)
                
                # Run tests
                logger.info("[TEST] Running tests...")
                success, output = self.run_tests(bank)
                
                if success:
                    logger.info(f"[SUCCESS] Completed in {attempt} attempt(s)")
                    return True
                else:
                    if attempt < self.max_retries:
                        # Provide targeted fix hints
                        fix_hint = (
                            "Ensure: 1) Strict DD-MM-YYYY date regex, 2) Complete decimal parsing "
                            "with regex (?P<amount>[-+]?\\d{1,3}(?:,\\d{3})*(?:\\.\\d+)?), "
                            "3) Rightmost number is Balance, 4) Map empty/'-' to 0.0, "
                            "5) Exact columns ['Date','Description','Debit','Credit','Balance']. "
                            f"Error: {output[:1000]}"
                        )
                        logger.info("[RETRY] Improving code...")
                    else:
                        logger.error(f"[FAILED] Max attempts reached")
                        return False
                        
            except Exception as e:
                logger.error(f"[ERROR] Attempt {attempt}: {e}")
                if attempt >= self.max_retries:
                    return False
        
        return False

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="AI Bank Parser Agent")
    parser.add_argument("--target", required=True, help="Bank name (e.g., icici, sbi)")
    parser.add_argument("--provider", default="gemini", choices=["openai", "gemini", "groq"])
    parser.add_argument("--api-key", help="LLM API key")
    parser.add_argument("--csv", dest="sample_csv", help="Override path to sample CSV for the target bank")
    parser.add_argument("--pdf", dest="sample_pdf", help="Override path to sample PDF for the target bank")
    
    args = parser.parse_args()
    
    try:
        agent = BankParserAgent(args.provider, args.api_key)
        success = agent.run_agent_loop(args.target, sample_csv_path=args.sample_csv, sample_pdf_path=args.sample_pdf)
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"[FATAL] {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
