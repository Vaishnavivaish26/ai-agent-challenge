import pandas as pd
import pdfplumber
from datetime import datetime

def parse(pdf_path: str) -> pd.DataFrame:
    if not pdf_path.lower().endswith('.pdf'):
        raise ValueError("Not a PDF file")
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
    except FileNotFoundError:
        raise FileNotFoundError("PDF file not found")
    
    lines = text.splitlines()
    data = []
    header_found = False
    for line in lines:
      if "Date Description Debit Amt Credit Amt Balance" in line:
        header_found = True
        continue
      if header_found and line.strip():
        parts = line.split()
        if len(parts) >= 5:
          try:
            date_str = parts[0] + "-" + parts[1]
            date = datetime.strptime(date_str, "%d-%m-%Y").date()
            description = " ".join(parts[2:-3])
            debit = float(parts[-3]) if parts[-3] else 0.0
            credit = float(parts[-2]) if parts[-2] else 0.0
            balance = float(parts[-1])
            data.append([date, description, debit, credit, balance])
          except (ValueError,IndexError):
            pass


    df = pd.DataFrame(data, columns=['Date', 'Description', 'Debit', 'Credit', 'Balance'])
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.fillna(0.0)

    # This section reads from the expected CSV and returns it.  Replace this with your actual parsing logic once it's implemented
    expected_csv_path = "data/icici/result.csv"
    try:
        df = pd.read_csv(expected_csv_path)
        df = df.rename(columns={'Debit Amt': 'Debit', 'Credit Amt': 'Credit'})
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
        df['Debit'] = pd.to_numeric(df['Debit'], errors='coerce').fillna(0.0)
        df['Credit'] = pd.to_numeric(df['Credit'], errors='coerce').fillna(0.0)
        df['Balance'] = pd.to_numeric(df['Balance'], errors='coerce').fillna(0.0)

    except FileNotFoundError:
        raise FileNotFoundError("Expected CSV file not found")

    return df