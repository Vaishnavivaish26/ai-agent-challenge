"""
Test suite for ICICI Bank Statement Parser

This module contains tests to validate the ICICI parser functionality
and ensure it correctly extracts data from PDF statements.

Author: AI Agent
Date: 2025
"""

import pytest
import pandas as pd
import os
import sys
from pathlib import Path

# Add the custom_parser directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "custom_parser"))

from icici_parser import parse


class TestICICIParser:
    """Test class for ICICI parser functionality."""
    
    @pytest.fixture
    def sample_pdf_path(self):
        """Fixture providing path to sample ICICI PDF."""
        return "data/icici/icici sample.pdf"
    
    @pytest.fixture
    def expected_csv_path(self):
        """Fixture providing path to expected CSV output."""
        return "data/icici/result.csv"
    
    @pytest.fixture
    def expected_df(self, expected_csv_path):
        """Fixture providing expected DataFrame from CSV."""
        if not os.path.exists(expected_csv_path):
            pytest.skip(f"Expected CSV file not found: {expected_csv_path}")
        
        df = pd.read_csv(expected_csv_path)
        
        # Convert date column to datetime for comparison
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
        
        # Rename columns to match parser output
        df = df.rename(columns={
            'Debit Amt': 'Debit',
            'Credit Amt': 'Credit'
        })
        
        # Fill NaN values with 0.0 for numeric columns
        df['Debit'] = df['Debit'].fillna(0.0)
        df['Credit'] = df['Credit'].fillna(0.0)
        
        # Convert numeric columns to float
        df['Debit'] = df['Debit'].astype(float)
        df['Credit'] = df['Credit'].astype(float)
        df['Balance'] = df['Balance'].astype(float)
        
        return df
    
    def test_parse_file_exists(self, sample_pdf_path):
        """Test that the PDF file exists."""
        assert os.path.exists(sample_pdf_path), f"PDF file not found: {sample_pdf_path}"
    
    def test_parse_returns_dataframe(self, sample_pdf_path):
        """Test that parse function returns a pandas DataFrame."""
        if not os.path.exists(sample_pdf_path):
            pytest.skip(f"PDF file not found: {sample_pdf_path}")
        
        result = parse(sample_pdf_path)
        assert isinstance(result, pd.DataFrame), "parse() should return a pandas DataFrame"
    
    def test_parse_dataframe_structure(self, sample_pdf_path):
        """Test that the returned DataFrame has correct structure."""
        if not os.path.exists(sample_pdf_path):
            pytest.skip(f"PDF file not found: {sample_pdf_path}")
        
        result = parse(sample_pdf_path)
        
        # Check columns
        expected_columns = ['Date', 'Description', 'Debit', 'Credit', 'Balance']
        assert list(result.columns) == expected_columns, f"Expected columns {expected_columns}, got {list(result.columns)}"
        
        # Check that DataFrame is not empty
        assert len(result) > 0, "DataFrame should not be empty"
    
    def test_parse_data_types(self, sample_pdf_path):
        """Test that the DataFrame has correct data types."""
        if not os.path.exists(sample_pdf_path):
            pytest.skip(f"PDF file not found: {sample_pdf_path}")
        
        result = parse(sample_pdf_path)
        
        # Check date column
        assert pd.api.types.is_datetime64_any_dtype(result['Date']), "Date column should be datetime type"
        
        # Check numeric columns
        for col in ['Debit', 'Credit', 'Balance']:
            assert pd.api.types.is_numeric_dtype(result[col]), f"{col} column should be numeric type"
    
    def test_parse_matches_expected_data(self, sample_pdf_path, expected_df):
        """Test that parsed data matches expected CSV data using DataFrame.equals()."""
        if not os.path.exists(sample_pdf_path):
            pytest.skip(f"PDF file not found: {sample_pdf_path}")
        
        result = parse(sample_pdf_path)
        
        # Compare DataFrames using pandas DataFrame.equals()
        try:
            # Sort both DataFrames by Date for consistent comparison
            result_sorted = result.sort_values('Date').reset_index(drop=True)
            expected_sorted = expected_df.sort_values('Date').reset_index(drop=True)
            
            # Use DataFrame.equals() for comparison as required
            if result_sorted.equals(expected_sorted):
                print(f"✅ Successfully parsed {len(result)} transactions")
                print("✅ All data matches expected CSV using DataFrame.equals()")
            else:
                # Provide detailed comparison for debugging
                print(f"❌ DataFrame.equals() returned False")
                print(f"Result DataFrame shape: {result_sorted.shape}")
                print(f"Expected DataFrame shape: {expected_sorted.shape}")
                
                # Check column names
                if list(result_sorted.columns) != list(expected_sorted.columns):
                    print(f"Column mismatch: {list(result_sorted.columns)} vs {list(expected_sorted.columns)}")
                
                # Check data types
                print("Result dtypes:")
                print(result_sorted.dtypes)
                print("Expected dtypes:")
                print(expected_sorted.dtypes)
                
                print("\nFirst 5 rows of result:")
                print(result_sorted.head())
                print("\nFirst 5 rows of expected:")
                print(expected_sorted.head())
                
                # Assert using DataFrame.equals() as required by the specification
                assert result_sorted.equals(expected_sorted), \
                    "DataFrames are not equal according to pandas DataFrame.equals()"
            
        except Exception as e:
            print(f"❌ Error during comparison: {e}")
            raise
    
    def test_parse_handles_missing_file(self):
        """Test that parse raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            parse("nonexistent_file.pdf")
    
    def test_parse_handles_invalid_pdf(self, tmp_path):
        """Test that parse handles invalid PDF files gracefully."""
        # Create a dummy text file
        dummy_file = tmp_path / "dummy.txt"
        dummy_file.write_text("This is not a PDF file")
        
        with pytest.raises(ValueError):
            parse(str(dummy_file))


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
