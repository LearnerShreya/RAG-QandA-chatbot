"""
preprocess.py
--------------
Cleans and preprocesses the loan dataset for embedding.
- Loads CSV
- Cleans column names
- Handles missing values
- Chunks long text fields
"""

import pandas as pd
from typing import List
import os

CHUNK_SIZE = 300

def load_and_clean_csv(csv_path: str) -> pd.DataFrame:
    """
    Loads the CSV, cleans column names, and handles missing values.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")
    df = pd.read_csv(csv_path)
    # Clean column names
    df.columns = [col.strip().replace(" ", "_").lower() for col in df.columns]
    # Fill missing values with empty string (for text fields)
    df = df.fillna("")
    return df

def chunk_text(text: str, max_length: int = 300) -> List[str]:
    """
    Splits long text into chunks of max_length (for embedding).
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_length):
        chunk = " ".join(words[i:i+max_length])
        if chunk:
            chunks.append(chunk)
    return chunks

def dataframe_to_chunks(df: pd.DataFrame, fields: List[str], max_length: int = CHUNK_SIZE) -> List[str]:
    """
    Converts selected fields from each row into text chunks.
    """
    all_chunks = []
    for _, row in df.iterrows():
        combined = " ".join([str(row[field]) for field in fields if field in row])
        all_chunks.extend(chunk_text(combined, max_length=max_length))
    return all_chunks

if __name__ == "__main__":
    # Example usage
    # The DATA_CSV constant is not defined in this file, so we'll use a placeholder.
    # In a real scenario, this would be defined elsewhere or passed as an argument.
    # For now, we'll just print a placeholder message.
    print("Preprocessing script is running. Please provide a CSV file path.")
