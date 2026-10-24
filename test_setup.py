import os
from dotenv import load_dotenv
import pandas, sklearn, datasets, pdfplumber

load_dotenv()                          # reads your .env file
key = os.getenv("GEMINI_API_KEY")     # gets the key from it

print("Libraries: OK")
print("Gemini key found:", key is not None and len(key) > 20)