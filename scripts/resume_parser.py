import os
import fitz  # PyMuPDF
import docx2txt
import openai
import json
import csv
import re
from typing import Dict
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Groq API Setup
openai.api_key = os.getenv("GROQ_API_KEY")
if not openai.api_key:
    raise ValueError("GROQ_API_KEY not found in environment or .env file")

openai.api_base = "https://api.groq.com/openai/v1"
MODEL = "llama3-8b-8192"

class ResumeParserAgent:
    def __init__(self, model=MODEL):
        self.model = model

    def extract_text(self, filepath: str) -> str:
        if filepath.endswith(".pdf"):
            return self._extract_pdf(filepath)
        elif filepath.endswith(".docx"):
            return self._extract_docx(filepath)
        else:
            raise ValueError("Unsupported file format. Please use a PDF or DOCX file.")

    def _extract_pdf(self, filepath: str) -> str:
        doc = fitz.open(filepath)
        text = ""
        for page in doc:
            text += page.get_text()
        return text.strip()

    def _extract_docx(self, filepath: str) -> str:
        return docx2txt.process(filepath).strip()

    def parse_with_llama(self, text: str) -> Dict[str, any]:
        prompt = f"""
You are an intelligent HR assistant. Parse the following resume and extract structured information.
Return ONLY valid JSON with no markdown, commentary, or explanations. The format must be:

{{
  "name": "",
  "email": "",
  "phone": "",
  "skills": [],
  "education": [],
  "work_experience": []
}}

Resume Text:
\"\"\"
{text}
\"\"\""""

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts resume information."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        content = response['choices'][0]['message']['content']
        print("Raw LLM output:\n", content)

        try:
            # Safely extract JSON block from the response
            json_str = re.search(r'\{[\s\S]*\}', content).group()
            return json.loads(json_str)
        except Exception as e:
            print("Failed to parse JSON. Raw content was:\n", content)
            raise ValueError(f"Could not parse structured JSON from LLM response: {e}")

    def flatten_section(self, section_list):
        if not isinstance(section_list, list):
            return ""
        flattened = []
        for item in section_list:
            if isinstance(item, dict):
                flat = ", ".join(f"{k}: {v}" for k, v in item.items())
                flattened.append(flat)
            else:
                flattened.append(str(item))
        return " | ".join(flattened)

    def save_to_csv(self, data: Dict[str, any], output_path: str):
        flat_data = {
            "name": data.get("name", ""),
            "email": data.get("email", ""),
            "phone": data.get("phone", ""),
            "skills": ", ".join(data.get("skills", [])),
            "education": self.flatten_section(data.get("education", [])),
            "work_experience": self.flatten_section(data.get("work_experience", []))
        }
        with open(output_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=flat_data.keys())
            writer.writeheader()
            writer.writerow(flat_data)

    def run(self, file_path: str) -> str:
        print("Extracting resume text...")
        text = self.extract_text(file_path)

        print("Parsing resume using LLaMA 3.1 via Groq...")
        structured_data = self.parse_with_llama(text)

        output_csv = f"parsed_resume_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
        self.save_to_csv(structured_data, output_csv)

        print(f"Resume parsed and saved to {output_csv}")
        return output_csv
