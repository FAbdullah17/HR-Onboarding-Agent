import os
import openai
import json
import csv
import re
from typing import List, Dict
from datetime import datetime, timedelta
from dotenv import load_dotenv

# === Load API Keys ===
load_dotenv()
openai.api_key = os.getenv("GROQ_API_KEY")
if not openai.api_key:
    raise ValueError("GROQ_API_KEY not found in .env")

openai.api_base = "https://api.groq.com/openai/v1"
MODEL = "llama3-8b-8192"

class TaskPlannerAgent:
    def __init__(self, model=MODEL):
        self.model = model

    def read_resume_csv(self, filepath: str) -> Dict[str, any]:
        with open(filepath, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            data = next(reader)

        resume_data = {
            "name": data.get("name", ""),
            "email": data.get("email", ""),
            "phone": data.get("phone", ""),
            "skills": [s.strip() for s in data.get("skills", "").split(",") if s.strip()],
            "education": data.get("education", ""),
            "work_experience": data.get("work_experience", "")
        }

        # Try to extract most recent job title from work experience text
        resume_data["job_title"] = self.extract_job_title(data.get("work_experience", ""))
        return resume_data

    def extract_job_title(self, experience_text: str) -> str:
        # Naive heuristic: pick the first role-like word group after "position" or similar
        match = re.search(r'position"\s*:\s*"([^"]+)"', experience_text, re.IGNORECASE)
        if match:
            return match.group(1)
        # Fallback: try to guess from content
        if "Engineer" in experience_text:
            return "Engineer"
        elif "Intern" in experience_text:
            return "Intern"
        elif "Manager" in experience_text:
            return "Manager"
        return "New Hire"

    def build_prompt(self, resume_data: Dict, start_date: str) -> str:
        return f"""
You are an intelligent HR assistant. Based on the following new hire's resume and their most recent job title, generate a personalized 30-day onboarding task plan.

Return a JSON list. Each task must include:
- "task_name"
- "description"
- "due_in_days"
- "assigned_to"
- "category"

Only return valid JSON.

Job Title: {resume_data['job_title']}
Start Date: {start_date}

Resume Summary:
\"\"\"
Name: {resume_data['name']}
Email: {resume_data['email']}
Phone: {resume_data['phone']}
Skills: {", ".join(resume_data['skills'])}
Education: {resume_data['education']}
Experience: {resume_data['work_experience']}
\"\"\"
"""

    def generate_tasks(self, resume_data: Dict, start_date: str) -> List[Dict]:
        prompt = self.build_prompt(resume_data, start_date)

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that creates onboarding plans."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )

        content = response['choices'][0]['message']['content']
        print("Raw LLM Output:\n", content)

        try:
            json_str = re.search(r'\[.*\]', content, re.DOTALL).group()
            return json.loads(json_str)
        except Exception as e:
            print("Failed to parse JSON. Raw content was:\n", content)
            raise ValueError(f"Could not parse onboarding task JSON from LLM response: {e}")

    def save_to_csv(self, tasks: List[Dict], start_date: str, output_path: str):
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        for task in tasks:
            due = task.get("due_in_days", 0)
            task["due_date"] = (start_dt + timedelta(days=int(due))).strftime("%Y-%m-%d")

        fieldnames = ["task_name", "description", "due_in_days", "due_date", "assigned_to", "category"]
        with open(output_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for t in tasks:
                writer.writerow(t)

    def run(self, resume_csv_path: str, start_date: str) -> str:
        print("[INFO] Reading resume CSV...")
        resume_data = self.read_resume_csv(resume_csv_path)

        print(f"Detected role: {resume_data.get('job_title')}")
        print("Generating onboarding tasks...")
        tasks = self.generate_tasks(resume_data, start_date)

        output_txt = f"task_plan_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
        self.save_to_csv(tasks, start_date, output_txt)

        print(f"Task plan saved to {output_txt}")
        return output_txt
