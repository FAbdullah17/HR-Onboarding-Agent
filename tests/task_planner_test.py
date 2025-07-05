from scripts.task_planner import TaskPlannerAgent

print("=== Task Planner Agent (Groq + LLaMA 3.1) ===")
resume_csv = input("Enter the path to the parsed resume CSV file: ").strip()
start_date = input("Enter the employee's start date (YYYY-MM-DD): ").strip()

agent = TaskPlannerAgent()
agent.run(
    resume_csv_path=resume_csv,
    start_date=start_date
)
