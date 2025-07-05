from scripts.resume_parser import ResumeParserAgent

if __name__ == "__main__":
    print("=== Resume Parser Agent (Groq + LLaMA 3.1) ===")
    file_path = input("Enter the path to the resume (PDF or DOCX): ").strip()

    try:
        agent = ResumeParserAgent()
        output_path = agent.run(file_path)
        print(f"\n✅ Resume successfully parsed and saved to: {output_path}")
    except Exception as e:
        print(f"\n❌ Error: {e}")