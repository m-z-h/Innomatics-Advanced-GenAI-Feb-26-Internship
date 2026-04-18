from langchain_core.prompts import PromptTemplate

extraction_prompt = PromptTemplate(
    input_variables=["resume"],
    template="""
You are an expert at extracting information from resumes.

Task: Extract the following from the provided resume text:
- Skills: List of technical skills mentioned
- Tools: List of tools, frameworks, or software mentioned
- Experience: Total years of experience as a string (e.g., "5 years")

Important: 
- Only extract information that is explicitly mentioned in the resume.
- Do NOT hallucinate or assume skills/tools/experience not present.
- Return ONLY valid JSON with no additional text.

Resume:
{resume}

Output format:
{{
  "skills": ["skill1", "skill2", ...],
  "tools": ["tool1", "tool2", ...],
  "experience": "X years"
}}
"""
)