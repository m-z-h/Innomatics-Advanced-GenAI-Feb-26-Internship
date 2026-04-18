from langchain_core.prompts import PromptTemplate

matching_prompt = PromptTemplate(
    input_variables=["extracted_data", "job_description"],
    template="""
You are an expert at matching resumes to job descriptions.

Task: Compare the extracted resume data with the job description and determine:
- Matched skills: Skills from the resume that match the job requirements
- Missing skills: Required skills from the job that are not in the resume
- Match percentage: Overall percentage of job requirements met (0-100)

Extracted Resume Data:
{extracted_data}

Job Description:
{job_description}

Output format (JSON only):
{{
  "matched_skills": ["skill1", "skill2", ...],
  "missing_skills": ["skill1", "skill2", ...],
  "match_percentage": 85
}}
"""
)