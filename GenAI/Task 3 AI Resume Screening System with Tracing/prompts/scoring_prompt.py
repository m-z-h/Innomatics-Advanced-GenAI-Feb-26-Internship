from langchain_core.prompts import PromptTemplate

scoring_prompt = PromptTemplate(
    input_variables=["matching_result", "job_description"],
    template="""
You are an expert at scoring candidate fit for job positions.

Task: Generate a fit score (0-100) based on:
- Skill match percentage
- Experience relevance
- Tools alignment with job requirements

Consider the job description requirements and the matching results.

Matching Results:
{matching_result}

Job Description:
{job_description}

Output format (JSON only):
{{
  "score": 85
}}
"""
)