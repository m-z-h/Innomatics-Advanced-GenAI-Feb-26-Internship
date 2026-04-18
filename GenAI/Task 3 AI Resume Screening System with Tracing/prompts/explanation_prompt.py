from langchain_core.prompts import PromptTemplate

explanation_prompt = PromptTemplate(
    input_variables=["score", "matching_result", "extracted_data", "job_description"],
    template="""
You are an expert at explaining candidate evaluations.

Task: Provide a clear, detailed explanation of why the given score was assigned, including:
- Strengths of the candidate
- Weaknesses or gaps
- Final recommendation

Score: {score}

Matching Results:
{matching_result}

Extracted Resume Data:
{extracted_data}

Job Description:
{job_description}

Output: A detailed explanation paragraph.
"""
)