from langchain_groq import ChatGroq
from prompts.explanation_prompt import explanation_prompt

# Initialize LLM
llm = ChatGroq(model="llama3-8b-8192", temperature=0)

# Create explanation chain
explanation_chain = explanation_prompt | llm