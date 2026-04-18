from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from prompts.matching_prompt import matching_prompt

# Initialize LLM
llm = ChatGroq(model="llama3-8b-8192", temperature=0)

# Create matching chain
matching_chain = matching_prompt | llm | JsonOutputParser()