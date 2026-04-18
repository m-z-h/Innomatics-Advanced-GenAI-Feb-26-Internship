from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from prompts.extraction_prompt import extraction_prompt

# Initialize LLM
llm = ChatGroq(model="llama3-8b-8192", temperature=0)

# Create extraction chain
extraction_chain = extraction_prompt | llm | JsonOutputParser()