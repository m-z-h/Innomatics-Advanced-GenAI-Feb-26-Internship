from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from prompts.scoring_prompt import scoring_prompt

# Initialize LLM
llm = ChatGroq(model="llama3-8b-8192", temperature=0)

# Create scoring chain
scoring_chain = scoring_prompt | llm | JsonOutputParser()