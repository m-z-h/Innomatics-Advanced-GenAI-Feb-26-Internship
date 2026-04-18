from typing import List

from langchain.chat_models.base import init_chat_model
from langchain_classic.chains.llm import LLMChain, PromptTemplate
from langchain_classic.chains.sequential import SimpleSequentialChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.agents import initialize_agent, AgentType
from langchain_classic.tools import Tool
from langchain_openai import OpenAIEmbeddings
from langchain_classic.vectorstores import VectorStore


def get_llm():
    """Return a configured chat model instance for OpenAI."""
    return init_chat_model(
        model="gpt-3.5-turbo",
        model_provider="openai",
        temperature=0.2,
    )


def basic_llm_call(prompt: str) -> str:
    """Run a direct LLM call and return the generated text."""
    llm = get_llm()
    return llm.predict(prompt)


def build_prompt_template() -> PromptTemplate:
    """Create a reusable prompt template for technical explanations."""
    return PromptTemplate(
        input_variables=["topic", "audience"],
        template=(
            "You are an expert software engineer. Explain {topic} to {audience} in a concise, "
            "technical, and professional style."
        ),
    )


def run_simple_chain(topic: str, audience: str) -> str:
    """Run a simple LLM chain using a prompt template."""
    llm = get_llm()
    prompt = build_prompt_template()
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(topic=topic, audience=audience)


def run_sequential_chain(topic: str) -> str:
    """Run a simple sequential chain that splits a task into two stages."""
    llm = get_llm()
    first_prompt = PromptTemplate(
        input_variables=["topic"],
        template="List the main architectural components of {topic}."
    )
    second_prompt = PromptTemplate(
        input_variables=["components"],
        template=(
            "Write a short technical summary of these components: {components}."
        )
    )
    first_chain = LLMChain(llm=llm, prompt=first_prompt)
    second_chain = LLMChain(llm=llm, prompt=second_prompt)
    sequential_chain = SimpleSequentialChain(chains=[first_chain, second_chain], verbose=False)
    return sequential_chain.run(topic=topic)


def run_memory_example(user_name: str, first_question: str, follow_up_question: str) -> str:
    """Demonstrate conversation memory across two turns."""
    llm = get_llm()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    prompt = PromptTemplate(
        input_variables=["user_name", "question"],
        template=(
            "You are a persistent AI assistant. Here is the conversation history:\n{chat_history}\n"
            "User: {user_name}\nQuestion: {question}\nAnswer in a technical and context-aware manner."
        ),
    )
    chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
    chain.run(user_name=user_name, question=first_question)
    return chain.run(user_name=user_name, question=follow_up_question)


def glossary_lookup(query: str) -> str:
    """A sample deterministic tool that looks up key LangChain terms."""
    glossary = {
        "LangChain": "A framework to build applications with language models, enabling chains, agents, memory, and retrieval.",
        "Vector store": "A similarity index used for semantic search and retrieval over document embeddings.",
        "Prompt template": "A reusable prompt pattern with named input variables for safe formatting.",
        "Agent": "A controller that decides when to invoke tools and composes results into final answers.",
        "Retrieval": "A process that finds relevant documents from a knowledge base using vector search.",
    }
    return glossary.get(query.strip(), "Term not found in glossary. Provide a precise LangChain concept.")


def create_agent_with_tool() -> str:
    """Initialize an agent and run a sample query using a glossary tool."""
    llm = get_llm()
    tool = Tool.from_function(
        func=glossary_lookup,
        name="GlossaryLookup",
        description="Lookup definitions for common LangChain and vector store terminology.",
    )
    agent = initialize_agent([tool], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)
    return agent.run("Define LangChain and explain how a vector store supports retrieval.")


def build_vector_store(documents: List[str]) -> VectorStore:
    """Create a vector store from a list of text documents."""
    embeddings = OpenAIEmbeddings()
    return VectorStore.from_texts(documents, embeddings)


def query_vector_store(vector_store: VectorStore, query: str, k: int = 3) -> List[str]:
    """Run a similarity search against the vector store."""
    results = vector_store.similarity_search(query, k=k)
    return [document.page_content for document in results]


def build_retrieval_chain(documents: List[str]):
    """Build a lightweight retrieval workflow from documents to an LLM answer."""
    vector_store = build_vector_store(documents)

    def retrieval_chain(query: str, k: int = 3) -> str:
        results = query_vector_store(vector_store, query, k=k)
        context = "\n\n".join(results)
        prompt = PromptTemplate(
            input_variables=["query", "context"],
            template=(
                "You are an expert technical assistant. Use the context below to answer the question.\n\n"
                "Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer clearly and concisely."
            ),
        )
        llm = get_llm()
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain.run(query=query, context=context)

    return retrieval_chain
