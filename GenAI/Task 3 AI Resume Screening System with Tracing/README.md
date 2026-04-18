# AI Resume Screening System with Tracing

A production-style AI-powered resume screening system built with LangChain and LangSmith tracing.

## Features

- **Skill Extraction**: Automatically extracts skills, tools, and experience from resumes
- **Job Matching**: Compares resume data with job requirements
- **Scoring**: Generates fit scores (0-100) based on multiple factors
- **Explanation**: Provides detailed explanations for scores
- **Tracing**: Full observability with LangSmith

## Project Structure

```
ai-resume-screening-system/
├── prompts/           # LangChain prompt templates
├── chains/            # LangChain chains (LCEL)
├── data/              # Sample resumes and job description
├── utils/             # Helper functions
├── main.py            # Main application
├── requirements.txt   # Dependencies
└── README.md          # This file
```

## Setup

1. **Clone or create the project folder**

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set environment variables**

   Create a `.env` file in the project root:

   ```env
   OPENAI_API_KEY=your_openai_api_key
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=your_langsmith_api_key
   LANGCHAIN_PROJECT=resume-screening
   ```

   - Get OpenAI API key from [OpenAI Platform](https://platform.openai.com/)
   - Get LangSmith API key from [LangSmith](https://smith.langchain.com/)

## Usage

Run the system:

```bash
python main.py
```

This will process all sample resumes and display results for each.

## LangSmith Tracing

To view traces in LangSmith:

1. Go to [LangSmith Dashboard](https://smith.langchain.com/)
2. Select project "resume-screening"
3. View runs for each resume processing pipeline

The system includes tracing for:
- Strong candidate (high match)
- Average candidate (moderate match)
- Weak candidate (low match)

## Pipeline Steps

1. **Extraction**: Parse resume for skills, tools, experience
2. **Matching**: Compare with job requirements
3. **Scoring**: Calculate fit score
4. **Explanation**: Generate detailed reasoning

## Customization

- Modify prompts in `prompts/` for different evaluation criteria
- Add more resumes in `data/resumes/`
- Update job description in `data/job_description.txt`
- Adjust LLM model in chains (e.g., use GPT-4 for better accuracy)

## Technologies

- **Python**: Core language
- **LangChain**: Framework for LLM applications
- **OpenAI GPT-3.5**: Language model
- **LangSmith**: Tracing and observability
- **LCEL**: LangChain Expression Language for chains