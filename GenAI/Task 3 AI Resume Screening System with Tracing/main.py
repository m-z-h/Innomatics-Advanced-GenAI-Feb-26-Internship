import os
from dotenv import load_dotenv
from utils.helpers import load_text_file, get_resume_files, get_job_description

# Load environment variables
load_dotenv()

# Check for required API keys
if not os.getenv("GROQ_API_KEY"):
    print("Error: GROQ_API_KEY not set. Please add your Groq API key to the .env file.")
    exit(1)

if not os.getenv("LANGCHAIN_API_KEY"):
    print("Warning: LANGCHAIN_API_KEY not set. Tracing will not work.")

# Import chains after checking keys
from chains.extraction_chain import extraction_chain
from chains.matching_chain import matching_chain
from chains.scoring_chain import scoring_chain
from chains.explanation_chain import explanation_chain

# Set LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "resume-screening"

def process_resume(resume_path, job_description):
    """Process a single resume through the pipeline."""
    try:
        # Load resume
        resume_text = load_text_file(resume_path)

        # Step 1: Extract skills, tools, experience
        extracted_data = extraction_chain.invoke({"resume": resume_text})
        print(f"Extracted Data for {os.path.basename(resume_path)}: {extracted_data}")

        # Step 2: Match with job description
        matching_result = matching_chain.invoke({
            "extracted_data": extracted_data,
            "job_description": job_description
        })
        print(f"Matching Result: {matching_result}")

        # Step 3: Score the candidate
        scoring_result = scoring_chain.invoke({
            "matching_result": matching_result,
            "job_description": job_description
        })
        score = scoring_result["score"]
        print(f"Fit Score: {score}")

        # Step 4: Generate explanation
        explanation = explanation_chain.invoke({
            "score": score,
            "matching_result": matching_result,
            "extracted_data": extracted_data,
            "job_description": job_description
        })
        print(f"Explanation: {explanation.content}")

        return {
            "resume": os.path.basename(resume_path),
            "extracted_data": extracted_data,
            "matching_result": matching_result,
            "score": score,
            "explanation": explanation.content
        }
    except Exception as e:
        print(f"Error processing {os.path.basename(resume_path)}: {str(e)}")
        return {
            "resume": os.path.basename(resume_path),
            "error": str(e)
        }

def main():
    """Main function to run the resume screening system."""
    # Load job description
    job_description = get_job_description()

    # Get resume files
    resume_files = get_resume_files()

    results = []
    for resume_file in resume_files:
        print(f"\n--- Processing {os.path.basename(resume_file)} ---")
        result = process_resume(resume_file, job_description)
        results.append(result)

    # Summary
    print("\n--- Summary ---")
    for result in results:
        if 'score' in result:
            print(f"{result['resume']}: Score {result['score']}")
        else:
            print(f"{result['resume']}: Error - {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()