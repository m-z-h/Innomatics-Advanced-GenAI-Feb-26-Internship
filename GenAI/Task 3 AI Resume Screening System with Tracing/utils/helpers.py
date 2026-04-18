import os

def load_text_file(file_path):
    """Load text from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def get_resume_files(data_dir="data/resumes"):
    """Get list of resume files."""
    resumes_dir = os.path.join(os.getcwd(), data_dir)
    return [os.path.join(resumes_dir, f) for f in os.listdir(resumes_dir) if f.endswith('.txt')]

def get_job_description(data_dir="data/job_description.txt"):
    """Load job description."""
    return load_text_file(os.path.join(os.getcwd(), data_dir))