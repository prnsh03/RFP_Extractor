# Emplay Technical Assessment
S PRANESH  - JAIN UNIVERSITY

## RFP Extractor Objective: 

This project is an automated tool for extracting, preprocessing, embedding, and structuring information from Request for Proposal (RFP) documents (PDFs and HTMLs). Using advanced AI techniques, including OpenAI embeddings and Llama-based semantic search, the project organizes the extracted data into predefined fields and stores it as structured JSON.

## Features

- **Multi-format support**: Extracts data from PDFs and HTML files.
- **Preprocessing**: Cleans and tokenizes extracted text for embeddings.
- **OpenAI embeddings**: Uses text-embedding-ada-002 for semantic vectorization.
- **PostgreSQL integration**: Embeddings stored for retrieval and semantic mapping.
- **Llama-based mapping**: Maps structured fields using a pre-trained Llama model.
- **JSON export**: Outputs structured data in structured_data.json.

## Requirements
Ensure you have the following installed:
- Python 3.8+
- PostgreSQL 12+
- Hugging Face Token (for gated Llama models)

### Python Libraries
Install the required libraries using:
''' 
pip install -r requirements.txt
'''

## PostgreSQL Configuration
1. Create a PostgreSQL database (e.g., rfp_extractor).
2. Set up an embeddings table with a column for vectors (pgvector extension required).

## Setup Instructions

## Install PostgreSQL pgvector
### Enable the pgvector extension in your database:
'''
CREATE EXTENSION IF NOT EXISTS vector;
'''

## Environment Variables

### Create a .env file with the following:
'''
OPENAI_API_KEY=your_openai_api_key
DATABASE_URL=postgresql://username:password@localhost:5432/rfp_extractor
HUGGINGFACE_TOKEN=your_huggingface_token
'''

## Hugging Face Access
Request access to the Llama 2 model and add your token to the .env file.

## Directory Structure
'''
.
├── Documents/
│   ├── Bid1/
│   │   ├── file1.pdf
│   │   ├── file2.html
│   ├── Bid2/
│       ├── file3.pdf
│       ├── file4.html
├── extractors/
│   ├── pdf_extractor.py
│   ├── html_extractor.py
│   ├── preprocess.py
├── embeddings/
│   ├── vectorizer.py
├── database/
│   ├── ingestion.py
├── retrieval/
│   ├── semantic_search.py
├── main.py
├── structured_data.json
├── requirements.txt
├── README.md
'''

# Running the Project

1. Start PostgreSQL: Ensure your database server is running.
2. Run the Main Script:
'''
python main.py
'''

## Main Workflow

### Extract and Preprocess:
* Reads PDFs and HTML files from the Documents directory.
* Extracts text using pdf_extractor.py and html_extractor.py.
* Preprocesses the text to clean unnecessary details.

### Create Embeddings:
* Converts the preprocessed text into vector embeddings using OpenAI's text-embedding-ada-002 model.

### Ingest Embeddings:
* Stores embeddings in PostgreSQL using the pgvector extension.

### Semantic Search and Mapping:
* Uses a pre-trained Llama model to semantically map text to 20 predefined fields.

### Export Structured Data:
* Outputs a JSON file (structured_data.json) with all extracted and structured data.

## Predefined Fields

## The tool maps text to the following fields:

* Project Name
* Bid Number
* Deadline
* Eligibility Criteria
* Scope of Work
* Budget
* Contact Information
* Submission Guidelines
* Legal Requirements
* Payment Terms
* Deliverables
* Evaluation Criteria
* Technical Requirements
* Supporting Documents
* Proposal Format
* Contract Period
* Milestones
* Penalty Clauses
* Confidentiality Clauses
* Termination Clauses

## Troubleshooting

### Common Errors
* ModuleNotFoundError: Ensure all dependencies are installed using pip install -r requirements.txt.
* 403 Client Error: Ensure you have access to the Hugging Face model and the token is valid.
* Database errors: Ensure pgvector is enabled in PostgreSQL and credentials are correct.

### Logs
Check console logs for detailed error messages.

## Contact
S Pranesh
+91 6383007285
Jain University
Bangalore - 620016
