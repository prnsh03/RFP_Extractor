# RFP Extractor Project

Thank you for the opportunity to work on this project. It has been a rewarding learning experience!

## RFP Extractor Objective

This project leverages advanced AI and NLP techniques, including OpenAI embeddings and a Llama-based semantic mapping system, to extract, preprocess, and structure information from Request for Proposal (RFP) documents (PDFs and HTMLs). The output is a structured JSON file containing the key fields from the RFPs, designed for ease of analysis and integration into other systems.

## Features

- **Multi-format support**: Extracts data from PDF and HTML files.
- **Preprocessing**: Cleans, normalizes, and prepares the extracted text.
- **OpenAI embeddings**: Utilizes `text-embedding-ada-002` for semantic representation.
- **PostgreSQL integration**: Embeddings are stored in a PostgreSQL database using the `pgvector` extension.
- **Llama-based mapping**: Maps extracted text to predefined fields using a Hugging Face Llama model.
- **Semantic search**: Enables intelligent mapping of embeddings to structured data fields.
- **JSON export**: Outputs the final structured data as `structured_data.json`.

## Requirements

Ensure you have the following installed and configured:

- **Python 3.8+**
- **PostgreSQL 12+** (with `pgvector` extension enabled)
- **Hugging Face Token** (for accessing gated Llama models)

### Python Libraries

Install required libraries using:
```bash
pip install -r requirements.txt
PostgreSQL Configuration
Create Database: Set up a PostgreSQL database (e.g., rfp_extractor).
Enable pgvector: Run the following command in your PostgreSQL database:
sql
Copy code
CREATE EXTENSION IF NOT EXISTS vector;
Environment Variables
Create a .env file in the root directory with the following content:

plaintext
Copy code
OPENAI_API_KEY=your_openai_api_key
DATABASE_URL=postgresql://username:password@localhost:5432/rfp_extractor
HUGGINGFACE_TOKEN=your_huggingface_token
Replace your_openai_api_key, username, password, and your_huggingface_token with actual values.

Directory Structure
The project directory is organized as follows:

plaintext
Copy code
.
├── Documents/                # Input folder containing RFP documents
│   ├── Bid1/
│   │   ├── file1.pdf
│   │   ├── file2.html
│   ├── Bid2/
│       ├── file3.pdf
│       ├── file4.html
├── extractors/               # Extractor scripts for PDFs and HTMLs
│   ├── pdf_extractor.py
│   ├── html_extractor.py
│   ├── preprocess.py
├── main.py                   # Main script to execute the workflow
├── structured_data.json      # Final structured JSON output
├── requirements.txt          # List of Python dependencies
├── README.md                 # Project documentation
Running the Project
Steps to Execute
Start PostgreSQL: Ensure your database server is running and accessible.
Run the main script:
bash
Copy code
python main.py
Workflow Overview
1. Extract and Preprocess Data
Reads PDF and HTML files from the Documents directory.
Extracts text using extract_text_from_pdf and extract_text_from_html functions.
Cleans and preprocesses text using preprocess_text for better embedding generation.
2. Create Embeddings
Converts preprocessed text into vector embeddings using OpenAI's text-embedding-ada-002 model.
3. Ingest Embeddings
Stores embeddings in a PostgreSQL database table (embeddings) with pgvector for efficient similarity search.
4. Semantic Search
Performs a dummy semantic search (can be enhanced for custom needs) to retrieve relevant embeddings.
5. Map to Structured Fields
Maps the retrieved search results to 20 predefined fields using a pre-trained Llama model and tokenizer.
6. Export to JSON
Saves the mapped structured data as structured_data.json in the root directory.
Predefined Fields
The extracted content is mapped to the following fields:

Title
Author
Date
Company
Executive Summary
Scope
Requirements
Assumptions
Risks
Pricing
Schedule
Deliverables
Terms and Conditions
Attachments
References
Metrics
Stakeholders
Contact Information
Approval
Notes
Troubleshooting
Common Issues
ModuleNotFoundError: Ensure all dependencies are installed using:
bash
Copy code
pip install -r requirements.txt
403 Error from Hugging Face: Ensure you have requested access to the required Llama model and your Hugging Face token is valid.
Database connection issues: Verify that pgvector is enabled in PostgreSQL and credentials in the .env file are correct.
Logs
Detailed logs are printed to the console for debugging purposes.

Contact
S. Pranesh
+91 6383007285
Jain University, Bangalore - 620016
