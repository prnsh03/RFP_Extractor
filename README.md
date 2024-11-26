# Emplay Analytics - Technical Assessment

Thank you for the opportunity to work on this project. It has been a rewarding learning experience!

## Objective

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
```
pip install -r requirements.txt
```
### PostgreSQL Configuration
1. Create Database: Set up a PostgreSQL database (e.g., rfp_extractor).
2. Enable ```pgvector```: Run the following command in your PostgreSQL database:
```
CREATE EXTENSION IF NOT EXISTS vector;
```

## Environment Variables
Create a .env file in the root directory with the following content:

```
OPENAI_API_KEY=your_openai_api_key
DATABASE_URL=postgresql://username:password@localhost:5432/rfp_extractor
HUGGINGFACE_TOKEN=your_huggingface_token
```
Replace   `openai_api_key` , `username`, `password`, and `huggingface_token` with actual values.

## Directory Structure
```
.
├── Documents/                # Input folder containing RFP documents
│   ├── Bid1/
│   │   ├── file1.pdf
│   │   ├── file2.html
│   ├── Bid2/
│       ├── file3.pdf
│       ├── file4.html
├── rfp_extractor.py          # Single script to execute the workflow
├── structured_data.json      # Final structured JSON output
├── requirements.txt          # List of Python dependencies
├── README.md                 # Project documentation
```
## Running the Project
### Steps to Execute
1. **Start PostgreSQL**: Ensure your database server is running and accessible.
2. **Run the main script:**
```
python rfp_extractor.py
```
## Workflow Overview
### 1. Extract and Preprocess Data
* Reads PDF and HTML files from the Documents directory.
* Extracts text using `extract_text_from_pdf` and `extract_text_from_html` functions.
* Cleans and preprocesses text using preprocess_text for better embedding generation.
### 2. Create Embeddings
* Converts preprocessed text into vector embeddings using OpenAI's `text-embedding-ada-002 model`.
### 3. Ingest Embeddings
* Stores embeddings in a PostgreSQL database table (`embeddings`) with `pgvector` for efficient similarity search.
### 4. Semantic Search
* Performs a dummy semantic search (can be enhanced for custom needs) to retrieve relevant embeddings.
### 5. Map to Structured Fields
Maps the retrieved search results to 20 predefined fields using a pre-trained Llama model and tokenizer.
### 6. Export to JSON
Saves the mapped structured data as `structured_data.json` in the root directory.

## Predefined Fields
The extracted content is mapped to the following fields:
```
Bid Number
Title
Due Date
Bid Submission Type
Term of Bid
Pre Bid Meeting
Installation
Bid Bond Requirement
Delivery Date
Payment Terms
Any Additional Documentation Required
MFG for Registration
Contract or Cooperative to Use
Model_no
Part_no
Product
Contact_info
Company_name
Bid Summary
Product Specification
```
## Troubleshooting
### Common Issues
- **ModuleNotFoundError**: Ensure all dependencies are installed using:
```
pip install -r requirements.txt
```
- **403 Error from Hugging Face**: Ensure you have requested access to the required Llama model and your Hugging Face token is valid.
- **Database connection issues**: Verify that `pgvector` is enabled in PostgreSQL and credentials in the `.env` file are correct.

## Contact
S PRANESH 
+91 6383007285
21btrcl085@jainuniversity.ac.in
Jain University, Bangalore 
