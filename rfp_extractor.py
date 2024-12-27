import os
import re
import json
import fitz  # PyMuPDF
import openai
import psycopg2
import numpy as np
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM

def extract_text_from_pdf(file_path):
    try:
        import fitz
        with fitz.open(file_path) as pdf:
            text = ""
            for page in pdf:
                text += page.get_text()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF {file_path}: {e}")
        return ""

def extract_text_from_html(file_path):
    """Extract text content from an HTML file."""
    try:
        # Opening the HTML file and reading its contents
        with open(file_path, "r", encoding="utf-8") as html_file:
            content = html_file.read()

        # Parsing the HTML content with BeautifulSoup
        soup = BeautifulSoup(content, "html.parser")

        # Removing script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()

        # Extracting visible text
        text = soup.get_text(separator=" ", strip=True)
        return text

    except Exception as e:
        print(f"Error extracting text from HTML {file_path}: {e}")
        return ""
    
def preprocess_text(text):
    """Clean and preprocess extracted text."""
    # Removing extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Removing non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text

# Setting my OpenAI API key
openai.api_key = "my_api_key"

def create_embeddings(documents):
    """Creating vector embeddings for a list of documents."""
    embeddings = []
    try:
        for doc in documents:
            print(documents)
            #using text-embedding-ada-002 model of openai (1,50,000 TPM)
            response = openai.Embedding.create(input=doc, model="text-embedding-ada-002")
            embeddings.append(response['data'][0]['embedding'])
            print(response)
    except Exception as e:
        print(f"Error creating embeddings: {e}")
    return embeddings


def ingest_embeddings_to_postgresql(embeddings):
    """Ingesting vector embeddings into PostgreSQL."""
    connection = connect_to_postgresql()
    if connection is None:
        return

    try:
        cursor = connection.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS embeddings (id SERIAL PRIMARY KEY, vector VECTOR(1536));")
        
        for embedding in embeddings:
            cursor.execute("INSERT INTO embeddings (vector) VALUES (%s);", (embedding,))
        
        connection.commit()
        print("Embeddings successfully ingested into PostgreSQL.")
    except Exception as e:
        print(f"Error ingesting embeddings: {e}")
    finally:
        cursor.close()
        connection.close()

def connect_to_postgresql():
    """Connect to PostgreSQL database."""
    try:
        connection = psycopg2.connect(
            database="fp_prnsh",
            user="postgres",
            password="1256",
            host="localhost",
            port="5432"
        )
        return connection
    except Exception as e:
        print(f"Error connecting to PostgreSQL: {e}")
        return None

def perform_semantic_search(embeddings, query_embedding, top_k=5):
    try:
        # Convert lists to numpy arrays
        embeddings_array = np.array(embeddings)
        query_embedding_array = np.array(query_embedding)

        # Normalize embeddings for cosine similarity
        embeddings_norm = embeddings_array / np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        query_norm = query_embedding_array / np.linalg.norm(query_embedding_array)

        # Compute cosine similarities
        cosine_similarities = np.dot(embeddings_norm, query_norm)

        # Get top-k indices based on similarity scores
        top_k_indices = np.argsort(-cosine_similarities)[:top_k]

        # Retrieve top-k results
        top_k_results = [{"index": idx, "similarity": cosine_similarities[idx]} for idx in top_k_indices]

        return top_k_results

    except Exception as e:
        print(f"Error during semantic search: {e}")
        return []


def map_to_fields(search_results, tokenizer, model, predefined_fields):
    structured_data = []
    try:
        for result in search_results:
            # Retrieve the text from the semantic search result
            text = result.get("text")  # Adjust based on actual result structure
            if not text:
                continue

            # Tokenize the input text
            input_ids = tokenizer.encode(text, return_tensors="pt")

            # Generate output from the language model
            output = model.generate(input_ids, max_new_tokens=50)
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

            # Map generated text to predefined fields
            structured_entry = {}
            for field in predefined_fields:
                structured_entry[field] = extract_field_value(field, generated_text)

            structured_data.append(structured_entry)
    except Exception as e:
        print(f"Error during mapping to fields: {e}")

    return structured_data


def extract_field_value(field, text):
    """
    Extract the value of a specific field from the generated text.
    """
    # Example implementation using simple keyword search:
    for line in text.splitlines():
        if line.startswith(field + ":"):
            return line[len(field) + 1:].strip()
    return None


# Paths
DOCUMENTS_DIR = "D:\Emplay\FINALLLL\RFP_Extractor Final Script\Documents"
OUTPUT_FILE = "structured_data.json"


# Load Llama 3.1 Model and Tokenizer
def load_llm():
    try:
        model_name = "meta-llama/Llama-2-7b-hf"  # Replace with actual Llama 3.1 model name from Hugging Face
        tokenizer = AutoTokenizer.from_pretrained(model_name,use_auth_token="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")
        model = AutoModelForCausalLM.from_pretrained(model_name,use_auth_token="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")
        return tokenizer, model
    except Exception as e:
        print(f"Error loading Llama model: {e}")
        return None, None

def extract_and_preprocess(documents_dir):
    """Extract and preprocess text from all PDF and HTML files."""
    extracted_data = []
    for folder_name in os.listdir(documents_dir):
        folder_path = os.path.join(documents_dir, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if file_name.endswith(".pdf"):
                    text = extract_text_from_pdf(file_path)
                elif file_name.endswith(".html"):
                    text = extract_text_from_html(file_path)
                else:
                    continue
                preprocessed_text = preprocess_text(text)
                extracted_data.append(preprocessed_text)
    return extracted_data


def main():
    # Step 1: Extract and preprocess data
    print("Extracting and preprocessing documents...")
    documents = extract_and_preprocess(DOCUMENTS_DIR)

    # Step 2: Create embeddings
    print("Creating embeddings...")
    embeddings = create_embeddings(documents)

    # Step 3: Ingest embeddings into PostgreSQL
    print("Ingesting embeddings into PostgreSQL...")
    ingest_embeddings_to_postgresql(embeddings)

    # Step 4: Perform semantic search
    print("Performing semantic search...")
    query = "Retrieve key bid details for processing."
    query_embedding = create_embeddings([query])[0]  # Create embedding for the query
    search_results = perform_semantic_search(embeddings, query_embedding, top_k=5)

    # Include the corresponding document text in search results
    for result in search_results:
        result["text"] = documents[result["index"]]  # Map index back to document text


    # Step 5: Map to structured fields using Llama 3.1
    print("Loading Llama 3.1 model...")
    tokenizer, model = load_llm()
    predefined_fields = [
        "Bid Number", "Title", "Due Date", "Bid Submission Type", "Term of Bid",
        "Pre Bid Meeting", "Installation", "Bid Bond Requirement", "Delivery Date",
        "Payment Terms", "Any Additional Documentation Required", "MFG for Registration",
        "Contract or Cooperative to Use", "Model_no", "Part_no", "Product",
        "Contact_info", "Company_name", "Bid Summary", "Product Specification"
    ]
    print("Mapping to structured fields...")
    structured_data = map_to_fields(search_results, tokenizer, model, predefined_fields)

    # Step 6: Save structured data to JSON
    print("Saving structured data...")
    with open(OUTPUT_FILE, "w") as json_file:
        json.dump(structured_data, json_file, indent=4)

    print(f"Structured data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()


