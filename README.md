# RAG Policy Assistant

A Retrieval-Augmented Generation system for answering HR and company policy questions.

This project uses:

- DPR Context Encoder for policy paragraph embeddings
- DPR Question Encoder for question embeddings
- FAISS for fast similarity search
- GPT-2 for answer generation

## How It Works

```text
Company Policy Document
        ↓
Split into paragraphs
        ↓
DPR Context Encoder
        ↓
FAISS Vector Index
        ↓
User Question
        ↓
DPR Question Encoder
        ↓
Retrieve Top-K Relevant Policy Paragraphs
        ↓
GPT-2 Generates Answer
------------------------------------
Project Structure:

rag-policy-assistant/
├── README.md
├── requirements.txt
├── .gitignore
├── config.yaml
├── data/
│   └── companyPolicies.txt
└── src/
    ├── __init__.py
    ├── main.py
    ├── config.py
    ├── data_loader.py
    ├── retriever.py
    └── generator.py
-------------------------------
Setup:
Create a virtual environment:
python3 -m venv myenv
source myenv/bin/activate

Upgrade pip:
python3 -m pip install --upgrade pip

Install dependencies:
python3 -m pip install -r requirements.txt

Run:
python3 src/main.py

Example Questions:
What is the mobile policy?
How do I submit a reimbursement request?
What is the vacation policy?
What is the remote work policy?

Notes
GPT-2 is used for generation, but stronger instruction-tuned models can improve answer quality.

