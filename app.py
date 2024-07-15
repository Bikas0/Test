import streamlit as st
import os
import json
import pandas as pd
from docx import Document
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables
load_dotenv()

# Azure OpenAI credentials
key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint_url = "https://interview-key.openai.azure.com/"
api_version = "2024-05-01-preview"
deployment_id = "interview"

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint_url,
    api_key=key
)

# Streamlit app layout
st.set_page_config(layout="wide")

# Add custom CSS for center alignment
st.markdown("""
    <style>
    .centered-title {
        text-align: center;
        font-size: 2.5em;
        margin-top: 0;
    }
    </style>
    """, unsafe_allow_html=True)

def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_terms_from_contract(contract_text):
    prompt = (
        "You are an AI tasked with analyzing a contract and extracting key terms and constraints. The contract contains "
        "various sections and subsections with terms related to budget constraints, types of allowable work, timelines, "
        "penalties, responsibilities, and other conditions for work execution. Your job is to extract these key terms and "
        "structure them in a clear JSON format, reflecting the hierarchy of sections and subsections. "
        "Ensure to capture all important constraints and conditions specified in the contract text. If a section or subsection "
        "contains multiple terms, list them all.\n\n"
        "Contract text:\n"
        f"{contract_text}\n\n"
        "Provide the extracted terms in JSON format."
    )

    try:
        response = client.chat.completions.create(
            model=deployment_id,
            messages=[
                {"role": "system", "content": "You are an AI specialized in extracting structured data from text documents."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1250,
            n=1,
            stop=None,
            temperature=0.1,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error extracting terms from contract: {e}")
        return None

def analyze_task_compliance(task_description, cost_estimate, contract_terms):
    print("Task D: ", task_description, cost_estimate)
    prompt = (
        "You are an AI tasked with analyzing a task description and its associated cost estimate for compliance with contract conditions. "
        "Below are the key terms and constraints extracted from the contract, followed by a task description and its cost estimate. "
        "Your job is to analyze each task description and specify if it violates any conditions from the contract. "
        "If there are violations, list the reasons for each violation. Provide detailed answers and do not give only true or false answers.\n\n"
        f"Contract terms:\n{json.dumps(contract_terms, indent=4)}\n\n"
        f"Task description:\n{task_description}\n"
        f"Cost estimate:\n{cost_estimate}\n\n"
        "Provide the compliance analysis in a clear JSON format."
    )

    try:
        response = client.chat.completions.create(
            model=deployment_id,
            messages=[
                {"role": "system", "content": "You are an AI specialized in analyzing text for compliance with specified conditions."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1250,
            n=1,
            stop=None,
            temperature=0.1,
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Error analyzing task compliance: {e}")
        return None

def main():
    st.markdown("<h1 class='centered-title'>Contract Compliance Analyzer</h1>", unsafe_allow_html=True)

    # Initialize session state
    if 'contract_terms' not in st.session_state:
        st.session_state.contract_terms = None
    if 'compliance_results' not in st.session_state:
        st.session_state.compliance_results = None

    # File upload buttons one after another
    docx_file = st.sidebar.file_uploader("Upload Contract Document (DOCX)", type="docx", key="docx_file")
    data_file = st.sidebar.file_uploader("Upload Task Descriptions (XLSX or CSV)", type=["xlsx", "csv"], key="data_file")
    submit_button = st.sidebar.button("Submit")

    if submit_button and docx_file and data_file:
        # Extract contract text and terms
        contract_text = extract_text_from_docx(docx_file)
        extracted_terms_json = extract_terms_from_contract(contract_text)

        if extracted_terms_json is None:
            return
        
        try:
            st.session_state.contract_terms = json.loads(extracted_terms_json)
        except json.JSONDecodeError as e:
            st.error(f"JSON decoding error: {e}")
            return
        
        # Read task descriptions and cost estimates from XLSX or CSV
        if data_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            tasks_df = pd.read_excel(data_file)
        else:
            tasks_df = pd.read_csv(data_file)

        compliance_results = []

        # Process tasks sequentially
        for _, row in tasks_df.iterrows():
            result = analyze_task_compliance(row['Task Description'], row['Amount'], st.session_state.contract_terms)
            if result is not None:
                print(result)
                compliance_results.append(result)
        
        st.session_state.compliance_results = compliance_results
        
    col1, col2 = st.columns(2)

    with col1:
        if st.session_state.contract_terms:
            st.write("Extracted Contract Terms:")
            st.json(st.session_state.contract_terms)
            
            # Download button for contract terms
            st.download_button(
                label="Download Contract Terms",
                data=json.dumps(st.session_state.contract_terms, indent=4),
                file_name="contract_terms.json",
                mime="application/json"
            )

    with col2:
        if st.session_state.compliance_results:
            st.write("Compliance Results:")
            st.json(st.session_state.compliance_results)

            # Download button for compliance results
            st.download_button(
                label="Download Compliance Results",
                data=json.dumps(st.session_state.compliance_results, indent=4),
                file_name="compliance_results.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()
