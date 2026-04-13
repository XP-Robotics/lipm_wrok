import os
import argparse
from bs4 import BeautifulSoup
from langchain_community.document_loaders import BSHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_google_genai import ChatGoogleGenerativeAI

def build_rag_system(directory_path, category_file):
    links = []
    
    # 1. Parse robotics.html to find linked robotics pages
    with open(category_file, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            # Look for links targeting the robotics directory
            if href.startswith("../robotics/") and href.endswith(".html"):
                links.append(os.path.normpath(os.path.join(os.path.dirname(category_file), href)))
    
    # Remove duplicates
    links = list(set(links))
    # Add the main category page as well
    links.append(category_file)
    
    print(f"Found {len(links)} documents to index based on {category_file}...")
    
    # 2. Extract content from the HTML files
    docs = []
    for link in links:
        if os.path.exists(link):
            loader = BSHTMLLoader(link)
            docs.extend(loader.load())
        else:
            print(f"Warning: file not found: {link}")
            
    # 3. Split the docs into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(f"Created {len(splits)} chunks from {len(docs)} documents.")

    # 4. Generate Embeddings (using a local open source model)
    print("Initializing local HuggingFace embeddings (downloads standard model on first run)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    return vectorstore

def main():
    parser = argparse.ArgumentParser(description="Run RAG on robotics HTML docs")
    parser.add_argument("--query", "-q", type=str, default="What are the equations of motion for a point mass model?", help="Query for the RAG system")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    category_file = os.path.join(base_dir, "category", "robotics.html")
    
    if not os.path.exists(category_file):
        print(f"Error: Could not find '{category_file}'. Please run from repo root.")
        return

    # To generate answers we need an LLM. We will check for the Gemini API key.
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("\n[NOTE] Please set your GOOGLE_API_KEY environment variable to use the Gemini LLM for answering.")
        print("Example: export GOOGLE_API_KEY='your_api_key'")
        print("Proceeding without LLM generation... we will only show the retrieved context.")
        llm = None
    else:
        # Use Gemini
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    # Build the RAG index
    vectorstore = build_rag_system(base_dir, category_file)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    if llm:
        system_prompt = (
            "You are an assistant for question-answering tasks focusing on robotics. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, say that you don't know. "
            "Use three sentences maximum and keep the answer concise.\n\n"
            "{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        print(f"\n--- Question: {args.query} ---")
        response = rag_chain.invoke({"input": args.query})
        
        print("\nAnswer:")
        print(response["answer"])
        print("\nSources:")
        for doc in response["context"]:
            print("-", doc.metadata["source"])
    else:
        print(f"\n--- Question: {args.query} ---")
        print("\nRetrieved Context (LLM not configured to answer):")
        docs = retriever.invoke(args.query)
        for i, doc in enumerate(docs):
            print(f"\n[Source {i+1}]: {doc.metadata['source']}")
            print(doc.page_content[:300].replace("\n", " ") + "...")

if __name__ == "__main__":
    main()
