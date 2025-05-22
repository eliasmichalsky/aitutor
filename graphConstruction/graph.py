import os
import glob
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import asyncio

load_dotenv()

os.environ["NEO4J_URI"] = "*****"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "*****"

graph = Neo4jGraph(refresh_schema=False)


llm = ChatOpenAI(
    temperature=0, 
    model_name="gpt-4o-mini"
)

llm_transformer = LLMGraphTransformer(
    llm=llm,
    node_properties=True,
    relationship_properties=True
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

def load_pdf_documents(pdf_folder_path="pdfs"):
    """Load all PDF documents from the specified folder."""
    documents = []
    
    pdf_files = glob.glob(os.path.join(pdf_folder_path, "*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_folder_path} folder")
        return documents
    
    print(f"Found {len(pdf_files)} PDF files")
    
    for pdf_file in pdf_files:
        print(f"Loading: {pdf_file}")
        try:
            loader = PyPDFLoader(pdf_file)
            pdf_docs = loader.load()
            
            for doc in pdf_docs:
                doc.metadata['source_file'] = os.path.basename(pdf_file)
                doc.metadata['domain'] = 'dementia_care'
            
            documents.extend(pdf_docs)
            print(f"Successfully loaded {len(pdf_docs)} pages from {os.path.basename(pdf_file)}")
            
        except Exception as e:
            print(f"Error loading {pdf_file}: {str(e)}")
    
    return documents

def process_documents(documents):
    """Split documents into smaller chunks for better processing."""
    all_splits = []
    
    for doc in documents:
        # Split the document into chunks
        splits = text_splitter.split_documents([doc])
        all_splits.extend(splits)
    
    print(f"Total document chunks: {len(all_splits)}")
    return all_splits

async def create_knowledge_graph(documents):
    """Create knowledge graph from documents."""
    
    batch_size = 5 
    all_graph_documents = []
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
        
        try:
            swedish_docs = []
            for doc in batch:
                swedish_content = f"""
Behåll all information på svenska. Extrahera entiteter och relationer från denna text på svenska:

{doc.page_content}
"""
                swedish_doc = Document(
                    page_content=swedish_content,
                    metadata=doc.metadata
                )
                swedish_docs.append(swedish_doc)
            
            batch_graph_docs = await llm_transformer.aconvert_to_graph_documents(swedish_docs)
            
            if batch_graph_docs:
                all_graph_documents.extend(batch_graph_docs)
                
                for j, graph_doc in enumerate(batch_graph_docs):
                    print(f"  Dokument {j+1}: {len(graph_doc.nodes)} noder, {len(graph_doc.relationships)} relationer")
                    
                    if graph_doc.nodes:
                        sample_nodes = [node.id for node in graph_doc.nodes[:3]]
                        print(f"    Example nodes: {sample_nodes}")
            else:
                print("No entities extracted from this batch.")
                
        except Exception as e:
            print(f"  Error when processing batch: {str(e)}")
            continue
    
    return all_graph_documents

def save_to_neo4j(graph_documents):
    """
    Save graph documents to Neo4j with a single attempt.
    
    Args:
        graph_documents: List of graph documents to save
    """
    print(f"\nAdding {len(graph_documents)} graph documents to Neo4j...")
    
    try:
        graph.add_graph_documents(graph_documents, include_source=True)
        print("KG saved successfully!")
        return True
        
    except Exception as e:
        print(f"Could not save KG: {str(e)}")
        
        print("Trying without include_source...")
        try:
            graph.add_graph_documents(graph_documents, include_source=False)
            print("Saved without include_source successfully!")
            return True
        except Exception as final_error:
            print(f"Could not save to Neo4j: {final_error}")
            return False


async def main():
    """Main function to orchestrate the knowledge graph creation."""
    
    documents = load_pdf_documents("pdfs")
    
    if not documents:
        print("No PDFs in that directory.")
        return
    
    # Process documents (split into chunks)
    processed_docs = process_documents(documents)
    
    # Create knowledge graph
    graph_docs = await create_knowledge_graph(processed_docs)
    
    if not graph_docs:
        print("No graph documents created.")
        return
    
    save_to_neo4j(graph_docs)

if __name__ == "__main__":
    asyncio.run(main())