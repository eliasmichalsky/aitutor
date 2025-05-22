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
import re
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship

load_dotenv()

os.environ["NEO4J_URI"] = "***"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "***"

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

def clean_text_content(text):
    """Clean text content to remove technical metadata and irrelevant information."""
    # Remove common PDF metadata patterns
    patterns_to_remove = [
        r'Microsoft\s+(Word|Office|365).*',
        r'Adobe\s+Acrobat.*',
        r'PDF\s+Producer.*',
        r'Creator:\s*.*',
        r'Producer:\s*.*',
        r'CreationDate:\s*.*',
        r'ModDate:\s*.*',
        r'www\.\w+\.\w+',
        r'http[s]?://[^\s]+',
        r'\b\d{4}-\d{2}-\d{2}\b',  # Remove standalone dates
        r'\b\d{2}:\d{2}:\d{2}\b',  # Remove standalone times
        r'Page\s+\d+\s+of\s+\d+',
        r'Sida\s+\d+\s+av\s+\d+',
    ]
    
    cleaned_text = text
    for pattern in patterns_to_remove:
        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)
    
    # Remove extra whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text

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
                # Clean the content
                doc.page_content = clean_text_content(doc.page_content)
                
                # Only keep essential metadata
                doc.metadata = {
                    'source_file': os.path.basename(pdf_file),
                    'domain': 'dementia_care'
                }
            
            documents.extend(pdf_docs)
            print(f"Successfully loaded {len(pdf_docs)} pages from {os.path.basename(pdf_file)}")
            
        except Exception as e:
            print(f"Error loading {pdf_file}: {str(e)}")
    
    return documents

def process_documents(documents):
    """Split documents into smaller chunks for better processing."""
    all_splits = []
    
    for doc in documents:
        # Skip very short documents that are likely metadata
        if len(doc.page_content.strip()) < 100:
            continue
            
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
                # Enhanced instruction to focus on meaningful content
                swedish_content = f"""
Extrahera entiteter och relationer från denna svenska text om demensvård. 
Fokusera på medicinska termer, behandlingar, symtom, organisationer och personer - inte teknisk metadata eller programvarunamn.
Ignorera referenser till Microsoft, Adobe, PDF, Word, eller andra tekniska detaljer.

Text: {doc.page_content}
"""
                swedish_doc = Document(
                    page_content=swedish_content,
                    metadata=doc.metadata
                )
                swedish_docs.append(swedish_doc)
            
            batch_graph_docs = await llm_transformer.aconvert_to_graph_documents(swedish_docs)
            
            if batch_graph_docs:
                # Filter out irrelevant nodes
                filtered_docs = []
                for graph_doc in batch_graph_docs:
                    filtered_nodes = []
                    filtered_relationships = []
                    
                    # Filter nodes to exclude technical metadata
                    for node in graph_doc.nodes:
                        node_id_lower = node.id.lower()
                        if not any(term in node_id_lower for term in [
                            'microsoft', 'office', 'word', 'adobe', 'pdf', 'acrobat',
                            'www.', 'http', '.com', '.se', '.org', 'creator', 'producer',
                            '365', 'windows'
                        ]):
                            filtered_nodes.append(node)
                    
                    filtered_node_ids = {node.id for node in filtered_nodes}
                    for rel in graph_doc.relationships:
                        if (rel.source.id in filtered_node_ids and 
                            rel.target.id in filtered_node_ids):
                            filtered_relationships.append(rel)
                    
                    if filtered_nodes:
                        graph_doc.nodes = filtered_nodes
                        graph_doc.relationships = filtered_relationships
                        filtered_docs.append(graph_doc)
                
                all_graph_documents.extend(filtered_docs)
                
                for j, graph_doc in enumerate(filtered_docs):
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

def check_neo4j_connection():
    try:
        # Create a simple test graph document with one node
        test_node = Node(
            id="connection_test",
            type="TestNode",
            properties={"timestamp": "test_connection", "removable": True}
        )
        
        # Create a simple test document as source
        test_source = Document(
            page_content="Test connection document",
            metadata={"purpose": "connection_test"}
        )
        
        test_graph_doc = GraphDocument(
            nodes=[test_node],
            relationships=[],
            source=test_source
        )
        
        graph.add_graph_documents([test_graph_doc], include_source=False)
        print("Neo4j connection test successful!")
        return True
        
    except Exception as e:
        print(f"Neo4j connection test failed: {str(e)}")
        return False

def save_to_neo4j(graph_documents):
    """
    Save graph documents to Neo4j without including sources.
    
    Args:
        graph_documents: List of graph documents to save
    """
    print(f"\nAdding {len(graph_documents)} graph documents to Neo4j (without sources)...")
    
    try:
        graph.add_graph_documents(graph_documents, include_source=False)
        print("Knowledge graph saved successfully without source metadata!")
        return True
        
    except Exception as e:
        print(f"Could not save to Neo4j: {str(e)}")
        return False

async def main():
    """Main function to orchestrate the knowledge graph creation."""
    
    # Test Neo4j connection first using the exact same save method
    if not check_neo4j_connection():
        print("Exiting: Cannot connect to Neo4j database.")
        return
    
    documents = load_pdf_documents("pdfs")
    
    if not documents:
        print("No PDFs in that directory.")
        return
    
    # Process documents (split into chunks)
    processed_docs = process_documents(documents)
    
    if not processed_docs:
        print("No meaningful content found after processing.")
        return
    
    # Create knowledge graph
    graph_docs = await create_knowledge_graph(processed_docs)
    
    if not graph_docs:
        print("No graph documents created.")
        return
    
    # Show summary of what will be saved
    total_nodes = sum(len(doc.nodes) for doc in graph_docs)
    total_relationships = sum(len(doc.relationships) for doc in graph_docs)
    print(f"\nReady to save: {total_nodes} nodes and {total_relationships} relationships")
    
    save_to_neo4j(graph_docs)
    print("\nProcess completed!")

if __name__ == "__main__":
    asyncio.run(main())