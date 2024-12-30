import os
from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core import StorageContext, load_index_from_storage, PromptTemplate

def extract_document(path: str) -> list:
    reader = SimpleDirectoryReader(input_files=[path])
    documents = reader.load_data(num_workers=4, show_progress=True)
    return documents

def vector_embediing():
    embed_model = HuggingFaceEmbedding(model_name= "BAAI/bge-small-en-v1.5")
    # embed_model = SentenceTransformer()
    return embed_model

def chunking(embed_model, documents):
    splitter = SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=95,
        embed_model=embed_model
    )
    nodes = splitter.get_nodes_from_documents(documents)
    return nodes

def vector_store(nodes, documents):
    Settings.embed_model = embed_model
    vector_index = VectorStoreIndex.from_documents(
        documents,
        show_progress=True,
        node_parser=nodes
    )
    vector_index.storage_context.persist(
        persist_dir= "./embedding"
    )
    return vector_index

if __name__ == "__main__":
    documents = extract_document("../public/9789241548373-vie.pdf")
    embed_model = vector_embediing()
    # nodes = chunking(embed_model, documents)
    # vector_index = vector_store(nodes, documents)

    llm = Groq(
        model="llama3-8b-8192",
        api_key=os.environ.get("GROQ_API_KEY")
    )

    Settings.embed_model = embed_model

    storage_context = StorageContext.from_defaults(
        persist_dir = "./embedding"
    )

    index = load_index_from_storage(
        storage_context
    )

    query_engine= index.as_query_engine(
        similarity_top_k = 5,
        llm = llm
    )

    query = """
        Tiền căn bệnh thấp tim
    """
    response = query_engine.query(query).response
    print(response)