import os
import asyncio
import pandas as pd
import numpy as np
from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex,Response
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.evaluation import DatasetGenerator, RelevancyEvaluator, CorrectnessEvaluator,FaithfulnessEvaluator
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.core.async_utils import asyncio_run

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

def vector_store(nodes, documents, embed_model):
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

def evaluation():
    pass

def display_results(name, eval_results):
    metric_dicts = []

    for eval_result in eval_results:
        metric_dict = eval_result.metric_vals_dict
        metric_dicts.append(metric_dict)

    full_df = pd.DataFrame(metric_dicts)
    
    hit_rate = full_df["hit_rate"].mean()
    mrr = full_df["mrr"].mean()

    metric_df = pd.DataFrame(
        {
            "Retirver Name": [name],
            "Hit Rate": [hit_rate],
            "MRR": [mrr],
        }
    )
    return metric_df

if __name__ == "__main__":
    documents = extract_document("../public/9789241548373-vie.pdf")
    embed_model = vector_embediing()
    # nodes = chunking(embed_model, documents)
    # vector_index = vector_store(nodes, documents, embed_model)
    Settings.llm = Groq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    Settings.embed_model = embed_model

    storage_context = StorageContext.from_defaults(
        persist_dir = "./embedding/Test"
    )

    vector_index = load_index_from_storage(
        storage_context, 
    )
    selected_documents = documents[50:55]

    data_generator = DatasetGenerator.from_documents(selected_documents)
    
    eval_questions = data_generator.generate_questions_from_nodes()
    query_engine = vector_index.as_query_engine()
    async def evaluate_single(question, response_text):
        relevancy_evaluator = RelevancyEvaluator()
        correctness_evaluator = CorrectnessEvaluator()
        faithfulness_evaluator = FaithfulnessEvaluator()

        eval_result_relevancy = await relevancy_evaluator.aevaluate( query=question, response=response_text)
        eval_result_correctness = await correctness_evaluator.aevaluate( query=question, response=response_text)
        eval_result_faithfulness = await faithfulness_evaluator.aevaluate( query=question, response=response_text)
        
        return [eval_result_correctness.score, eval_result_relevancy.score, eval_result_faithfulness.score]
    
    async def main():
        all_results = []
        
        for question in eval_questions:
            query_engine = vector_index.as_query_engine()
            response_vector = query_engine.query(question)
            results = await evaluate_single(question, response_vector)
            all_results.append(results)
        print(all_results)

    asyncio.run(main()) 

    # query = """
    #     Bệnh thấp tim ở chương 4.9
    # """
    # response = query_engine.query(query)
    # print(response.source_nodes[0].get_text())
    # print("---------------------------------------------------")
    # print(response.response)