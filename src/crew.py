from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import (
    CrewBase, 
    agent, 
    crew, 
    task
)
from dotenv import load_dotenv
import os
from llama_index.core import (
	load_index_from_storage, 
	StorageContext,
	Settings, 
)

from crewai_tools import LlamaIndexTool
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

load_dotenv()

@CrewBase
class HealthChatbot():
	"""Health_chatbot crew"""
	agents_config = "config/agents.yaml"
	tasks_config = "config/tasks.yaml"

	def __init__(self) -> None:
		self.groq_llm = LLM(model = "groq/llama3-70b-8192", api_key=os.getenv("GROQ_API_KEY"), max_tokens=512)
		# Settings.embed_model = HuggingFaceEmbedding(model_name = "BAAI/bge-small-en-v1.5")
		# storage_context = StorageContext.from_defaults(
        # 	persist_dir = "src/embedding/pdf_2"
    	# )

		# index = load_index_from_storage(
        # 	storage_context
    	# )
		
		# query_engine= index.as_query_engine(
        # 	similarity_top_k = 10,
		# 	llm = Groq(model="llama3-8b-8192", api_key=os.environ.get("GROQ_API_KEY"))
    	# )

		# self.query_tool = LlamaIndexTool.from_query_engine(
		# 	query_engine,
		# )
	
	@agent
	def healthier_advice(self) -> Agent:
		return Agent(
			config=self.agents_config['healthier_advice'],
			verbose=True,
			llm = self.groq_llm,
			# tools = [self.query_tool]
		)

	@task
	def health_advisor_task(self) -> Task:
		return Task(
			config=self.tasks_config['health_advisor_task'],
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the LatestAiDevelopment crew"""
		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
		)