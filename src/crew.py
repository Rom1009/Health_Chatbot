from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import PDFSearchTool
from dotenv import load_dotenv
import os
load_dotenv()

@CrewBase
class HealthChatbot():
	"""Health_chatbot crew"""
	agents_config = "config/agents.yaml"
	tasks_config = "config/tasks.yaml"

	def __init__(self) -> None:
		self.groq_llm = LLM(model = "groq/llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"), max_tokens=500)
		self.pdf_tools = PDFSearchTool(
			pdf = "../public/9789289057622-eng.pdf",

			config = dict(
				llm = dict(
					provider = "groq",
					config = dict(
						model = "groq/gemma2-9b-it",
					)
				),
				
				embedder=dict(
					provider = "huggingface",
					config=dict(
						model = "BAAI/bge-small-en-v1.5",
					)	
				),
			),

		)
		
	# @agent
	# def symptom_specialist(self) -> Agent:
	# 	return Agent(
	# 		config=self.agents_config['symptom_specialist'],
	# 		verbose=True,
	# 		llm = self.groq_llm,
	# 		tools = [self.pdf_tools]
	# 	)
	
	# @agent
	# def lifestyle_health_coach(self) -> Agent:
	# 	return Agent(
	# 		config=self.agents_config['lifestyle_health_coach'],
	# 		verbose=True,
	# 		llm = self.groq_llm,
	# 		tools = [self.pdf_tools]
	# 	)
	
	@agent
	def healthier_advice(self) -> Agent:
		return Agent(
			config=self.agents_config['healthier_advice'],
			verbose=True,
			llm = self.groq_llm,
			tools = [self.pdf_tools]
		)

	# @task
	# def symptom_specialist_task(self) -> Task:
	# 	return Task(
	# 		config=self.tasks_config['symptom_specialist_task'],
	# 	)
	
	# @task
	# def lifestyle_health_coach_task(self) -> Task:
	# 	return Task(
	# 		config=self.tasks_config['lifestyle_health_coach_task'],
	# 	)

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