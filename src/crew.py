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
		self.groq_llm = LLM(model = "groq/gemma-7b-it", api_key=os.getenv("GROQ_API_KEY"), max_tokens=1000)
		self.pdf_tools = PDFSearchTool(
			pdf = "../public/dataset/9789289057622-eng.pdf",

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
		
	@agent
	def healthier_advice(self) -> Agent:
		return Agent(
			config=self.agents_config['healthier_advice'],
			verbose=True,
			llm = self.groq_llm,
			tools = [self.pdf_tools]
		)
	
	@agent
	def symptom_specialist(self) -> Agent:
		return Agent(
			config=self.agents_config['symptom_specialist'],
			verbose=True,
			llm = self.groq_llm,
			tools = [self.pdf_tools]
		)
	
	@agent
	def lifestyle_health_coach(self) -> Agent:
		return Agent(
			config=self.agents_config['lifestyle_health_coach'],
			verbose=True,
			llm = self.groq_llm,
			tools = [self.pdf_tools]
		)

	@task
	def research_task(self) -> Task:
		return Task(
			config=self.tasks_config['health_advisor_task'],
		)
	
	@task
	def research_task(self) -> Task:
		return Task(
			config=self.tasks_config['symptom_specialist_task'],
		)
	
	@task
	def research_task(self) -> Task:
		return Task(
			config=self.tasks_config['lifestyle_health_coach_task'],
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