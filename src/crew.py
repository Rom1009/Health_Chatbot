from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv
import os
load_dotenv()

@CrewBase
class HealthChatbot():
	"""Health_chatbot crew"""
	agents_config = "config/agents.yaml"
	tasks_config = "config/tasks.yaml"
	
	def __init__(self) -> None:
		self.groq_llm = LLM(model = "groq/gemma-7b-it", api_key=os.getenv("GROQ_API_KEY"), max_tokens=100)
		
	@agent
	def healthier_advice(self) -> Agent:
		return Agent(
			config=self.agents_config['healthier_advice'],
			verbose=True,
			llm = self.groq_llm,
		)

	@task
	def research_task(self) -> Task:
		return Task(
			config=self.tasks_config['research_task'],
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