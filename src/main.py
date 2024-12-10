#!/usr/bin/env python
# src/my_project/main.py
from crew import LatestAiDevelopmentCrew


def run():
    """
    Run the crew.
    """
    inputs = {
        'topic': 'AI Agents'
    }
    LatestAiDevelopmentCrew().crew().kickoff(inputs=inputs)

if __name__ == "__main__":
    run()

