"""
Agent to navigate Calendly booking and collect required information.

@dev Ensure OPENAI_API_KEY is set in your environment variables.
"""

import os
import sys

os.environ['ANONYMIZED_TELEMETRY'] = 'false'

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from browser_use import Agent, Controller, Browser
import logging


logging.basicConfig(level=logging.DEBUG)


load_dotenv()

controller = Controller()

# Define a custom action to ask the user for information
@controller.action('Ask user for information')
def ask_user(question: str) -> str:
    user_input = input(f"{question}\nYour input: ")
    return user_input

async def get_calendly_options(calendly_link: str, model: ChatOpenAI, controller: Controller, browser: Browser) -> str:
    task = (
        f"Go to the Calendly link {calendly_link}. "
        "Select the first available time slot, click next, find out what details are required to schedule the meeting, and report back the required details."
    )
    
    get_input_details_agent = Agent(task=task, llm=model, controller=controller, browser=browser)
    
    history = await get_input_details_agent.run()
    
    get_input_details_agent.create_history_gif(output_path='get_calendly_options.gif', show_logo=False, show_goals=False, show_task=False, font_size=20)
    
    necessary_details = history.history[-1].result[-1].extracted_content
    return necessary_details


async def book_calendly_meeting(calendly_link: str, date: str, user_input: str, model: ChatOpenAI, controller: Controller, browser: Browser):
    follow_up_task = (
        f"Go to the Calendly link {calendly_link}. "
        f"Book a calendly meeting for {date} using these details: {user_input}"
    )
    
    follow_up_task_agent = Agent(task=follow_up_task, llm=model, controller=controller, browser=browser)
    follow_up_history = await follow_up_task_agent.run()
    
    follow_up_task_agent.create_history_gif(output_path='book_calendly_meeting.gif', show_logo=False, show_goals=False, show_task=False, font_size=20)
    
    return follow_up_history


async def main():
    calendly_link = "https://calendly.com/founders-asteroid-hhaf/30min?month=2025-01"
    date = "20.1.2025 10:00 PST"
    
    model = ChatOpenAI(model='gpt-4o-mini', temperature=0)
    browser = Browser()
    
    necessary_details = await get_calendly_options(calendly_link, model, controller, browser)
    user_input = ask_user(f"Please provide the following details: {necessary_details}")
    
    follow_up_history = await book_calendly_meeting(calendly_link, date, user_input, model, controller, browser)
    
    print('--------------------------------')
    print('Agent finished with the following history:')
    print(follow_up_history)


if __name__ == '__main__':
    asyncio.run(main())

