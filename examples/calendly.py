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

from openai import OpenAI

from asteroid_sdk.registration.initialise_project import asteroid_init, register_tool_with_supervisors
from asteroid_sdk.supervision.base_supervisors import human_supervisor
from asteroid_sdk.wrappers.openai import asteroid_openai_client

from browser_use.supervisors import agent_output_supervisor

# Initialize logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize the project
run_id = asteroid_init(project_name="Browser Use", task_name="Calendly")

# Initialize the OpenAI client
openai_client = OpenAI()
wrapped_openai_client = asteroid_openai_client(
    openai_client, run_id
)



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
    
    
    run_id = asteroid_init(project_name="Calendly", task_name="Book a meeting")

    # Initialize the OpenAI client
    openai_client = OpenAI()
    wrapped_openai_client = asteroid_openai_client(
        openai_client, run_id
    )

    tool = {
        'name': 'AgentOutput',
        'description': """
    Agent output is a 'Master Tool' with access to manny other tools. The tools that it has access too are as follows:
    Functions:
    Search Google in the current tab: 
        {search_google: {'query': {'type': 'string'}}}
    Navigate to URL in the current tab: 
        {go_to_url: {'url': {'type': 'string'}}}
    Go back: 
        {go_back: {}}
    Click element: 
        {click_element: {'index': {'type': 'integer'}, 'xpath': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'default': None}}}
    Input text into a input interactive element: 
        {input_text: {'index': {'type': 'integer'}, 'text': {'type': 'string'}, 'xpath': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'default': None}}}
    Switch tab: 
        {switch_tab: {'page_id': {'type': 'integer'}}}
    Open url in new tab: 
        {open_tab: {'url': {'type': 'string'}}}
    Extract page content to get the text or markdown : 
        {extract_content: {'value': {'default': 'text', 'enum': ['text', 'markdown', 'html'], 'type': 'string'}}}
    Complete task: 
        {done: {'text': {'type': 'string'}}}
    Scroll down the page by pixel amount - if no amount is specified, scroll down one page: 
        {scroll_down: {'amount': {'anyOf': [{'type': 'integer'}, {'type': 'null'}], 'default': None}}}
    Scroll up the page by pixel amount - if no amount is specified, scroll up one page: 
        {scroll_up: {'amount': {'anyOf': [{'type': 'integer'}, {'type': 'null'}], 'default': None}}}
    Send strings of special keys like Backspace, Insert, PageDown, Delete, Enter, Shortcuts such as `Control+o`, `Control+Shift+T` are supported as well. This gets used in keyboard.press. Be aware of different operating systems and their shortcuts: 
        {send_keys: {'keys': {'type': 'string'}}}
    If you dont find something which you want to interact with, scroll to it: 
        {scroll_to_text: {'text': {'type': 'string'}}}
    Get all options from a native dropdown: 
        {get_dropdown_options: {'index': {'type': 'integer'}}}
    Select dropdown option for interactive element index by the text of the option you want to select: 
        {select_dropdown_option: {'index': {'type': 'integer'}, 'text': {'type': 'string'}}}
        """,
        'attributes': {
            'Any': 'Any'
        }
    }
    
    register_tool_with_supervisors(
    tool=tool,
    supervision_functions=[[agent_output_supervisor, human_supervisor()]],
    run_id=run_id,
)
    
    model = ChatOpenAI(model='gpt-4o',
            client=wrapped_openai_client.chat.completions,
            root_client=wrapped_openai_client)
    
    browser = Browser()
    
    # necessary_details = await get_calendly_options(calendly_link, model, controller, browser)
    # user_input = ask_user(f"Please provide the following details: {necessary_details}")
    user_input = "David, david@asteroid.com, 1234567890"
    
    follow_up_history = await book_calendly_meeting(calendly_link, date, user_input, model, controller, browser)
    
    print('--------------------------------')
    print('Agent finished with the following history:')
    print(follow_up_history)


if __name__ == '__main__':
    asyncio.run(main())

