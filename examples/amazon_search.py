"""
Simple try of the agent.

@dev You need to add OPENAI_API_KEY to your environment variables.
"""
import os
import sys

from asteroid_sdk.registration.initialise_project import asteroid_init, register_tool_with_supervisors
from asteroid_sdk.supervision.base_supervisors import human_supervisor
from asteroid_sdk.wrappers.openai import asteroid_openai_client

from browser_use.supervisors import agent_output_supervisor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio

from langchain_openai import ChatOpenAI

from browser_use import Agent
from openai import OpenAI
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

run_id = asteroid_init(project_name="Browser Use", task_name="Amazon Search")

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

llm = ChatOpenAI(model='gpt-4o',
                     client=wrapped_openai_client.chat.completions,
                     root_client=wrapped_openai_client)

agent = Agent(
	task='Go to amazon.co.uk, search for laptop, sort by best rating, and give me the price of the first result. Then add it to the cart, and try to checkout',
	llm=llm,
)

async def main():
    await agent.run(max_steps=20)
    agent.create_history_gif()


asyncio.run(main())
