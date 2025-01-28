"""
Automates buying a product on Amazon using the agent.

Note: Ensure you have added your OPENAI_API_KEY to your environment variables.
Also, make sure to close your Chrome browser before running this script so it can open in debug mode.
"""

import os
import sys
import asyncio
import logging

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
from browser_use import Agent, Controller
from browser_use.browser.browser import Browser, BrowserConfig, BrowserContextConfig
from openai import OpenAI

from asteroid_sdk.registration.initialise_project import asteroid_init, register_tool_with_supervisors
from asteroid_sdk.supervision.base_supervisors import human_supervisor, auto_approve_supervisor
from asteroid_sdk.wrappers.openai import asteroid_openai_client

# Import evaluation and computer_use modules
from browser_use.asteroid_browser_use.evaluation import finalize_task
from browser_use.asteroid_browser_use.computer_use import register_computer_use_action
from browser_use.asteroid_browser_use.actions import register_asteroid_actions

# Initialize logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)



TASK_NAME = "Insurance Car Quote"

# Initialize the project
run_id = asteroid_init(project_name="Insurance Car Quote", task_name=TASK_NAME)

# Initialize the OpenAI client
openai_client = OpenAI()
wrapped_openai_client = asteroid_openai_client(
    openai_client, run_id
)

# Register the tool with supervisors
tool = {
    'name': 'AgentOutput',
    'description': """
Agent output is a 'Master Tool' with access to many other tools. The tools that it has access to are as follows:
Functions:
Search Google in the current tab:
    {search_google: {'query': {'type': 'string'}}}
Navigate to URL in the current tab:
    {go_to_url: {'url': {'type': 'string'}}}
Go back:
    {go_back: {}}
Click element:
    {click_element: {'index': {'type': 'integer'}, 'xpath': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'default': None}}}
Input text into an interactive element:
    {input_text: {'index': {'type': 'integer'}, 'text': {'type': 'string'}, 'xpath': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'default': None}}}
Switch tab:
    {switch_tab: {'page_id': {'type': 'integer'}}}
Open URL in new tab:
    {open_tab: {'url': {'type': 'string'}}}
Extract page content to get the text or markdown:
    {extract_content: {'value': {'default': 'text', 'enum': ['text', 'markdown', 'html'], 'type': 'string'}}}
Complete task:
    {done: {'text': {'type': 'string'}}}
Scroll down the page by pixel amount - if no amount is specified, scroll down one page:
    {scroll_down: {'amount': {'anyOf': [{'type': 'integer'}, {'type': 'null'}], 'default': None}}}
Scroll up the page by pixel amount - if no amount is specified, scroll up one page:
    {scroll_up: {'amount': {'anyOf': [{'type': 'integer'}, {'type': 'null'}], 'default': None}}}
Send strings of special keys like Backspace, Insert, PageDown, Delete, Enter, Shortcuts such as `Control+o`, `Control+Shift+T` are supported as well. This gets used in keyboard.press. Be aware of different operating systems and their shortcuts:
    {send_keys: {'keys': {'type': 'string'}}}
If you don't find something which you want to interact with, scroll to it:
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
    supervision_functions=[[auto_approve_supervisor]],
    run_id=run_id,
)

# Initialize the browser
browser = Browser(
    config=BrowserConfig(
        headless=False,
        new_context_config=BrowserContextConfig(
            apply_click_styling=True,
            apply_form_related=True,
            browser_window_size={"width": 1024, "height": 768}, # These are values for Anthropic computer use
            save_recording_path='agent_executions/'
        ),
    )
)
controller = Controller()

# Register computer_use action from the computer_use module
register_computer_use_action(controller)

# Register Asteroid actions
register_asteroid_actions(controller, str(run_id))

# Initialize the LLM
llm = ChatOpenAI(
    model="gpt-4o",
    client=wrapped_openai_client.chat.completions,
    root_client=wrapped_openai_client
)

INSURANCE_WEBSITES = [
    "https://insurify.com/",
    "https://www.compare.com/",
    "https://app.coverage.com/#/flow/drivers/1/currently-insured"
    # "https://www.policygenius.com/",
    # "https://www.thezebra.com/",
    # "https://www.trustedchoice.com/car-insurance/",   
]

# Define the agent's task
prompt_template = """
Visit the car insurance comparison website: {website}

Navigate to the homepage or the car insurance quote section.

Fill out the required form fields with the following test data:

Vehicle: Toyota Camry, 2020, VIN optional (if asked)
Driver: 30 years old, no prior accidents, single, good credit score
Location: San Francisco, CA, ZIP code 94103
Coverage: Standard liability coverage with $500 deductible

Submit the form and wait for the quotes page to load.

Scrape the following details from the resulting quotes page:

Insurance provider name
Monthly or annual premium cost
Coverage details (if available)
Additional fees or special offers
Save the data in the following format to a file in this or similar format:

"website": "{website}",
"provider": "Geico",
"monthly_premium": "$78",
"annual_premium": "$936",
"coverage": "State minimum",
"additional_info": "Roadside assistance included"
"""

PARALLELISED = True

async def run_agent_for_website(website):
    prompt = prompt_template.format(website=website)
    agent = Agent(
        task=prompt,
        llm=llm,
        controller=controller,
        browser=browser,
    )
    browser.message_manager = agent.message_manager
    await agent.run(max_steps=120)
    await finalize_task(agent, TASK_NAME, str(run_id))

async def main():
    if PARALLELISED:
        await asyncio.gather(*[run_agent_for_website(website) for website in INSURANCE_WEBSITES])
    else:
        for website in INSURANCE_WEBSITES:
            await run_agent_for_website(website)
    # TODO: Merge all the results together
    
    
    await browser.close()

if __name__ == '__main__':
    asyncio.run(main())

