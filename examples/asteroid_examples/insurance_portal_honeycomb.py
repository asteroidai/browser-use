"""
Automates filling out a form on Honeycomb Insurance using the agent.

Note: Ensure you have added your OPENAI_API_KEY to your environment variables.
"""

import os
import sys
import asyncio
import logging

from pathlib import Path
import uuid

from browser_use.agent.views import ActionResult
from browser_use.asteroid_browser_use.actions import register_asteroid_actions
from browser_use.browser.context import BrowserContext

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
import datetime
from browser_use import Agent, Controller
from browser_use.browser.browser import Browser, BrowserConfig
from openai import OpenAI
import time

from asteroid_sdk.registration.initialise_project import asteroid_init, register_tool_with_supervisors
from asteroid_sdk.supervision.base_supervisors import human_supervisor, llm_supervisor, auto_approve_supervisor
from asteroid_sdk.wrappers.openai import asteroid_openai_client

# Initialize logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize the project
run_id = asteroid_init(project_name="Insurance Portal", task_name="Form Filling")

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
    supervision_functions=[
        [
            # auto_approve_supervisor,
            llm_supervisor(instructions="Escalate to a human supervisor when the agent gets to the final form on the page where it says: Contact Details & Desired Insurance Coverage Limits. Don't focus on anything else! Approve all other actions."),
            human_supervisor()
        ]
    ],
    run_id=run_id,
)

# Initialize the browser
browser = Browser(
    config=BrowserConfig(
        headless=False,
    )
)
controller = Controller()

register_asteroid_actions(controller, str(run_id), folder_name="insurance_portal_honeycomb")

# Initialize the LLM
llm = ChatOpenAI(
    model='gpt-4o',
    client=wrapped_openai_client.chat.completions,
    root_client=wrapped_openai_client
)

url = "https://honeycombinsurance.com/"

# url = "https://app.honeycombinsurance.com/quote/start?d=eyJhZGRyZXNzIjp7ImFkZHJlc3NDb21wb25lbnQiOnsic3RyZWV0TnVtYmVyIjoiMjIzOCIsInN0cmVldE5hbWUiOiJHZWFyeSBCb3VsZXZhcmQiLCJjaXR5IjoiU2FuIEZyYW5jaXNjbyIsImNvdW50eSI6IlNhbiBGcmFuY2lzY28gQ291bnR5Iiwic3RhdGVBYnIiOiJDQSIsInN0YXRlIjoiQ2FsaWZvcm5pYSIsInppcGNvZGUiOiI5NDExNSIsInppcCI6Ijk0MTE1In0sImZvcm1hdHRlZEFkZHJlc3MiOiIyMjM4IEdlYXJ5IEJsdmQsIFNhbiBGcmFuY2lzY28sIENBIDk0MTE1LCBVU0EiLCJsYXRpdHVkZSI6MzcuNzgzNTU5NCwibG9uZ2l0dWRlIjotMTIyLjQ0MDY3MzZ9LCJ1dG0iOnt9fQ==)"

# Define the agent's task
prompt = f"""
Insurance Form Filling Task

You are an insurance agent tasked with filling out an insurance form on the Honeycomb Insurance website for a client. The client details are as follows:

Address: 222 Mason Street, San Francisco, CA, USA

It's a condo association.

Effective Date: 02.02.

1. **Property Details:**
   - Type of Construction: Select "Wood Frame".
   - Detached Garage: Select "No".
   - Year Built: Enter "1998".
   - Total Square Feet: Enter "500".
   - Number of Units: Enter "1".
   - Number of Vacant Units: Enter "0".
   - Number of Units Rented: Enter "0".
   - Number of Buildings: Enter "1".
   - Number of Parking Spaces: Enter "5".
   - Number of Stories: Enter "9".

2. **Facilities & Usage:**
   - Facilities: Select "Swimming Pool" and "Playground".

3. **Building Systems:**
   - Roof: Select "Past 5 years".
   - Plumbing: Select "5-15 years ago".
   - Electrical: Select "Over 30 years ago".
   - HVAC: Select "Past 5 years".
   - Water Heater: Select "5-15 years ago".

4. **Property Condition & Maintenance:**
   - Roof Type: Select "Flat".
   - Building Exterior: Select "Brick or Masonry".
   - Heating, Ventilation & AC Service: Select ">24 months".
   - Plumbing Service: Select "Smart".
   - Parking and Walkways: Select "Fair".
   - Maintenance: Select "24/7".

5. **Contact Details & Desired Insurance Coverage Limits:**
   - Named Insured: Enter "John Doe".
   - Building Limit: Enter "$1,340,000".
   - Business Personal Property Limit: Enter "$20,000".
   - Effective Date: Enter "2023-12-01".
   - Email: Enter "johndoe@example.com".
   - Phone: Enter "555-123-4567".
   - First Name: Enter "John".
   - Last Name: Enter "Doe".
   - Current Carrier: Enter "Type Carrier".

**Instructions**:

1. Navigate to [Honeycomb Insurance Website] {url}


3. Select the correct property type (e.g., Condo Association).

4. Fill in the property details as prompted on each page, then click "Next" until you reach the final page that you also fill in.

5. When you get to the final page saying: "Contact Details & Desired Insurance Coverage Limits", fill out the page, effective date is: 02.02. - Make sure to use this format, only input the date, the year is already filled in!!! Then finish execution.


**Additional Notes**:

- Use the `input_text` and `select_dropdown_option` functions to fill out text fields and select options.
- Ensure all mandatory fields are filled.
- Use the `extract_content` function if you need to verify any information on the page.

Your final output should be concise and include any relevant information collected during the process.
"""

agent = Agent(
    task=prompt,
    llm=llm,
    controller=controller,
    browser=browser,
)

async def main():
    await agent.run(max_steps=60)
    await browser.close()
    agent.create_history_gif(output_path=f'agent_history_{time.time()}.gif', show_goals=False, show_task=False, show_logo=False)

if __name__ == '__main__':
    asyncio.run(main())
