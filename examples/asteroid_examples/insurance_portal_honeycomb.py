"""
Automates filling out a form on Honeycomb Insurance using the agent.

Note: Ensure you have added your OPENAI_API_KEY to your environment variables.
"""

import os
import sys
import asyncio
import logging
import datetime
import time


from langchain_openai import ChatOpenAI
from browser_use import Agent, Controller
from openai import OpenAI

from asteroid_sdk.registration.initialise_project import asteroid_init, register_tool_with_supervisors
from asteroid_sdk.supervision.base_supervisors import human_supervisor, llm_supervisor, auto_approve_supervisor
from asteroid_sdk.wrappers.openai import asteroid_openai_client
# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from browser_use.asteroid_browser_use.utils import init_browser, do_update_run_metadata
from browser_use.asteroid_browser_use.actions import browser_use_tool
from browser_use.asteroid_browser_use.computer_use import register_computer_use_action
from browser_use.asteroid_browser_use.evaluation import finalize_task

# Initialize logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize the project
TASK_NAME = "Form Filling"
run_id = asteroid_init(project_name="Insurance Portal", task_name=TASK_NAME)

# Initialize the OpenAI client
openai_client = OpenAI()
wrapped_openai_client = asteroid_openai_client(
    openai_client, run_id
)

register_tool_with_supervisors(
    tool=browser_use_tool,
    supervision_functions=[
        [
            auto_approve_supervisor
            # llm_supervisor(instructions="Escalate to a human supervisor when the agent gets to the final form on the page where it says: Contact Details & Desired Insurance Coverage Limits. Don't focus on anything else! Approve all other actions."),
            # human_supervisor()
        ]
    ],
    run_id=run_id,
)
# Initialize the LLM
llm = ChatOpenAI(
    model='gpt-4o',
    client=wrapped_openai_client.chat.completions,
    root_client=wrapped_openai_client
)

# url = "https://honeycombinsurance.com/"

url = "https://app.honeycombinsurance.com/quote/start?d=eyJhZGRyZXNzIjp7ImFkZHJlc3NDb21wb25lbnQiOnsic3RyZWV0TnVtYmVyIjoiMjIzOCIsInN0cmVldE5hbWUiOiJHZWFyeSBCb3VsZXZhcmQiLCJjaXR5IjoiU2FuIEZyYW5jaXNjbyIsImNvdW50eSI6IlNhbiBGcmFuY2lzY28gQ291bnR5Iiwic3RhdGVBYnIiOiJDQSIsInN0YXRlIjoiQ2FsaWZvcm5pYSIsInppcGNvZGUiOiI5NDExNSIsInppcCI6Ijk0MTE1In0sImZvcm1hdHRlZEFkZHJlc3MiOiIyMjM4IEdlYXJ5IEJsdmQsIFNhbiBGcmFuY2lzY28sIENBIDk0MTE1LCBVU0EiLCJsYXRpdHVkZSI6MzcuNzgzNTU5NCwibG9uZ2l0dWRlIjotMTIyLjQ0MDY3MzZ9LCJ1dG0iOnt9fQ==)"

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

3. **Building Systems:** - You should use computer_use to fill in the building systems.
   - Roof: Select "Past 5 years".
   - Plumbing: Select "5-15 years ago".
   - Electrical: Select "Over 30 years ago".
   - HVAC: Select "Past 5 years".
   - Water Heater: Select "5-15 years ago". - You needto scroll to see this one and then click next.

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

5. When you get to the final page saying: "Contact Details & Desired Insurance Coverage Limits", fill out the page, effective date is: 02.02. - Make sure to use this format, only input the date, the year is already filled in! Fill out all additional fields and get quotes! Do not finish until you have quotes!

6. At the last page, answer everything No, then select that you have active policy and submit.

**Additional Notes**:

- Use the `input_text` and `select_dropdown_option` functions to fill out text fields and select options.
- Ensure all mandatory fields are filled.
- Use the `extract_content` function if you need to verify any information on the page.

Your final output should be concise and include any relevant information collected during the process.
"""


async def main():
    browser, debug_url, session_id = init_browser()
    
    controller = Controller()

    # Register computer_use action from the computer_use module
    register_computer_use_action(controller)
    
    
    agent = Agent(
        task=prompt,
        llm=llm,
        controller=controller,
        browser=browser,
)

    # Important, this is needed so the computer use action can access the message manager
    browser.message_manager = agent.message_manager
    browser.llm = llm
    
    folder_name = "agent_executions/insurance_portal_honeycomb"
    
    await agent.run(max_steps=60)
    await finalize_task(agent, TASK_NAME, str(run_id), folder_name, evaluate=False, output_format="gif")
    await browser.close()

if __name__ == '__main__':
    asyncio.run(main())
