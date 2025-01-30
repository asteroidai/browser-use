"""
Automates buying a product on Amazon using the agent.

Note: Ensure you have added your OPENAI_API_KEY to your environment variables.
Also, make sure to close your Chrome browser before running this script so it can open in debug mode.
"""

import os
import sys
import asyncio
import logging
import datetime

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



TASK_NAME = "parcel_zoning_info"
# TASK_NAME = "parcel_zoning_layers"

# Initialize the project
run_id = asteroid_init(project_name="Data Centers", task_name=TASK_NAME)

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

time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
folder_name = f"agent_executions/{TASK_NAME}/recording_{TASK_NAME.replace(' ', '_')}_{time_str}_{run_id}"
os.makedirs(folder_name, exist_ok=True)


# Initialize the browser
browser = Browser(
    config=BrowserConfig(
        headless=False,
        new_context_config=BrowserContextConfig(
            apply_click_styling=True,
            apply_form_related=True,
            browser_window_size={"width": 1024, "height": 768}, # These are values for Anthropic computer use
            save_recording_path=folder_name
        ),
    )
)
controller = Controller()

# Register computer_use action from the computer_use module
register_computer_use_action(controller)

# Register Asteroid actions
register_asteroid_actions(controller, str(run_id), folder_name)

# Initialize the LLM
llm = ChatOpenAI(
    model="gpt-4o",
    client=wrapped_openai_client.chat.completions,
    root_client=wrapped_openai_client
)


# You can use the get_text function to get text directly from the DOM, which is usually more efficient that using your vision. 
# Define the agent's task
prompt_parcel_zoning_info = """
Planning & Zoning Task

You are a planning and zoning agent. You are tasked with finding information about a parcel of land from a local government website. With the address and coordinates given, you need to find the parcel and then find the zoning information for that parcel. You should return all of the zoning information for the parcel, and take a screenshot of the full parcel. 

Here are some specific instructions that are relevant to this particular government website:
- Click on Planning & Zoning application
- Click in the search bar
- Search using the address given 
- Will present with two options, address points or parcels. Click on parcels.
- Maximize the pop-up window by clicking the maximize button on the top right corner. Use the computer use action to achieve this.
- After maximizing the pop-up window, extract all the data from it. This may require multiple steps:
    - Take a screenshot of the pop-up window.
    - Use the write_to_file function append the text parcel data from the pop-up window to a file.
    - Scroll down to capture all the data. Take additional screenshots and save the new data to the file until all data is captured.
- Close the pop-up window by clicking the close button on the top right corner. Use the computer use action to achieve this.
- Zoom out to view the entire parcel by sending the keys `Control` + `-`.
- Take a screenshot of the full parcel.


Here is some advice that you have acquired over time when working with this website:
- When taking a screenshot, make sure that you don't have any modals open that cover the parcel of land. You must be able to see the full parcel. Before taking a screenshot, ask yourself, are you sure that you have closed all modals?

If you are struggling with a task, or seem to be stuck in a loop (defined as repeating the same action to no avail 3 times), you can request human supervisor to help you with your task.
Your final output should be written to file using the write_to_file function.

Information about the task:

Base Webpage: https://abilene.maps.arcgis.com/apps/webappviewer/index.html?id=fc76608ae8394a5cb2b4d8a245262275 
Address: 6402 Marigold St 
Coordinates:  32.467141, -99.810052
"""


prompt_parcel_zoning_layers = """
Parcel Zoning Layers

You are a planning and zoning agent. You are tasked with finding information about a parcel of land from a local government website. With the address and coordinates given, you need to find the parcel and then find the zoning information for that parcel. You should return all of the zoning information for the parcel, and take a screenshot of the full parcel. You can use the get_text function to get text directly from the DOM, which is usually more efficient that using your vision. 

Here are some specific instructions that are relevant to this particular government website:
- Click on Planning & Zoning application
- Click in the search bar
- Search using the address given 
- Will present with two options, address points or parcels. Click on parcels.
- Now click on the Layer List on the left side bar, it's the second icon from the top.
- Zoom out so the full parcel boundaries are visible. You can do this by sending the keys `Control` + `-` to zoom out.
- Click on the layer options Icon. It's next to the search button on the top of the layers menu.
- Turn all layers off
- Now iterate through all of the layers! For each layer, do the following:
    - Press the drop down to expand the legend
    - Turn on the layer
    - Take a screenshot of both the map and the legend. Make sure that the layer has loaded before taking the screenshot.
    - Click on the parcel (or centre of the screen)
    - For some layers it may show a pop up window, read the data from this and write it to a file using the write_to_file function.

Here is some advice that you have acquired over time when working with this website:
- When taking a screenshot, make sure that you don't have any modals open that cover the parcel of land. You must be able to see the full parcel. Before taking a screenshot, ask yourself, are you sure that you have closed all modals?

If you are struggling with a task, or seem to be stuck in a loop (defined as repeating the same action to no avail 3 times), you can request human supervisor to help you with your task.
Your final output should be written to file using the write_to_file function.

Information about the task:

Base Webpage: https://abilene.maps.arcgis.com/apps/webappviewer/index.html?id=fc76608ae8394a5cb2b4d8a245262275 
Address: 6402 Marigold St 
Coordinates:  32.467141, -99.810052
"""


agent = Agent(
    task=prompt_parcel_zoning_info,
    llm=llm,
    controller=controller,
    browser=browser,
)

# Important, this is needed so the computer use action can access the message manager
browser.message_manager = agent.message_manager
browser.llm = llm

async def main():
    await agent.run(max_steps=30)
    await finalize_task(agent, TASK_NAME, str(run_id), folder_name)
    await browser.close()

if __name__ == '__main__':
    asyncio.run(main())

