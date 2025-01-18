"""
Automates buying a product on Amazon using the agent.

Note: Ensure you have added your OPENAI_API_KEY to your environment variables.
Also, make sure to close your Chrome browser before running this script so it can open in debug mode.
"""

import os
import sys
import asyncio
import logging

from pathlib import Path
import uuid

from browser_use.agent.views import ActionResult
from browser_use.browser.context import BrowserContext

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI

from browser_use import Agent, Controller
from browser_use.browser.browser import Browser, BrowserConfig
from openai import OpenAI

from asteroid_sdk.registration.initialise_project import asteroid_init, register_tool_with_supervisors
from asteroid_sdk.supervision.base_supervisors import human_supervisor
from asteroid_sdk.wrappers.openai import asteroid_openai_client

# from browser_use.supervisors import agent_output_supervisor

# Initialize logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize the project
run_id = asteroid_init(project_name="Browser Use", task_name="Planning & Zoning")

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
    supervision_functions=[],
    # supervision_functions=[[human_supervisor()]],
    run_id=run_id,
)

# Initialize the browser
browser = Browser(
    config=BrowserConfig(
        headless=False,
        # WSL 2 Chrome
        # chrome_instance_path='/mnt/c/Program Files/Google/Chrome/Application/chrome.exe',
        # wss_url='ws://host.docker.internal:9222'
    )
)
controller = Controller()


@controller.action(
    'Write important output information to a file',
    requires_browser=False,
)
async def write_to_file(content: str):
    with open(f'output_{run_id}.txt', 'w') as f:
        f.write(content)
    return ActionResult(extracted_content='Output written to file')

@controller.action(
    'Get text from element - retrieves the text content from a DOM element at the specified index',
    requires_browser=True,
)
async def get_text(index: int, browser: BrowserContext):
    dom_el = await browser.get_dom_element_by_index(index)

    if dom_el is None:
        return ActionResult(error=f'No element found at index {index}')

    try:
        text_content = dom_el.get_all_text_till_next_clickable_element()
        msg = f'Successfully retrieved text from element at index {index}'
        logger.info(f'{msg}: {text_content}')
        return ActionResult(extracted_content=text_content)
    except Exception as e:
        logger.debug(f'Error getting text content: {str(e)}')
        return ActionResult(error=f'Failed to get text from element at index {index}')

@controller.action(
    'Get human supervisor help',
    requires_browser=False,
)
async def get_human_supervisor_help():
    # Get help via the CLI
    print("Getting help via the CLI")
    user_input = input("Please give me some help")
    return ActionResult(extracted_content=user_input)

@controller.action(
    'Screenshot the current page', requires_browser=True
)
async def screenshot(browser: BrowserContext):
    path = f'screenshot_{uuid.uuid4()}.png'
    page = await browser.get_current_page()
    await browser.remove_highlights()
    await page.screenshot(path=path)

    msg = '📸  Screenshot taken'
    logger.info(msg)
    return ActionResult(extracted_content=msg, include_in_memory=True)


# Initialize the LLM
llm = ChatOpenAI(
    model='gpt-4o',
    client=wrapped_openai_client.chat.completions,
    root_client=wrapped_openai_client
    )

# Define the agent's task

prompt = """
Address: 6402 Marigold St 
Coordinates:  32.467141, -99.810052
Base Webpage: https://abilene.maps.arcgis.com/apps/webappviewer/index.html?id=fc76608ae8394a5cb2b4d8a245262275 


Planning & Zoning Task

You are a planning and zoning agent. You are tasked with finding information about a parcel of land from a local government website. With the address and coordinates given, you need to find the parcel and then find the zoning information for that parcel. You should return all of the zoning information for the parcel, and take a screenshot of the full parcel. You can use the get_text function to get text directly from the DOM, which is usually more efficient that using your vision. 

Here are some specific instructions that are relevant to this particular government website:
- Click on Planning & Zoning application
- Click in the search bar
- Search using the address given 
- Will present with two options, address points or parcels. Click on parcels
- Scrape all the data from the pop up window
- Take a screenshot of the full parcel

Here is some advice that you have acquired over time when working with this website:
- When taking a screenshot, make sure that you don't have any modals open that cover the parcel of land. You must be able to see the full parcel. Before taking a screenshot, as yourself, are you sure that you have closed all modals?

If you are struggling with a task, or seem to be stuck in a loop (defined as repeating the same action to no avail 3 times), you can reach out to a human supervisor that will help you with your task.
Your final output should be written to file using the write_to_file function.
"""

# prompt2 = """

# Task 2 - Parcel Zoning Layers 

# Address: 6402 Marigold St 
# Coordinates:  32.467141, -99.810052
# Base Webpage: https://abilene.maps.arcgis.com/apps/webappviewer/index.html?id=fc76608ae8394a5cb2b4d8a245262275

# Navigate to the base webpage, load the address and take a screenshot of the map
# Make sure that the map is zoomed out to the full extent of the parcel
# Make sure that you view the map in parcel mode and that the parcel is visible
# """

# Click on Planning & Zoning application
# Click in the search bar
# Search using the address given 
# Will present with two options, address points or parcels. Click on parcels
# Now click on the Layer List on the left side bar 
# Zoom out so the full parcel boundaries are visible 
# Click on the layer options Icon 
# Turn all layers off 
# Iterate through the layers
# - Press the drop down to expand the legend 
# - Turn on the layer 
# - Take a screenshot of both the map and the legend and feed into vision model to analyse 
# - Click on the parcel (or centre of the screen) 
#     - For some layers it may show a pop up window, scrape the data from this
# """

agent = Agent(
    task=prompt,
    llm=llm,
    controller=controller,
    browser=browser,
)

async def main():
    await agent.run(max_steps=30)
    await browser.close()
    agent.create_history_gif()


if __name__ == '__main__':
    asyncio.run(main())

