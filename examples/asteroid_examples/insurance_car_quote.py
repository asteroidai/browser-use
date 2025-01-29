"""
Automates buying a product on Amazon using the agent.

Note: Ensure you have added your OPENAI_API_KEY to your environment variables.
Also, make sure to close your Chrome browser before running this script so it can open in debug mode.
"""

import os
import sys
import asyncio
import logging
from typing import Dict, Optional

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asteroid_sdk.api.generated.asteroid_api_client.client import Client
from langchain_openai import ChatOpenAI
from browser_use import Agent, Controller
from browser_use.browser.browser import Browser, BrowserConfig, BrowserContextConfig
from openai import OpenAI

from asteroid_sdk.registration.initialise_project import asteroid_init, register_tool_with_supervisors
from asteroid_sdk.supervision.base_supervisors import human_supervisor, auto_approve_supervisor
from asteroid_sdk.wrappers.openai import asteroid_openai_client
from asteroid_sdk.api.generated.asteroid_api_client.api.run.update_run_metadata import sync_detailed
from asteroid_sdk.api.generated.asteroid_api_client.models.update_run_metadata_body import UpdateRunMetadataBody

# Import evaluation and computer_use modules
from browser_use.asteroid_browser_use.evaluation import finalize_task
from browser_use.asteroid_browser_use.computer_use import register_computer_use_action
from browser_use.asteroid_browser_use.actions import register_asteroid_actions

from browserbase import Browserbase

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('insurance_quote.log')
    ]
)

# Set logging levels for specific packages
logging.getLogger('asteroid_sdk').setLevel(logging.INFO)
logging.getLogger('browser_use').setLevel(logging.DEBUG)
logging.getLogger('browserbase').setLevel(logging.INFO)
logging.getLogger('playwright').setLevel(logging.INFO)

logger = logging.getLogger(__name__)

TASK_NAME = "Insurance Car Quote"

# Initialize the OpenAI client
openai_client = OpenAI()

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

def init_browser(browserbase_api_key: Optional[str] = None, browserbase_project_id: Optional[str] = None) -> tuple[Browser, Optional[str]]:
    """Initialize browser with Browserbase if credentials are provided, otherwise use local browser.
    Returns a tuple of (Browser, debug_url)"""
    cdp_url = None
    headless = True
    debug_url = None

    if browserbase_api_key and browserbase_project_id:
        logger.info("Connecting to Browserbase session...")
        bb = Browserbase(api_key=browserbase_api_key)
        session = bb.sessions.create(project_id=browserbase_project_id)
        debug_urls = bb.sessions.debug(session.id)

        debug_url = debug_urls.debugger_fullscreen_url
        logger.info(f"Browserbase Debug Connection URL: {debug_url}")
        logger.info(f"Browserbase CDP URL: {session.connect_url}")

        cdp_url = session.connect_url
        headless = False
    else:
        logger.info("Using local browser (no Browserbase).")

    browser = Browser(
        config=BrowserConfig(
            headless=headless,
            cdp_url=cdp_url,
            new_context_config=BrowserContextConfig(
                apply_click_styling=True,
                apply_form_related=True,
                browser_window_size={"width": 1024, "height": 768},
                save_recording_path='agent_executions/'
            ),
        )
    )
    
    return browser, debug_url

# Define the agent's task
prompt_template = """Get online car insurance quotes for a 2020 Toyota Camry in San Francisco, CA. The full details are: 
Vehicle: Toyota Camry, 2020, VIN optional (if asked) 
Driver: 30 years old, no prior accidents, single, good credit score 
Location: San Francisco, CA, ZIP code 94103 
Coverage: Standard liability coverage with $500 deductible. 
Name: Peter Phillips. 
Date of birth: 1990-01-01, Male, active driver license, no certificate needed, got license when 18 years old, credit score 680, did Bachelor, did not serve in the military. 
I'm employed, car is in storage, never had. 
Input any other details based on your best judgement. Get me the quotes!

Visit the car insurance comparison website: {website}

Navigate to the homepage or the car insurance quote section.

Fill out the required form fields with the provided test data.

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

Once you have this data, use the done tool to output the data.
"""

PARALLELISED = True

async def do_update_run_metadata(run_id: str, run_metadata: Dict[str, str]) -> None:
    """
    Updates the run metadata with the provided run_metadata
    """
    print(f"Updating run metadata for run_id: {run_id}")
    
    asteroid_client = Client(
        base_url="http://localhost:8080/api/v1",
        headers={"X-Asteroid-Api-Key": os.getenv("ASTEROID_API_KEY")}
    )

    metadata = UpdateRunMetadataBody.from_dict(run_metadata)

    try:
        sync_detailed(
            run_id, 
            client=asteroid_client,
            body=metadata
        )
    except Exception as e:
        logger.error(f"Error updating run metadata: {e}")

    logger.info(f"Run metadata has been updated with: {run_metadata}")

async def run_agent_for_website(website):
    # Replace the browser initialization with the new function
    browser, debug_url = init_browser(
        browserbase_api_key=os.getenv("BROWSERBASE_API_KEY"),
        browserbase_project_id=os.getenv("BROWSERBASE_PROJECT_ID")
    )
    controller = Controller()

    # Register computer_use action from the computer_use module
    register_computer_use_action(controller)

    # Extract website name for the task
    website_name = website.replace("https://", "").replace("www.", "").split(".")[0].capitalize()
    task_name = f"{website_name}"
    
    # Create a new run for each website
    run_id = asteroid_init(project_name="Insurance Car Quote", task_name=task_name)
    print(f"Run ID: {run_id}")

    # Register Asteroid actions
    register_asteroid_actions(controller, str(run_id))
    
    # Initialize OpenAI client for this run
    wrapped_openai_client = asteroid_openai_client(
        openai_client, run_id
    )
    
    # Register tool with supervisors for this run
    register_tool_with_supervisors(
        tool=tool,
        supervision_functions=[[auto_approve_supervisor]],
        run_id=run_id,
    )
    
    # Initialize LLM with the run-specific client
    llm = ChatOpenAI(
        model="gpt-4o",
        client=wrapped_openai_client.chat.completions,
        root_client=wrapped_openai_client
    )
    
    prompt = prompt_template.format(website=website)
    agent = Agent(
        task=prompt,
        llm=llm,
        controller=controller,
        browser=browser,
    )

    run_metadata = {
        "agent_name": "default_web",
        "task_name": task_name,
        "parent_run_id": "",
        "task": prompt
    }

    # Add CDP and debugger URLs to metadata if using Browserbase
    if browser.config.cdp_url:
        run_metadata.update({
            "browserbase_cdp_url": browser.config.cdp_url,
            "browserbase_debugger_url": debug_url
        })

    try:
        await do_update_run_metadata(run_id, run_metadata)
    except Exception as e:
        logger.error(f"Error updating run metadata: {e}")

    browser.message_manager = agent.message_manager
    await agent.run(max_steps=120)
    await finalize_task(agent, task_name, str(run_id))
    await browser.close()

INSURANCE_WEBSITES = [
    "https://insurify.com/",
    "https://insurify.com/",
    "https://insurify.com/",
    "https://www.compare.com/",
    "https://app.coverage.com/#/flow/drivers/1/currently-insured"
    "https://www.policygenius.com/",
    "https://www.thezebra.com/",
    "https://www.trustedchoice.com/car-insurance/",   
    "https://www.gabi.com",
    "https://www.coverage.com",
    "https://quotewizard.com",
    "https://www.valuepenguin.com",
    "https://www.nerdwallet.com",
]

async def main():
    if PARALLELISED:
        tasks = []
        for website in INSURANCE_WEBSITES:
            tasks.append(run_agent_for_website(website))
            # Add 500ms delay between launching parallel tasks
            await asyncio.sleep(1)
        await asyncio.gather(*tasks)
    else:
        for website in INSURANCE_WEBSITES:
            await run_agent_for_website(website)
            # Add 500ms delay between agents
            await asyncio.sleep(1)

if __name__ == '__main__':
    asyncio.run(main())

