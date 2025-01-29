
from browserbase import Browserbase
from browserbase.types.session_create_params import BrowserSettings, BrowserSettingsContext
import time
import os
from langchain_openai import ChatOpenAI
from browser_use.browser.browser import Browser, BrowserConfig, BrowserContextConfig
from browser_use import Agent
from playwright.async_api import async_playwright
from playwright.sync_api import sync_playwright

import asyncio

# BROWSERBASE LOGIC

BROWSERBASE_PROJECT_ID = "a8994b49-d8d5-4502-8f54-214d80089b6c"
BROWSERBASE_API_KEY = os.environ["BROWSERBASE_API_KEY"]
USE_BROWSERBASE = True

async def main():
    cdp_url = None
    if USE_BROWSERBASE:
        bb = Browserbase(api_key=os.environ.get("BROWSERBASE_API_KEY"))
        
        # Create new context
        context = bb.contexts.create(project_id=BROWSERBASE_PROJECT_ID)
        
        browser_settings = BrowserSettings(context=BrowserSettingsContext(id=context.id, persist=False))
        session = bb.sessions.create(project_id=BROWSERBASE_PROJECT_ID, browser_settings=browser_settings)
        session_id = session.id
        debug_urls = bb.sessions.debug(session_id)
        debug_connection_url = debug_urls.debugger_fullscreen_url
        # This is the URL that we need to use to connect to the browser
        print("Browserbase Debug Connection URL: ", debug_connection_url)  
        cdp_url = session.connect_url
        
    
    browser = Browser(config=BrowserConfig(
        headless=False,
        cdp_url=cdp_url,
        new_context_config=BrowserContextConfig(
            # apply_click_styling=True,
            # apply_form_related=True,
            # browser_window_size={"width": 1024, "height": 768}, # These are values for Anthropic computer use
            save_recording_path='agent_executions/'
        )
        )
    )
    
    agent = Agent(
        task="Go to google and search for 'dog adoption", # "Go to https://www.ycombinator.com/apply and apply to YC. Apply with Tinder for dogs idea. Make up all the information. Don't open new tabs or windows!!!",
        llm=ChatOpenAI(model="gpt-4o"),
        browser=browser,
        retry_delay=10
    )
    result = await agent.run()
    browser.close()
    print(result)

asyncio.run(main())