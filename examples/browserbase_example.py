
import requests
from playwright.sync_api import sync_playwright, Page
from browserbase import Browserbase
import time
import os
from langchain_openai import ChatOpenAI
from browser_use import Agent, Browser, BrowserConfig
from playwright.async_api import async_playwright, Page

import asyncio

# BROWSERBASE LOGIC

BROWSERBASE_PROJECT_ID = "a8994b49-d8d5-4502-8f54-214d80089b6c"
BROWSERBASE_API_KEY = os.environ["BROWSERBASE_API_KEY"]
USE_BROWSERBASE = True

def create_session() -> str:
    """
    Create a Browserbase session - a single browser instance.

    :returns: The new session's ID.
    """
    sessions_url = "https://api.browserbase.com/v1/sessions"
    headers = {
        "Content-Type": "application/json",
        "x-bb-api-key": BROWSERBASE_API_KEY,
    }
    # Include your project ID in the json payload
    json = {"projectId": BROWSERBASE_PROJECT_ID}

    response = requests.post(sessions_url, json=json, headers=headers)

    # Raise an exception if there wasn't a good response from the endpoint
    response.raise_for_status()
    return response.json()["id"]

def get_browser_url(session_id: str) -> str:
    """
    Get the URL to show the live view for the current browser session.

    :returns: URL
    """
    session_url = f"https://api.browserbase.com/v1/sessions/{session_id}/debug"
    headers = {
        "Content-Type": "application/json",
        "x-bb-api-key": BROWSERBASE_API_KEY,
    }
    response = requests.get(session_url, headers=headers)

    # Raise an exception if there wasn't a good response from the endpoint
    response.raise_for_status()
    return response.json()["debuggerFullscreenUrl"]


async def main():
    
    
    if USE_BROWSERBASE:
        session_id = create_session()
        live_browser_url = get_browser_url(session_id)
        print("Browserbase Session ID: ", session_id)
        print("Browserbase Live Browser URL: ", live_browser_url)
        
    async with async_playwright() as playwright:
        
        cdp_url = None
        if USE_BROWSERBASE:
            browser = await playwright.chromium.connect_over_cdp(
                f"wss://connect.browserbase.com?apiKey={BROWSERBASE_API_KEY}&sessionId={session_id}"
            )
            
            cdp_url = f"wss://connect.browserbase.com?apiKey={BROWSERBASE_API_KEY}&sessionId={session_id}"
            
        browser = Browser(config=BrowserConfig(
            headless=False,
            cdp_url=cdp_url
            )
        )
        
         agent = Agent(
            task="Go to https://www.ycombinator.com/apply and apply to YC. Apply with Tinder for dogs idea. Make up all the information. Don't open new tabs or windows!!!",
            llm=ChatOpenAI(model="gpt-4o"),
            browser=browser,
            retry_delay=15
        )
        result = await agent.run()
        browser.close()
        print(result)

asyncio.run(main())



# def run(browser_tab: Page):
#     # Instruct the browser to go to the SF MOMA page, and click on the Tickets link
#     browser_tab.goto("https://www.sfmoma.org")
#     browser_tab.get_by_role("link", name="Tickets").click()

#     # Print out a bit of info about the page it landed on
#     print(f"{browser_tab.url=} | {browser_tab.title()=}")

#     ...


# with sync_playwright() as playwright:
#     # A session is created implicitly
#     browser = playwright.chromium.connect_over_cdp(
#         f"wss://connect.browserbase.com?apiKey={API_KEY}"
#     )

#     # Print a bit of info about the browser we've connected to
#     print(
#         "Connected to Browserbase.",
#         f"{browser.browser_type.name} version {browser.version}",
#     )

#     context = browser.contexts[0]
#     browser_tab = context.pages[0]

#     try:
#         # Perform our browser commands
#         run(browser_tab)

#     finally:
#         # Clean up
#         browser_tab.close()
#         browser.close()


