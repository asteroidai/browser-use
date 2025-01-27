import uuid
import datetime
import logging

from browser_use.agent.views import ActionResult
from browser_use.browser.context import BrowserContext

logger = logging.getLogger(__name__)

async def write_to_file(content: str, run_id: str):
    with open(f'output_{run_id}.txt', 'w') as f:
        f.write(content)
    return ActionResult(extracted_content='Output written to file')

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

async def get_human_supervisor_help(browser: BrowserContext):
    """
    Get help from to perform action in the browser. Human can take over the browser, perform the action and agent will continue execution.
    """
    # Get help via the CLI
    print("Getting help via the CLI") # TODO: Implement escalation to Asteroid, we take over the browser, record action and then continue execution
    user_input = input("Please give me some help")
    # Get screenshot of the current page to see what the user was doing
    # path = f'screenshot_{uuid.uuid4()}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    # page = await browser.get_current_page()
    # await page.screenshot(path=path)
    # user_input = "I'm done"
    # Record user action
    # await browser.record_user_action(user_input) TODO: Implement this
    
    return ActionResult(extracted_content=user_input)

async def screenshot(browser: BrowserContext):
    path = f'screenshot_{uuid.uuid4()}.png'
    page = await browser.get_current_page()
    await page.screenshot(path=path)

    msg = '📸  Screenshot taken'
    logger.info(msg)
    return ActionResult(extracted_content=msg, include_in_memory=True)

def register_asteroid_actions(controller, run_id: str):
    @controller.action(
        'Write important output information to a file',
        requires_browser=False,
    )
    async def action_write_to_file(content: str):
        return await write_to_file(content, run_id)
    
    @controller.action(
        'Get text from element - retrieves the text content from a DOM element at the specified index',
        requires_browser=True,
    )
    async def action_get_text(index: int, browser: BrowserContext):
        return await get_text(index, browser)
    
    @controller.action(
        'Get human supervisor help - get help from a human to perform an action in the browser.',
        requires_browser=True,
    )
    async def action_get_human_supervisor_help(browser: BrowserContext):
        return await get_human_supervisor_help(browser)
    
    @controller.action(
        'Screenshot the current page', requires_browser=True
    )
    async def action_screenshot(browser: BrowserContext):
        return await screenshot(browser)