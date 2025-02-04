import uuid
import datetime
import logging

from browser_use.agent.views import ActionResult
from browser_use.browser.context import BrowserContext

from asteroid_sdk.interaction.helper import pause_run, wait_for_unpaused

logger = logging.getLogger(__name__)

async def write_to_file(content: str, folder_name: str):
    with open(f'{folder_name}/output_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.txt', 'a') as f:
        f.write(content)
    return ActionResult(extracted_content='Output appended to file')

async def get_text(index: int, browser: BrowserContext):
    dom_el = await browser.get_dom_element_by_index(index)

    if dom_el is None:
        return ActionResult(error=f'No element found at index {index}')
    try:
        # text_content = dom_el.get_all_text_till_next_clickable_element()
        text_content = dom_el.get_all_text()
        msg = f'Successfully retrieved text from element at index {index}'
        logger.info(f'{msg}: {text_content}')
        return ActionResult(extracted_content=text_content)
    except Exception as e:
        logger.debug(f'Error getting text content: {str(e)}')
        return ActionResult(error=f'Failed to get text from element at index {index}')

async def perform_get_human_supervisor_help(browser: BrowserContext, run_id: str):
    """
    Get help from to perform action in the browser. Human can take over the browser, perform the action and agent will continue execution.
    """
    try:
        pause_run(run_id)
        await wait_for_unpaused(run_id)
    except Exception as e:
        logger.error(f'Error pausing run {run_id}: {e}')
        return ActionResult(error=f'Failed to pause run {run_id}')

    
    return ActionResult(extracted_content='Run was paused, human supervisor corrected the state, agent is now able to continue execution')


async def perform_solve_captcha(browser: BrowserContext, run_id: str):
    """
    Solve a captcha.
    """
    # For now we escalate to human to solve captcha
    # TODO: Implement better captcha solving
    await get_human_supervisor_help(browser, run_id)

async def perform_screenshot(browser: BrowserContext, folder_name: str):
    path = f'{folder_name}/screenshot_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    page = await browser.get_current_page()
    await page.screenshot(path=path)

    msg = '📸  Screenshot taken'
    logger.info(msg)
    return ActionResult(extracted_content=msg, include_in_memory=True)

def register_asteroid_actions(controller, run_id: str, folder_name: str):
    @controller.action(
        'Write important output information to a file',
        requires_browser=False,
    )
    async def action_write_to_file(content: str):
        return await write_to_file(content, folder_name)
    
    # @controller.action(
    #     'Get text from element - retrieves the text content from a DOM element at the specified index',
    #     requires_browser=True,
    # )
    # async def action_get_text(index: int, browser: BrowserContext):
    #     return await get_text(index, browser)
    
    @controller.action(
        'Get help from a human to perform an action in the browser.',
        requires_browser=True,
    )
    async def get_human_supervisor_help(browser: BrowserContext):
        return await perform_get_human_supervisor_help(browser, run_id)
    
    @controller.action(
        'Screenshot the current page', requires_browser=True
    )
    async def screenshot(browser: BrowserContext):
        return await perform_screenshot(browser, folder_name)

    @controller.action(
        'Solve a captcha', requires_browser=True
    )
    async def solve_captcha(browser: BrowserContext):
        return await perform_solve_captcha(browser, run_id)


browser_use_tool = {
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
