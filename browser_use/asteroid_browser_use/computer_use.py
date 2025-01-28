import uuid
import datetime
import base64
import anthropic
import logging

from browser_use.browser.context import BrowserContext
from browser_use.agent.views import ActionResult

logger = logging.getLogger(__name__)

async def execute_computer_actions(actions, page):
    if isinstance(actions, dict):
        actions = [actions]

    if not hasattr(page, "_anthropic_state"):
        page._anthropic_state = {"mouse_x": 0, "mouse_y": 0}
    mouse_x = page._anthropic_state["mouse_x"]
    mouse_y = page._anthropic_state["mouse_y"]

    for action_dict in actions:
        action_name = action_dict.get("action")
        coordinate = action_dict.get("coordinate")
        text = action_dict.get("text")

        if not action_name:
            print("No action provided; skipping.")
            continue

        # Mouse movement
        if action_name == "mouse_move":
            if not isinstance(coordinate, list) or len(coordinate) != 2:
                print("Error: 'mouse_move' requires 'coordinate' = [x, y]. Skipping.")
                continue
            x, y = coordinate
            await page.mouse.move(x, y)
            mouse_x, mouse_y = x, y
            print(f"Moved mouse to ({x}, {y}).")

        # Left-click drag
        elif action_name == "left_click_drag":
            if not isinstance(coordinate, list) or len(coordinate) != 2:
                print("Error: 'left_click_drag' requires 'coordinate' = [x, y]. Skipping.")
                continue
            await page.mouse.down()
            print(f"Mouse down at ({mouse_x}, {mouse_y}).")
            x, y = coordinate
            await page.mouse.move(x, y)
            print(f"Mouse dragged to ({x}, {y}).")
            mouse_x, mouse_y = x, y
            await page.mouse.up()
            print(f"Mouse up at ({x}, {y}).")

        elif action_name in ("left_click", "right_click", "middle_click", "double_click"):
            if action_name == "left_click":
                await page.mouse.click(mouse_x, mouse_y, button="left", click_count=1)
                print(f"Left click at ({mouse_x}, {mouse_y}).")
            elif action_name == "right_click":
                await page.mouse.click(mouse_x, mouse_y, button="right", click_count=1)
                print(f"Right click at ({mouse_x}, {mouse_y}).")
            elif action_name == "middle_click":
                await page.mouse.click(mouse_x, mouse_y, button="middle", click_count=1)
                print(f"Middle click at ({mouse_x}, {mouse_y}).")
            elif action_name == "double_click":
                await page.mouse.click(mouse_x, mouse_y, button="left", click_count=2, delay=100)
                print(f"Double click at ({mouse_x}, {mouse_y}).")

        elif action_name == "type":
            if not isinstance(text, str):
                print("Error: 'type' action must include 'text' string. Skipping.")
                continue
            await page.keyboard.type(text)
            print(f"Typed text: {text}")

        elif action_name == "key":
            if not isinstance(text, str):
                print("Error: 'key' action must include 'text' string. Skipping.")
                continue
            segments = text.split("+")
            combo_mods = segments[:-1]
            final_key = segments[-1]
            for mod in combo_mods:
                await page.keyboard.down(mod.strip())
            await page.keyboard.press(final_key.strip())
            for mod in reversed(combo_mods):
                await page.keyboard.up(mod.strip())
            print(f"Pressed key combination: {text}")

        elif action_name == "screenshot":
            path = action_dict.get("path", f"screenshot_{uuid.uuid4()}.png")
            await page.screenshot(path=path)
            print(f"Saved screenshot to {path}")

        elif action_name == "cursor_position":
            print(f"Current mouse position is ({mouse_x}, {mouse_y}).")

        else:
            print(f"Unknown action: {action_name}")

    page._anthropic_state["mouse_x"] = mouse_x
    page._anthropic_state["mouse_y"] = mouse_y

async def computer_use(browser: BrowserContext):
    page = await browser.get_current_page()
    screenshot_data = await page.screenshot()
    message_manager = getattr(browser.browser, 'message_manager', None)
    
    if not message_manager:
        return ActionResult(
            error='No message_manager found on this browser context!',
        )

    # 2) Now you can get the entire conversation:
    input_messages = message_manager.get_messages()
    
    # Convert the screenshot data to a base64 string
    screenshot_base64 = base64.b64encode(screenshot_data).decode('utf-8')
    
    ANTHROPIC_COMPUTER_USE_SYSTEM_PROMPT = """
You are extremely capable browser agent. You will get a history of a browser session and a screenshot of the current page. You will then need to decide what to do next to fulfill the user goal!

Try to perform actions together. For example, if you want to move and click on an element, give me all the actions together - move mouse to element, click on element.
"""

    messages = [
        {
            'role': 'user',
            'content': 'This is the current conversation history: ' + str(input_messages)
        },
        {
            'role': 'user',
            'content': [
                {
                    'type': 'image',
                    'source': {
                        'type': 'base64',
                        'media_type': 'image/png',
                        'data': screenshot_base64,
                    },
                }
            ]
        },
        {
            'role': 'user',
            'content': "This is the current screenshot of the page. Don't take another screenshot, give me actions to perform on the page to move forward!"
        },
    ]

    client = anthropic.Anthropic()    

    response = client.beta.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=[
            {
                "type": "computer_20241022",
                "name": "computer",
                "display_width_px": 1024,
                "display_height_px": 768,
                "display_number": 1,
            },
        ],
        messages=messages,
        system=ANTHROPIC_COMPUTER_USE_SYSTEM_PROMPT,
        betas=["computer-use-2024-10-22"],
    )

    actions = []
    for content_item in response.content:
        if content_item.type == 'text':
            print(content_item.text)
        elif content_item.type == 'tool_use':
            tool_name = content_item.name
            action = content_item.input
            if tool_name == 'computer':
                actions.append(action)
    await execute_computer_actions(actions, page)
    return ActionResult(extracted_content=f'Executed actions: {actions}')

def register_computer_use_action(controller):
    @controller.action(
        'Perform computer use actions. Call this when you want to click outside of the highlighted elements! It will enable you to click anywhere on the page, for example dropdowns, buttons, etc.',
        requires_browser=True,
    )
    async def perform_computer_use(browser: BrowserContext):
        return await computer_use(browser)
