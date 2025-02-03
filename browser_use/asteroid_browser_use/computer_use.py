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
            try:
                for mod in combo_mods:
                    await page.keyboard.down(mod.strip())
                await page.keyboard.press(final_key.strip())
                for mod in reversed(combo_mods):
                    await page.keyboard.up(mod.strip())
                print(f"Pressed key combination: {text}")
            except Exception as e:
                print(f"Error pressing key combination: {text}")
                print(f"Error: {e}")
            

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

def preprocess_input_messages(messages):
    """
    Preprocess the conversation history messages.
    This function takes a list of message objects and returns a cleaned single string.
    It:
      - Collapses extra whitespace.
      - Removes verbose error details by keeping only the first part of messages that include 'Error:'.
    """
    processed = []
    for msg in messages:
        # If the message has a 'content' attribute, use it; otherwise, convert the message to string.
        content = getattr(msg, 'content', str(msg))
        # Collapse multiple whitespace into a single space and remove newlines.
        cleaned = " ".join(content.split())
        # If an error is present, trim the content before the 'Error:' part.
        if "Error:" in cleaned:
            cleaned = cleaned.split("Error:")[0].strip()
        processed.append(cleaned)
    return "\n".join(processed)

async def anthropic_computer_use(browser: BrowserContext, width: int, height: int):
    page = await browser.get_current_page()
    screenshot_data = await page.screenshot()
    message_manager = getattr(browser.browser, 'message_manager', None)
    llm = getattr(browser.browser, 'llm', None)
    if llm is None or getattr(llm, '_client', None) is None:
        client = anthropic.Anthropic() 
    else:
        client = llm._client
        
    if not message_manager:
        return ActionResult(
            error='No message_manager found on this browser context!',
        )
    
    # Retrieve and preprocess the conversation history.
    input_messages = message_manager.get_messages()
    preprocessed_history = preprocess_input_messages(input_messages)
    
    # Improved system prompt with very clear instructions.
    ANTHROPIC_COMPUTER_USE_SYSTEM_PROMPT = """
You are an advanced browser automation agent with the following responsibilities:
1. You receive a preprocessed conversation history that summarizes the browser session – including system messages,
   human inputs, tool outputs, and error notifications.
2. You also receive a current screenshot of the webpage as a base64-encoded PNG image.
3. Analyze both the preprocessed history and the screenshot to determine the current state of the webpage.
4. Based solely on these inputs, decide the next sequence of cohesive browser actions to achieve the user's goal.
5. NEVER instruct the system to take a new screenshot – use the provided image without any further screenshot requests.
6. Your response must be in valid JSON format with a clear plan that includes:
   - "current_state": a dictionary containing:
         • "evaluation_previous_goal": your assessment of previous actions
         • "memory": any important context to be remembered
         • "next_goal": the next objective based on current context
   - "action": a list of browser actions required to complete the next step.
7. When actions are related (for example, moving the mouse and then clicking), combine them into consecutive steps.
Ensure clarity and conciseness in your decision-making.
"""
    
    messages = [
        {
            'role': 'user',
            'content': 'Here is the preprocessed conversation history:\n' + preprocessed_history
        },
        {
            'role': 'user',
            'content': (
                "Below is the most recent screenshot of the current page. Do NOT request or take another screenshot. "
                "Analyze this image along with the above conversation history, and determine the next set "
                "of coordinated browser actions (for example, moving the mouse to an element and clicking it)."
            )
        },
        {
            'role': 'user',
            'content': [
                {
                    'type': 'image',
                    'source': {
                        'type': 'base64',
                        'media_type': 'image/png',
                        'data': base64.b64encode(screenshot_data).decode('utf-8'),
                    },
                }
            ]
        }
    ]
    
    response = client.beta.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=[
            {
                "type": "computer_20241022",
                "name": "computer",
                "display_width_px": width,
                "display_height_px": height,
                "display_number": 1,
            },
        ],
        messages=messages,
        system=ANTHROPIC_COMPUTER_USE_SYSTEM_PROMPT,
        betas=["computer-use-2024-10-22"],
    )
    
    actions = []
    text_content = ""
    for content_item in response.content:
        if content_item.type == 'text':
            text_content += content_item.text
            print(content_item.text)
        elif content_item.type == 'tool_use':
            tool_name = content_item.name
            action = content_item.input
            if tool_name == 'computer':
                actions.append(action)
                
    logger.info(f'Executing computer use actions: {actions}')
    await execute_computer_actions(actions, page)
    return ActionResult(extracted_content=f'Text content: {text_content}\nExecuted actions: {actions}', include_in_memory=True)

def register_computer_use_action(controller, width: int, height: int):
    @controller.action(
        'Perform computer use actions. Call this when you want to click outside of the highlighted elements! It will enable you to click anywhere on the page, for example dropdowns, buttons, etc.',
        requires_browser=True,
    )
    async def computer_use(browser: BrowserContext):
        return await anthropic_computer_use(browser, width, height)