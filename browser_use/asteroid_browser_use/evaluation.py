import datetime
import json
import os
import logging
from browser_use import Agent
from pydantic import BaseModel
from openai import OpenAI
import base64
import requests
from pathlib import Path
from PIL import Image
import io
import asyncio
import tempfile

logger = logging.getLogger(__name__)

try:
    from moviepy import ImageSequenceClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    logger.warning("moviepy not installed, video export will not be available.")

# Initialize OpenAI client
openai_client = OpenAI()

# Define Pydantic models for structured outputs

class TaskSuccessResponse(BaseModel):
    score: int
    explanation: str

class ConversationSummary(BaseModel):
    summary: str

async def evaluate_task_success(agent: Agent) -> TaskSuccessResponse:
    """
    Call the OpenAI API to rate how well the agent completed the instructions (1-10). 10 is the highest score.
    It uses both the final conversation and the agent's original instructions.
    Returns the numerical score.
    """

    conversation = agent.message_manager.get_messages()
    conversation_text = str(conversation) #TODO: Improve parsing of the conversation

    # Encode all screenshots to Base64
    screenshots = [agent_history.state.screenshot for agent_history in agent.history.history if agent_history.state.screenshot]
    screenshot_messages = []
    for screenshot in screenshots:
        try:
            # base64_image = encode_image(screenshot)
            image_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{screenshot}"}
            }
            screenshot_messages.append(image_message)
        except Exception as e:
            logger.error(f"Error encoding screenshot: {e}")

    score_prompt = f"""
Instructions:
{agent.task}

Conversation:
{conversation_text}

Evaluate the agent's performance on a scale from 1 to 10, where 1 means not completed and 10 means fully completed. Consider if all instructions were followed and tasks completed. Provide a JSON response with a score and explanation.

Example:
{{
  "score": 8,
  "explanation": "The agent completed most tasks but missed some details."
}}
"""

    # Prepare the messages with text and images
    messages = [
        {"role": "system", "content": "You are an evaluation assistant. Assess the agent's task completion diligently, providing a score and explanation."},
        {"role": "user", "content": score_prompt},
    ]

    # Append screenshot image messages
    messages.append({"role": "user", "content": screenshot_messages})

    try:
        completion = openai_client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=messages,
            response_format=TaskSuccessResponse,
        )
        task_success = completion.choices[0].message.parsed
        return task_success
    except Exception as e:
        logger.error(f"Error evaluating task success: {e}")
        return TaskSuccessResponse(score=0, explanation="Error evaluating task success")

class SummaryResponse(BaseModel):
    summary: str

async def summarize_conversation(agent: Agent) -> str:
    """
    Summarize the conversation to store for future use.
    """
    conversation = agent.message_manager.get_messages()
    conversation_text = str(conversation) #TODO: Improve parsing of the conversation
    summarize_prompt = f"""
Provide a bullet-point summary of the steps the agent took to accomplish the task.

Task:
{agent.task}

Conversation:
{conversation_text}

Respond with a JSON object containing the summary.

Example:
{{
  "summary": "- Step 1\n- Step 2\n- Step 3"
}}
"""

    try:
        completion = openai_client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are a conversation summarizer."},
                {"role": "user", "content": summarize_prompt},
            ],
            response_format=SummaryResponse,
        )
        summary = completion.choices[0].message.parsed.summary
        return summary
    except Exception as e:
        logger.error(f"Error summarizing conversation: {e}")
        return "Summary unavailable."

async def convert_rrweb_recording(
    events_file: str,
    output_file: str,
    output_format: str = "gif",
    fps: int = 2,
    browser_size: dict = None
):
    """
    Convert rrweb events to either GIF or MP4 video.
    By default, the rrweb events are replayed in a minimal page (without the rrweb-player UI).
    
    :param events_file: Path to the .json file containing rrweb events
    :param output_file: Output path for the GIF or video file
    :param output_format: "gif" or "mp4"
    :param fps: Frames per second for the output (used for video or for GIF timing)
    :param browser_size: dict - if provided, we use this to set the browser size
    """
    from playwright.async_api import async_playwright

    # Load rrweb events
    with open(events_file, 'r', encoding="utf-8") as f:
        events = json.load(f)

    # HTML without the rrweb-player UI
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8" />
        <script src="https://cdn.jsdelivr.net/npm/rrweb@latest/dist/rrweb.min.js"></script>
        <style>
            body {{
                margin: 0;
                overflow: hidden;
            }}
        </style>
    </head>
    <body></body>
    <script>
        const events = {json.dumps(events)};
        const replayer = new rrweb.Replayer(events, {{
            root: document.body,
            skipInactive: true,
        }});
        replayer.play();
    </script>
    </html>
    """

    # Write our minimal replay HTML to a temporary file
    temp_html = Path("temp_replay.html")
    temp_html.write_text(html_content, encoding="utf-8")

    # We'll collect frames as PIL Images in RAM
    frames = []

    # Set desired dimensions
    desired_width = browser_size.get('width', 800) if browser_size else 800
    desired_height = browser_size.get('height', 600) if browser_size else 600

    # Use Playwright to open the page and capture
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        # Create a new browser context with device_scale_factor set to 1
        context = await browser.new_context(
            viewport={"width": desired_width, "height": desired_height},
            device_scale_factor=1
        )
        page = await context.new_page()

        # Adjust viewport if needed
        await page.set_viewport_size({"width": desired_width, "height": desired_height})

        await page.goto(f"file://{temp_html.absolute()}")
        # Give some time for the replay to start
        await page.wait_for_timeout(1000)

        # Capture frames on incremental events
        for event in events:
            if event.get('type') == 3:  # 'incremental' or mouse interaction
                await asyncio.sleep(1.0 / fps)
                screenshot = await page.screenshot(type='png')
                image = Image.open(io.BytesIO(screenshot)).convert("RGB")
                # Ensure the image has the correct dimensions
                image = image.resize((desired_width, desired_height), Image.ANTIALIAS)
                frames.append(image)
                logger.debug(f"Captured frame {len(frames)}")

        await browser.close()

    # Cleanup the temporary HTML
    temp_html.unlink()

    if not frames:
        logger.warning("No frames captured for rrweb events. Output will not be created.")
        return

    if output_format.lower() == "gif":
        # Save frames as GIF
        frames[0].save(
            output_file,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / fps),  # ms per frame
            loop=0,
            optimize=True
        )
        logger.info(f"Browserbase replay GIF created at: {output_file}")

    elif output_format.lower() == "mp4":
        if not MOVIEPY_AVAILABLE:
            logger.error("moviepy not installed. Unable to create MP4 video.")
            return

        # Convert frames to a video with MoviePy
        with tempfile.TemporaryDirectory() as tmpdir:
            frame_paths = []
            for i, frame in enumerate(frames):
                # Ensure the frame is in RGB mode
                if frame.mode != 'RGB':
                    frame = frame.convert('RGB')
                
                frame_path = os.path.join(tmpdir, f"frame_{i:06d}.png")
                frame.save(frame_path, format="PNG")
                frame_paths.append(frame_path)
                logger.debug(f"Saved frame {i} to {frame_path}")

            # Create the video clip
            try:
                clip = ImageSequenceClip(frame_paths, fps=fps)
                clip.write_videofile(output_file, codec='libx264', audio=False)
                logger.info(f"Browserbase replay MP4 video created at: {output_file}")
            except Exception as e:
                logger.error(f"Failed to create MP4 video: {e}")
    else:
        logger.error(f"Unsupported output_format: {output_format}. Please use 'gif' or 'mp4'.")

async def finalize_task(agent, task_name: str, run_id: str, folder_name: str,
                        evaluate: bool = False, session_id: str = None,
                        output_format: str = "gif", browser_size: dict = None):
    """
    Creates a folder for the agent execution, saves the conversation summary, 
    the final GIF (if available), the final score, etc.
    Additionally, if session_id is provided (Browserbase session), fetches the 
    rrweb events and converts them to either a GIF or MP4 video, and saves to the folder.
    
    :param agent: The Agent instance
    :param task_name: The name of the task
    :param run_id: The unique run ID
    :param folder_name: Folder path to store data
    :param evaluate: bool - if True, evaluate the agent's performance
    :param session_id: str or None - if provided, we attempt to fetch Browserbase session data
    :param output_format: "gif" or "mp4" to specify how we want the rrweb data captured
    :param browser_size: dict - if provided, we use this to set the browser size
    """
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(folder_name, exist_ok=True)

    # 1. Create the agent's locally captured history GIF
    logger.info(f"Creating history GIF for {folder_name}")
    gif_path = os.path.join(folder_name, f"{task_name.replace(' ', '_')}_{time_str}_{run_id}.gif")
    agent.create_history_gif(
        output_path=gif_path,
        show_goals=False,
        show_task=False,
        show_logo=False
    )
    logger.info(f"History GIF created at {gif_path}")

    # 2. Summarize conversation
    logger.info(f"Summarizing conversation for {folder_name}")
    summary = await summarize_conversation(agent)
    summary_path = os.path.join(folder_name, f"summary_{time_str}_{run_id}.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)

    # 3. Evaluate task success (if requested)
    if evaluate:
        logger.info(f"Evaluating task success for {folder_name}")
        score = await evaluate_task_success(agent)
        score_path = os.path.join(folder_name, f"score_{time_str}_{run_id}.txt")
        with open(score_path, "w", encoding="utf-8") as f:
            f.write(f"Task completion score (1-10): {score.score}\n\n{score.explanation}")
        logger.info(f"Agent scored on the completion of the task: {score}")

    # 4. If Browserbase session was used, retrieve rrweb events and convert to GIF or MP4
    if session_id:
        try:
            logger.info(f"Browserbase session {session_id} detected. Attempting to fetch recording.")
            # Fetch the rrweb recording from Browserbase
            url = f"https://api.browserbase.com/v1/sessions/{session_id}/recording"
            headers = {"X-BB-API-Key": os.environ.get("BROWSERBASE_API_KEY")}
            response = requests.get(url, headers=headers)
            response_data = response.json()

            json_file_path = os.path.join(folder_name, f"browserbase_session_{session_id}.json")
            with open(json_file_path, "w", encoding="utf-8") as f:
                json.dump(response_data, f, indent=4)

            # Convert the rrweb JSON to GIF or MP4
            extension = output_format.lower()
            if extension not in ["gif", "mp4"]:
                extension = "gif"
                logger.warning("Invalid output_format. Defaulting to 'gif'.")

            session_recording_path = os.path.join(folder_name, f"browserbase_session_{session_id}.{extension}")
            await convert_rrweb_recording(
                events_file=json_file_path,
                output_file=session_recording_path,
                output_format=extension,
                fps=2,
                browser_size=browser_size
            )
        except Exception as e:
            logger.warning(f"Failed to retrieve or convert Browserbase session recording: {e}")

    logger.info(f"Finalized task data saved in {folder_name}")
    # (Optional) Save entire conversation
    # conversation_path = os.path.join(folder_name, f"conversation_{run_id}.json")
    # with open(conversation_path, "w", encoding="utf-8") as f:
    #     json.dump(agent.message_manager.get_messages(), f, indent=2, ensure_ascii=False) #TODO: This fails: TypeError: Object of type SystemMessage is not JSON serializable

    logger.info(f"Finalized task data saved in {folder_name}")

