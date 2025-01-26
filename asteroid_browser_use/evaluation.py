import datetime
import json
import os
import logging
from browser_use import Agent
from pydantic import BaseModel
from openai import OpenAI
import base64

logger = logging.getLogger(__name__)

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

async def finalize_task(agent: Agent, task_name: str, run_id: str):
    """
    Creates a folder for the agent execution, saves the conversation summary, 
    the final GIF (if available), the final score, etc.
    """
    # Create a timestamp
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"agent_executions/recording_{task_name.replace(' ', '_')}_{time_str}_{run_id}"
    os.makedirs(folder_name, exist_ok=True)

    # Create the history GIF
    gif_path = os.path.join(folder_name, f"{task_name.replace(' ', '_')}_{time_str}_{run_id}.gif")
    agent.create_history_gif(output_path=gif_path, show_goals=False, show_task=False, show_logo=False)
    
    # Summarize conversation
    summary = await summarize_conversation(agent)
    summary_path = os.path.join(folder_name, f"summary_{time_str}_{run_id}.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)

    # Evaluate task success
    score = await evaluate_task_success(agent)
    score_path = os.path.join(folder_name, f"score_{time_str}_{run_id}.txt")
    with open(score_path, "w", encoding="utf-8") as f:
        f.write(f"Task completion score (1-10): {score.score}\n\n{score.explanation}")
    logger.info(f"Agent scored on the completion of the task: {score}")

    # (Optional) Save entire conversation
    # conversation_path = os.path.join(folder_name, f"conversation_{run_id}.json")
    # with open(conversation_path, "w", encoding="utf-8") as f:
    #     json.dump(agent.message_manager.get_messages(), f, indent=2, ensure_ascii=False) #TODO: This fails: TypeError: Object of type SystemMessage is not JSON serializable

    logger.info(f"Finalized task data saved in {folder_name}")

