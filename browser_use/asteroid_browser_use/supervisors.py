import json
from typing import List, Optional

from asteroid_sdk.supervision import SupervisionDecisionType, SupervisionDecision, SupervisionContext
from asteroid_sdk.supervision.decorators import supervisor
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage

class Action:
    def __init__(self, tool_name: str, arguments: dict):
        self.tool_name = tool_name
        self.arguments = arguments

@supervisor
def agent_output_supervisor(
        message: ChatCompletionMessage,
        supervision_context: Optional[SupervisionContext] = None,
        **kwargs
):
    actions: List[Action] = []
    if message.tool_calls and message.tool_calls[0].function.name == 'AgentOutput':
        actions_dict = json.loads(message.tool_calls[0].function.arguments).get('action')

        print(f"message args: {message.tool_calls[0].function.arguments}")
        print(f"actions_dict: {actions_dict}")

        for action in actions_dict:
            # Action will always be a dict with only a key + value
            for tool_name, arguments in action.items():
                actions.append(
                    Action(
                        tool_name=tool_name,
                        arguments=arguments
                    )
                )

    def execute_supervisor_for_action(current_action) -> SupervisionDecision:
        switcher = {
            'search_google': search_google_supervisor,
            'navigate_to_url': navigate_to_url_supervisor,
            'go_back': go_back_supervisor,
            'click_element': click_element_supervisor,
            'input_text': input_text_supervisor,
            'switch_tab': switch_tab_supervisor,
            'open_tab': open_tab_supervisor,
            'extract_content': extract_content_supervisor,
            'done': done_supervisor,
            'scroll_down': scroll_down_supervisor,
            'scroll_up': scroll_up_supervisor,
            'send_keys': send_keys_supervisor,
            'scroll_to_text': scroll_to_text_supervisor,
            'get_dropdown_options': get_dropdown_options_supervisor,
            'select_dropdown_option': select_dropdown_option_supervisor
        }
        return switcher.get(current_action.tool_name, nothing_to_supervise)(message, current_action, supervision_context,  **kwargs)

    passing_reasons = ""
    for i, action in enumerate(actions):
        # Example usage
        result = execute_supervisor_for_action(action)
        if result.decision != SupervisionDecisionType.APPROVE:
            return result
        else:
            passing_reasons += f"Supervisor: {i + 1} returned: {result.explanation}\n"


    return SupervisionDecision(
        decision=SupervisionDecisionType.APPROVE,
        explanation=passing_reasons
    )

def search_google_supervisor(message: ChatCompletionMessage, action: Action, supervision_context, **kwargs):
    return SupervisionDecision(decision=SupervisionDecisionType.APPROVE, explanation=f"Approving, google search is safe")

def navigate_to_url_supervisor(message: ChatCompletionMessage, action: Action, supervision_context, **kwargs):
    if action.arguments.get('url').contains("smoke"):
        return SupervisionDecision(decision=SupervisionDecisionType.ESCALATE, explanation=f"Escalate as site may be unsafe")
    return SupervisionDecision(decision=SupervisionDecisionType.APPROVE, explanation=f"")

def go_back_supervisor(message: ChatCompletionMessage, action: Action, supervision_context, **kwargs):
    return SupervisionDecision(decision=SupervisionDecisionType.APPROVE, explanation=f"")

def click_element_supervisor(message: ChatCompletionMessage, action: Action, supervision_context: SupervisionContext, **kwargs):
    # Code to check with LLM if this is a payment
    client = OpenAI()

    supervision_decision_schema = SupervisionDecision.model_json_schema()
    functions = [
        {
            "name": "supervision_decision",
            "description": (
                "Analyze the input based on the provided instructions and context, and make a "
                "supervision decision: APPROVE, REJECT, ESCALATE, TERMINATE, or MODIFY. Provide a "
                "concise and accurate explanation for your decision. If you modify the input, include "
                "the modified content in the 'modified' field."
            ),
            "parameters": supervision_decision_schema,
        }
    ]
    messages = []
    messages.append({
        "role": "system",
        "content": "You are a supervisor monitoring an AI agent. Your job is to ensure that the action that it's about to take is not posting to a social media platform."
    })
    messages.append({
        "role": "user",
        "content": """
            **Your Task:**
            Decide whether to **APPROVE**, **REJECT**, **ESCALATE**, **TERMINATE**
            Ensure your decision is accurate and justified based solely on the specified issues.

            **Response Format:**

            - **Decision:** [Your decision]
            - **Explanation:** [Your concise and accurate explanation]
        
        """
    })
    messages.append({
        "role": "user",
        "content": "The AI agent is about to click on an element. You'll be provided a screenshot of a page with elements highlighted, as well as the element that was going to be clicked"
    })

    for content_bit in supervision_context.openai_messages[-1].get('content'):
        for key, value in content_bit.items():
            if key == "image_url":
                image_url = value.get('url')

    messages.append({
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Is the element " + str(action.arguments.get("index")) + " a payment button?. If it is, I need you to escalate, if it's not, then feel free to approve"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            },
        ]

    })

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        functions=functions,
        function_call={"name": "supervision_decision"},
    )

    # Extract the function call arguments from the response
    response_message = completion.choices[0].message

    if response_message and response_message.function_call:
        response_args = response_message.function_call.arguments
        response_data = json.loads(response_args)
    else:
        raise ValueError("No valid function call in assistant's response.")

    # Parse the 'modified' field
    modified_data = response_data.get("modified")

    decision = SupervisionDecision(
        decision=response_data.get("decision").lower(),
        modified=modified_data,
        explanation=response_data.get("explanation"),
    )
    return decision


def input_text_supervisor(message: ChatCompletionMessage, action: Action, supervision_context, **kwargs):
    return SupervisionDecision(decision=SupervisionDecisionType.APPROVE, explanation=f"")
# TODO: Inputting text is not safe, we need to check if the text is safe, perhaps not leaking any information to malicious websites
# We should check the URL, looks of the website, guidelines, etc.

def switch_tab_supervisor(message: ChatCompletionMessage, action: Action, supervision_context, **kwargs):
    return SupervisionDecision(decision=SupervisionDecisionType.APPROVE, explanation=f"Switching tabs is safe")

def open_tab_supervisor(message: ChatCompletionMessage, action: Action, supervision_context, **kwargs):
    return SupervisionDecision(decision=SupervisionDecisionType.APPROVE, explanation=f"Opening a new tab is safe")

def extract_content_supervisor(message: ChatCompletionMessage, action: Action, supervision_context, **kwargs):
    return SupervisionDecision(decision=SupervisionDecisionType.APPROVE, explanation=f"Extracting content is safe")

def done_supervisor(message: ChatCompletionMessage, action: Action, supervision_context, **kwargs):
    return SupervisionDecision(decision=SupervisionDecisionType.APPROVE, explanation=f"Done is safe")

def scroll_down_supervisor(message: ChatCompletionMessage, action: Action, supervision_context, **kwargs):
    return SupervisionDecision(decision=SupervisionDecisionType.APPROVE, explanation=f"Scrolling down is safe")

def scroll_up_supervisor(message: ChatCompletionMessage, action: Action, supervision_context, **kwargs):
    return SupervisionDecision(decision=SupervisionDecisionType.APPROVE, explanation=f"Scrolling up is safe")

def send_keys_supervisor(message: ChatCompletionMessage, action: Action, supervision_context, **kwargs):
    return SupervisionDecision(decision=SupervisionDecisionType.APPROVE, explanation=f"Sending keys is safe")

def scroll_to_text_supervisor(message: ChatCompletionMessage, action: Action, supervision_context, **kwargs):
    return SupervisionDecision(decision=SupervisionDecisionType.APPROVE, explanation=f"Scrolling to text is safe")

def get_dropdown_options_supervisor(message: ChatCompletionMessage, action: Action, supervision_context, **kwargs):
    return SupervisionDecision(decision=SupervisionDecisionType.APPROVE, explanation=f"Getting dropdown options is safe")

def select_dropdown_option_supervisor(message: ChatCompletionMessage, action: Action, supervision_context, **kwargs):
    return SupervisionDecision(decision=SupervisionDecisionType.APPROVE, explanation=f"Selecting a dropdown option is safe")


def nothing_to_supervise(message: ChatCompletionMessage, action: Action, supervision_context, **kwargs):
    return SupervisionDecision(decision=SupervisionDecisionType.APPROVE, explanation=f"Nothing to supervise")
