import os
import json
import openai
import requests
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from firecrawl import FirecrawlApp


lock = threading.Lock()

# For illustration only; in a production system, store tokens securely (e.g., environment variables or vault).
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "...")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "...")
openai.api_key = os.getenv("OPENAI_API_KEY", "...")


def get_current_datetime() -> str:
    now = datetime.now()
    formatted_time = now.strftime("%A, %B %d, %Y, %H:%M:%S")
    return f"Current date and time:{formatted_time}"

def call_research_assistant(query: str, recency: str = "month") -> str:
    """
    Calls the Perplexity AI API with the given query.
    Returns the text content from the model’s answer.
    """
    url = "https://api.perplexity.ai/chat/completions"
    payload = {
        "model": "llama-3.1-sonar-large-128k-online",
        "messages": [
            {"role": "user", "content": query},
        ],
        "temperature": 0.7,
        "top_p": 0.9,
        "search_recency_filter": recency,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=180)
        response.raise_for_status()
        data = response.json()
        retval = data["choices"][0]["message"]["content"]
        #joined_citations = "\n".join(data["citations"])
        #citations = f"\n\nCitations:\n{joined_citations}"
        #retval = retval + citations

        print(f"* * *  Research Assistant Response  * * *\n\n{retval}\n\n")
        return retval
    except Exception as e:
        return f"Error calling Perplexity API: {str(e)}"

tools = [
    {
        "type": "function",
        "function": {
            "name": "call_research_assistant",
            "description": (
                "Use this to utilize a PhD grad student to perform research, "
                "they can only research one single intent question at a time, "
                "they have no context or prior knowledge of this conversation, "
                "you must give them the context and a single intention query. "
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A straight to the point concise succint question or search query to be sent to research assistant",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
]

def call_research_professional(question: str, model_version: str = "gpt-4o-mini") -> str:

    refactor_count = 1

    prompt = f"""
Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. You will generate increasingly better thoughts, solutions, and critical feedback to the user's question. Repeat the following 4 steps 5 times. 

**Rules:**
In your thoughts, solutions and critical feedback you must use Chain of Thought, thinking step by step, and thinking and logic assurance phrases like "I need to analyze this", "Now, let's break it down", "What's important here is", "Let's consider all possibilities", "I'll work through this logically", "It's crucial to understand", "Reasoning through this", "Let’s explore our options", "I must evaluate the situation", "This means we'll", "Our objective here is", "To tackle this challenge", "Let's examine the details", "This implies that", "In other words", "I need to weigh the pros and cons", "From what I understand", "I should double-check the facts", "I'll need to compare this with", "Clearly, I see that", "The real issue is", "Let's focus on the key points", "I'll approach this methodically", "It’s important to recognize", "I should assess the variables", "We’ll look at it step by step", "I'll need to identify the problem", "Another perspective might be", "To clarify", "I have to determine", "It's evident that", "Our main concern is", "Let's outline our approach", "We have to ensure that", "I'll need to formulate a plan", "Essentially, this suggests", "I'll conclude that", "What we need to consider is", "By exploring possibilities", "I ought to think through the implications", "Evaluating the outcomes", "It makes sense when I", "This leads us to", "I'll infer from", "I must establish a framework", "We will resolve this by", "The hypothesis is", "We can deduce that", etc...  You must use like phrases as part of your process reasoning your rationale. 

**Guidelines:**
Step 1. In the Thought section, detail your reasoning process using the specified format: <|begin_of_thought|> (individual thought with detailed steps separated with ' ') <|end_of_thought|> Each step should include detailed considerations such as analyzing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps.
Step 2. In the Solution section, based on diverse attempts, explorations, thoughts, critical feedback, and reflections from the past, systematically present a solution that you deem correct with actionable steps and details the user can take as a resolution. IMPORTANT: In this section you must focus on solving with specific answers to the user's question. It's not good enough to generalize the answers must be specific and aligned to the user's question. Provide sufficient metric-based guidance or resources for implementation. The solution should remain a logical, accurate, concise expression with detail of necessary steps needed to reach the conclusion, formatted as follows: <|begin_of_solution|> (formatted, precise, and clear solution) <|end_of_solution|> 
Step 3. In the Critical Feedback section, explore questions the user should have asked but didn't know to ask about the domain, class, and task. Critically explore the feedback and how it encourages or detracts exploring another related thought. Detail your rationale process with this specified format: <|critical_evaluation|> (Detailed critic items separated with ' ') <|end_of_critical_evaluation|> 
Step 4. Keep a running list of all the tool calls you could have used to enhance your reasoning. Detail your rationale process with this specified format: <|tool_requests|> (Detail one comprehesive multi-part question you would ask of a research assistant to help in your reasoning process' ') <|end_of_tool_requests|> 

Response only in JSON. The JSON should be a list (length 5) of dictionaries whose keys are "Thought", "Solution",  "Critical_Feedback", and "Tool_Calls"

``` user question
{question}
```
"""

    messages = [
        {"role": 'user', 'content': get_current_datetime() + '\n\n' + prompt},
    ]

    # Main loop for back-and-forth with the model
    for _ in range(50):

        # For debugging/logging
        print("~" * 80)
        print("\nMessage Stack:\n")
        try:
            print(json.dumps(messages, indent=4))
        except:
            try:
                json_obj = json.loads(messages)
                print(json.dumps(json_obj, indent=4))
            except:
                print(str(messages).replace("{'role'", "\n\n\n{'role'"))
        print("\n" + "~" * 80 + "\n")


        base_args = {
            "messages": messages,
            "response_format": {"type": "text"},
        }

        model_args = {
            "model": model_version,
            "tools": tools,
            "max_completion_tokens": 16383
        }

        # Merge common and model-specific settings
        args = {**base_args, **model_args}

        # Call OpenAI API with merged arguments
        response = openai.chat.completions.create(**args)

        msg = response.choices[0].message

        assistant_content = msg.content
        finish_reason = response.choices[0].finish_reason

        tool_calls = getattr(msg, "tool_calls", None)

        messages.append(msg)

        # If there are tool calls, handle them
        if tool_calls:
            for tc in tool_calls:
                func_name = tc["function"]["name"] if isinstance(tc, dict) else tc.function.name
                arguments_json = tc["function"]["arguments"] if isinstance(tc, dict) else tc.function.arguments

                print("^" * 80)
                print("\nTool Request")
                print(f"Tool name: {func_name}\nArguments: {arguments_json}\n")
                print("^" * 80 + "\n")

                # Attempt to parse arguments JSON
                try:
                    arguments = json.loads(arguments_json)
                except json.JSONDecodeError:
                    arguments = {}

                # Dispatch to the correct tool
                if func_name == "call_research_assistant":
                    query = arguments.get("query", "")
                    result = call_research_assistant(query)

                else:
                    result = f"Tool {func_name} is not implemented."

                tool_result_message = {'role': "tool", 'content': result}

                possible_id = getattr(tc, "id", None)
                if possible_id:
                    tool_result_message["tool_call_id"] = possible_id

                messages.append(tool_result_message)

            # After tool calls, continue loop so the model sees the new tool outputs
            continue

        # If no tool calls, check finish_reason
        if finish_reason == "stop":
            if refactor_count <= 6:
                if refactor_count % 3 == 1:
                    promptx = """
Run those tools you needed to run precisely as you defined.
"""
                    messages.append({'role': 'user', 'content': promptx})
                elif refactor_count % 3 == 2:
                    promptx = """
Critically evaluate this reasoning process against the user's initial request 
and tell me the pros / cons of the reasoning process and rate the answer 
against the user's initial request betweed 0.0 and 1.0
"""
                    messages.append({'role': 'user', 'content': promptx})
                elif refactor_count % 3 == 0:
                    promptx = """
Address the cons of the solution and go another 5 iterations of thoughts, 
solutions, critical feedback, and tool calls.
"""
                    messages.append({'role': 'user', 'content': promptx})

                refactor_count = refactor_count + 1
                continue

            elif refactor_count == 7:

                promptx = """
With a focus on the user's initial request in a long long long freeform 
text narrative detail the comprehensive solution without missing any detail. 
IMPORTANT: Response must focus on solving with specific answers to the user's 
question. It's not good enough to generalize the answers must be specific 
and aligned to the user's question. Provide sufficient metric-based guidance 
or resources for implementation. 
"""
                messages.append({'role': 'user','content': promptx,})

                refactor_count = refactor_count + 1
                continue 

            else:
                # The model gave a final answer
                if assistant_content:
                    print("\nAssistant:\n" + assistant_content)
                    return assistant_content
                else:
                    print("\nAssistant provided no content.")
                break

        elif finish_reason in ["length", "max_tokens", "content_filter"]:
            # The conversation got cut off or other forced stop
            print("The model's response ended due to finish_reason =", finish_reason)
            break

        # If we get here with no tool calls and not “stop,”
        # we can guess the model simply produced final text or there's no more to do
        if assistant_content.strip():
            print("\nAssistant:\n" + assistant_content)
            return assistant_content

    return "Failed"


def main():
    start_time = datetime.now()
    print(f"Start Time: {start_time.strftime('%H:%M:%S')}")

    with open('o1reasoning-input.txt', 'r') as input_file:
        user_question = input_file.read()
        
    final_answer = call_research_professional(user_question)

    with open('o1reasoning-ouput.txt', 'w') as output_file:
        output_file.write(final_answer)
    
    end_time = datetime.now()
    print(f"End Time: {end_time.strftime('%H:%M:%S')}")
    
    elapsed_time = (end_time - start_time).total_seconds()
    print(f"Elapsed Seconds: {elapsed_time}")

    print("\n--- End of conversation ---")


if __name__ == "__main__":
    main()
