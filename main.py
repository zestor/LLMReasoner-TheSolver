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
import unicodedata
from o1reasoning_calculator import Calculator

lock = threading.Lock()

# For illustration only; in a production system, store tokens securely (e.g., environment variables or vault).
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "...")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "...")
openai.api_key = os.getenv("OPENAI_API_KEY", "...")

MAX_LOOPS = 100
MAX_ITERATIONS_AND_TOOLS = 5
MAX_SCORE_MISSES = 6
PASS_THRESHOLD = 0.91

scores = []

def unicode_to_ascii(text):
    text = markdown_latex_to_plain_text(text)
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')

def markdown_latex_to_plain_text(text):
    """
    Convert a string with markdown and LaTeX formatting into a plain text string.
    
    Parameters:
    - text: str, the input string containing markdown and LaTeX.
    
    Returns:
    - str, the plain text representation of the input string.
    """
    # Convert headers
    text = re.sub(r'^(#{1,6})\s*(.*?)\s*$', r'\2', text, flags=re.MULTILINE)
    
    # Convert Markdown bold and italics
    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'\1', text)  # Bold & Italic
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)      # Bold
    text = re.sub(r'\*(.+?)\*', r'\1', text)          # Italic
    
    # Convert Markdown lists
    text = re.sub(r'^\s*[-*+]\s+', '- ', text, flags=re.MULTILINE)  # Unordered lists

    # Convert nested lists or checkboxes
    text = re.sub(r'^\s*[0-9]+\.\s+', '', text, flags=re.MULTILINE)  # Remove ordered list numbers
    
    # Convert inline LaTeX fractions
    text = re.sub(r'\\frac{(.*?)}{(.*?)}', r'(\1/\2)', text)
    
    # Remove LaTeX environments and commands
    text = re.sub(r'\\(?:begin|end){.*?}', '', text)
    
    # Remove LaTeX delimiters for math expressions
    text = re.sub(r'\$+', '', text)

    # Simplify other LaTeX commands by removing backslashes
    text = re.sub(r'\\([a-zA-Z]+)', r'\1', text)

    # Handle math superscripts
    text = re.sub(r'\^(\{?.*?\}?)', lambda m: '**' + m.group(1).strip('{}'), text)
    
    # Handle LaTeX emphasis and braces
    text = re.sub(r'\{\\em (.+?)\}', r'\1', text)
    text = text.replace('{', '').replace('}', '')
    text = text.replace('\[', '[').replace('\]', ']')
    text = text.replace('\(', '(').replace('\)', ')')

    # Clean up extra whitespace
    text = ' '.join(text.split()).replace(' .', '.').replace(' ,', ',')
    
    return text.strip()

def add_score(score):
    # Append the new score to the global list
    scores.append(score)
    # Print all scores in order
    print_scores()

def print_scores():
    # Print scores separated by commas
    print("Scores:", ", ".join(map(str, scores)))
    # Log the tool result
    try:
        with open('o1reasoning-intermediate.txt', 'a') as output_file:
            output_file.write("^" * 80)
            output_file.write("\nScores:")
            output_file.write(", ".join(map(str, scores)))
            output_file.write("^" * 80 + "\n")                        

    except IOError:
        print("An error occurred while writing to the file.")    
    
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

        #print(f"* * *  Research Assistant Response  * * *\n\n{retval}\n\n")
                    
        return retval
    except Exception as e:
        return f"Error calling Perplexity API: {str(e)}"

def call_openai_gpt40_llm(prompt: str, model: str = "gpt-4o", messages: Optional[List[Dict[str, str]]] = None) -> str:
    """
    Calls OpenAI with model='gpt-4o' for advanced GPT-4 style reasoning or sub-queries.
    """
    helper_messages = []

    if messages is None:
        helper_messages = [
            {'role': 'user', 'content': get_current_datetime() + '\n' + prompt}
        ]
    else:
        helper_messages = messages.copy()
        # Append the user message if messages were provided
        helper_messages.append({'role': 'user', 'content': prompt})
    
    try:
        completion = openai.chat.completions.create(
            model=model,
            messages=helper_messages
        )

        return completion.choices[0].message.content
    except Exception as e:
        return f"Error calling OpenAI model='{model}': {str(e)}"

tools=[
{
    "type": "function",
    "function": {
    "name": "calculate",
    "description": (
        "A mathematical expression to evaluate. The expression can include simple arithmetic operations like addition (+), subtraction (-), "
        "multiplication (*), division (/), exponentiation (**), and parentheses for grouping. Additionally, it supports advanced functions such as "
        "trigonometric functions (sin, cos, tan), logarithmic functions (log, log10), exponential functions (exp), complex numbers (using j or I), "
        "matrix operations, special functions (e.g., gamma, factorial), and variable assignments. Examples include: "
        "\"2 + 3 * 4\", "
        "\"sin(pi / 4) + log(E**2, 10)\", "
        "\"a = 5\", "
        "\"(a + b) * (c - d)\", "
        "\"e**(1j * pi) + cosh(pi/3) - sinh(pi/3)\", etc."
    ),
        "parameters": {
        "type": "object",
        "required": [
        "expression",
        "precision"
        ],
        "properties": {
        "expression": {
            "type": "string",
            "description": "A Python mathematical expression to evaluate"
        },
        "precision": {
            "type": "number",
            "description": "Number of decimal places for the answer"
        }
        },
        "additionalProperties": False
    },
    "strict": True
    }
},
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
    "required": [
        "query"
    ],
    "properties": {
    "query": {
        "type": "string",
        "description": "A straight to the point concise succint question or search query to be sent to research assistant",
    }
    },
    "additionalProperties": False
    },
    "strict": True
    }
},
{
    "type": "function",
    "function": {
    "name": "call_openai_gpt40_llm",
    "description": (
            "Use this to call OpenAI GPT-4o LLM for your prompt"
        ),         
    "parameters": {
    "type": "object",
    "required": [
        "prompt"
    ],
    "properties": {
    "prompt": {
        "type": "string",
        "description": "Your prompt for GPT-4o LLM",
    }
    },
    "additionalProperties": False
    },
    "strict": True
    }
},
]

def convert_invalid_json_to_valid(input_str):
    # Remove markdown code block delimiters using regex
    input_str = re.sub(r'```json\s*', '', input_str, flags=re.IGNORECASE)
    input_str = re.sub(r'```\s*', '', input_str)

    # Trim any remaining leading/trailing whitespace
    input_str = input_str.strip()
    
    # Fix issues with missing braces and colons
    try:
        # Repair structure: ensure that "Critical_Evaluation" is enclosed properly
        # Check if the input doesn't already start and end with appropriate braces
        if not input_str.startswith('{'):
            input_str = '{' + input_str
        
        if not input_str.endswith('}'):
            input_str = input_str + '}'
        
        # Correct the structure by replacing misplaced or missing colons/commas
        input_str = re.sub(r'"Critical_Evaluation":\s*', '"Critical_Evaluation": {', input_str, count=1)

        input_str = input_str + '}'

        # Debug: print statements for checking the sanitized string form
        # print("Corrected JSON String:", input_str)

        print(f"this is the reformatted json\n\n{input_str}\n\n")
        # Load the JSON data
        data = json.loads(input_str)

        return json.dumps(data, indent=4)

    except json.JSONDecodeError as e:
        return f"Error decoding JSON: {e}"
    
def parse_rating_response(response_data, threshold: float):

    print(f"parse_rating_response\n\n{response_data}\n\n")
    try:
        json_data = ""
        if not '\n' in response_data:
            json_data = convert_invalid_json_to_valid(response_data)   
        else:
            lines = response_data.splitlines()
            json_data = "\n".join(line for line in lines if not line.strip().startswith('```'))

        print(f"Loading this json data\n\n{json_data}\n\n")

        data = json.loads(json_data)
        if 'Critical_Evaluation' in data:
            evaluation = data['Critical_Evaluation']
            if all(key in evaluation for key in ['Pros', 'Cons', 'Rating']):
                try:
                    # Attempt to convert the rating to a float
                    rating = float(evaluation['Rating'])
                    add_score(rating)
                    return rating >= threshold
                except (ValueError, TypeError) as e:
                    print("FAILED parse_rating_response: ", e)
                    pass
    except json.JSONDecodeError:
        print("FAILED json.JSONDecodeError parse_rating_response")
        pass
    return False

def call_research_professional(question: str, model_version: str = "gpt-4o-mini") -> str:

    refactor_count = 1

    prompt = f"""
Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. You will generate increasingly better thoughts, solutions, and critical feedback to the user's question. Repeat the following 4 steps {MAX_ITERATIONS_AND_TOOLS} times. 

**Rules:**
In your thoughts, solutions and critical feedback you must use Chain of Thought, thinking step by step, and thinking and logic assurance phrases like "I need to analyze this", "Now, let's break it down", "What's important here is", "Let's consider all possibilities", "I'll work through this logically", "It's crucial to understand", "Reasoning through this", "Let’s explore our options", "I must evaluate the situation", "This means we'll", "Our objective here is", "To tackle this challenge", "Let's examine the details", "This implies that", "In other words", "I need to weigh the pros and cons", "From what I understand", "I should double-check the facts", "I'll need to compare this with", "Clearly, I see that", "The real issue is", "Let's focus on the key points", "I'll approach this methodically", "It’s important to recognize", "I should assess the variables", "We’ll look at it step by step", "I'll need to identify the problem", "Another perspective might be", "To clarify", "I have to determine", "It's evident that", "Our main concern is", "Let's outline our approach", "We have to ensure that", "I'll need to formulate a plan", "Essentially, this suggests", "I'll conclude that", "What we need to consider is", "By exploring possibilities", "I ought to think through the implications", "Evaluating the outcomes", "It makes sense when I", "This leads us to", "I'll infer from", "I must establish a framework", "We will resolve this by", "The hypothesis is", "We can deduce that", etc...  You must use like phrases as part of your process reasoning your rationale. 

**Guidelines:**
Step 1. In the Thought section, detail your reasoning process using the specified format: <|begin_of_thought|> (individual thought with detailed steps separated with ' ') <|end_of_thought|> Each step should include detailed considerations such as analyzing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps.
Step 2. In the Solution section, based on diverse attempts, explorations, thoughts, critical feedback, and reflections from the past, systematically present a solution that you deem correct with actionable steps and details the user can take as a resolution. IMPORTANT: In this section you must focus on solving with specific answers to the user's question. It's not good enough to generalize the answers must be specific and aligned to the user's question. Provide sufficient metric-based guidance or resources for implementation. The solution should remain a logical, accurate, concise expression with detail of necessary steps needed to reach the conclusion, formatted as follows: <|begin_of_solution|> (formatted, precise, and clear solution) <|end_of_solution|> 
Step 3. In the Critical Feedback section, explore questions the user should have asked but didn't know to ask about the domain, class, and task. Critically explore the feedback and how it encourages or detracts exploring another related thought. Detail your rationale process with this specified format: <|critical_evaluation|> (Detailed critic items separated with ' ') <|end_of_critical_evaluation|> 
Step 4. Keep a running list of all the tool calls you could have used to enhance your reasoning. Detail your rationale process with this specified format: <|tool_requests|> (Detail one comprehesive multi-part question you would ask of a research assistant to help in your reasoning process' ') <|end_of_tool_requests|> 

Response only in JSON. The JSON should be a list (length {MAX_ITERATIONS_AND_TOOLS}) of dictionaries whose keys are "Thought", "Solution",  "Critical_Feedback", and "Tool_Calls"

``` user question
{question}
```
"""

    messages = [
        {"role": 'user', 'content': get_current_datetime() + '\n\n' + prompt},
    ]

    score_misses = 0
    is_final_answer = False

    # Main loop for back-and-forth with the model
    for _ in range(MAX_LOOPS):

        # For debugging/logging
        print("~" * 80)
        print("\nMessage Stack:\n")
        try:
            print(json.dumps(messages[-1], indent=4))
        except:
            try:
                json_obj = json.loads(messages[-1])
                print(json.dumps(json_obj, indent=4))
            except:
                print(str(messages[-1]).replace("{'role'", "\n\n\n{'role'"))
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

        # Log to a file
        try:
            with open('o1reasoning-intermediate.txt', 'a') as output_file:
                if finish_reason is not None:
                    output_file.write(f"{finish_reason}\n")
                if assistant_content is not None:
                    output_file.write(f"{assistant_content}\n")
                if tool_calls:
                    output_file.write(f"{tool_calls}\n")
                output_file.write("=" * 80 + "\n" + "=" * 80 + "\n")
        except IOError:
            print("An error occurred while writing to the file.")

        # If there are tool calls, handle them
        if tool_calls:
            for tc in tool_calls:
                func_name = tc["function"]["name"] if isinstance(tc, dict) else tc.function.name
                arguments_json = tc["function"]["arguments"] if isinstance(tc, dict) else tc.function.arguments

                print("^" * 80)
                print(f"Tool name: {func_name}\nArguments: {arguments_json}\n")
                print("^" * 80 + "\n")

                # Log the tool result
                try:
                    with open('o1reasoning-intermediate.txt', 'a') as output_file:
                        output_file.write("^" * 80)
                        output_file.write(f"Tool name: {func_name}\nArguments: {arguments_json}\n")
                        output_file.write("^" * 80 + "\n")                        

                except IOError:
                    print("An error occurred while writing to the file.")
                    
                # Attempt to parse arguments JSON
                try:
                    arguments = json.loads(arguments_json)
                except json.JSONDecodeError:
                    arguments = {}

                # Dispatch to the correct tool
                if func_name == "call_research_assistant":
                    query = arguments.get("query", "")
                    result = call_research_assistant(query)
                elif func_name == "calculate":
                    expression = arguments.get("expression", "")
                    precision = arguments.get("precision", 0)
                    calc = Calculator()
                    result = calc.calculate(expression, precision)
                elif func_name == "call_openai_gpt40_llm":
                    prompt_gpt4o = arguments.get("prompt","")
                    result = call_openai_gpt40_llm(prompt_gpt4o)
                else:
                    result = f"Tool {func_name} is not implemented."

                tool_result_message = {'role': "tool", 'content': result}

                possible_id = getattr(tc, "id", None)
                if possible_id:
                    tool_result_message["tool_call_id"] = possible_id

                messages.append(tool_result_message)

                # Log the tool result
                try:
                    with open('o1reasoning-intermediate.txt', 'a') as output_file:
                        output_file.write(f"* * *  Research Assistant Response  * * *\n\n")
                        output_file.write(f"{result}\n")
                        output_file.write("=" * 80 + "\n")
                except IOError:
                    print("An error occurred while writing to the file.")

            print(f"\n\n\nREFACTOR_COUNT {refactor_count} {refactor_count % 3}\n\n\n")

            #if refactor_count % 3 == 2:
            promptx = """
Critically evaluate this reasoning process against the user's initial request 
and provide a list of both pros / cons statements regarding the reasoning process 
and rating as to how well the process has gathered enough information to answer 
the user's initial request betweed 0.0 and 1.0. With 1.0 being the highest score.

Respond only in JSON following the example template below.

```json
{
    "Critical_Evaluation": {
        "Pros": [
        ],
        "Cons": [
        ],
        "Rating": 0.0
    }
}
```
"""
            messages.append({'role': 'user', 'content': promptx})

            # Log the tool result
            try:
                with open('o1reasoning-intermediate.txt', 'a') as output_file:
                    output_file.write(f"{promptx}\n")
                    output_file.write("=" * 80 + "\n")
            except IOError:
                print("An error occurred while writing to the file.")

            refactor_count = refactor_count + 1
            continue

        # If no tool calls, check finish_reason
        if finish_reason == "stop":

            print(f"\n\n\nREFACTOR_COUNT {refactor_count} {refactor_count % 3}\n\n\n")

            if is_final_answer:
                # The model gave a final answer
                if assistant_content:
                    promptz = f"""
Rewrite this content in to a well structured article with formatting.

{assistant_content}
"""
                    #assistant_content = call_openai_gpt40_llm(promptz)

                    print("\nAssistant:\n" + assistant_content)
                    return assistant_content
                else:
                    print("\nAssistant provided no content.")
                break          

            if refactor_count % 3 == 1:
                promptx = """
You now have the opportunity to run up to 5 tools to assist your response.
"""
                messages.append({'role': 'user', 'content': promptx})

                # Log the tool result
                try:
                    with open('o1reasoning-intermediate.txt', 'a') as output_file:
                        output_file.write(f"{promptx}\n")
                        output_file.write("=" * 80 + "\n")
                except IOError:
                    print("An error occurred while writing to the file.")

            elif refactor_count % 3 == 0:

                is_pass_threshold = parse_rating_response(assistant_content, PASS_THRESHOLD)

                print(f"\n\n\nPASSED THRESHOLD {PASS_THRESHOLD} {is_pass_threshold}\n\n\n")

                if is_pass_threshold or score_misses >= MAX_SCORE_MISSES:
                    promptx = """
You are an highly successful investigative reporter. 

IMPORTANT: Response must answer all aspects of the user's question 
even those the user did not know to ask because they were unfamiliar 
with the domain, class, and task at hand. Cover all aspects of the 
who, what, when, where, why and how without specifically mentioning 
those words as sections or titles. It's not good enough to generalize 
as the answers must be specific and aligned to the user's question. 
Provide sufficient detailed metric-based guidance when responding. 
Response must not be json but text as if writing an research paper or 
investigative article.

With a focus on the user's question and considering the entire 
conversation present your well structured report with the detailed analysis 
and comprehensive solution with all any details from the conversation. 
"""
                    messages.append({'role': 'user','content': promptx,})

                    is_final_answer = True

                    # Log the tool result
                    try:
                        with open('o1reasoning-intermediate.txt', 'a') as output_file:
                            output_file.write(f"{promptx}\n")
                            output_file.write("=" * 80 + "\n")
                    except IOError:
                        print("An error occurred while writing to the file.")
                else:
                    score_misses = score_misses + 1

                    scores_text = "\nScores:" + ", ".join(map(str, scores))

                    promptx = f"""
You are a highly successful people manager with all company resources 
at your disposal. Your employee is performing the following task and 
has received the following scores and feedback. Response must include 
your best motivational speech to the employee to substantially increase 
their score on the task. Provide incentives for good performance and 
discourage poor performance through constructive feedback or consequences. 
Respond as if you are talking to them directly without mentioning their name.

``` task
{question}
```

``` Iterative scores in order based on the initial draft and the latest version
{scores_text}
```

``` feedback
{assistant_content}
```
"""
                    manager_feedback = call_openai_gpt40_llm(promptx)

                    promptx = f"""
Your scores so far on how well you have done are {scores_text}.
The boss has asked that I send words of encouragement to get the score higher.
Here is what the boss said.

``` boss statements
{manager_feedback}
```

Address the cons of the solution and go another {MAX_ITERATIONS_AND_TOOLS} iterations of reasoning process.
This should be a focus on thoughts, solutions, and potential tool calls. 
But don't make tool calls that will come later.
"""
                    messages.append({'role': 'user', 'content': promptx})

                    # Log the tool result
                    try:
                        with open('o1reasoning-intermediate.txt', 'a') as output_file:
                            output_file.write(f"{promptx}\n")
                            output_file.write("=" * 80 + "\n")
                    except IOError:
                        print("An error occurred while writing to the file.")                    

            refactor_count = refactor_count + 1
            continue

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

    with open('o1reasoning-intermediate.txt', 'w') as output_file:
        output_file.write("")

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
