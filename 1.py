# %% [markdown]
# # Building Agents with AutoGen
# 
# This notebook introduces **Microsoft AutoGen**, a framework for building multi-agent conversations powered by Large Language Models (LLMs). You will learn how to configure an LLM, create conversable agents, and run simple question-answer and code-execution tasks.
# 
# ---
# 
# ## Learning objectives
# 
# By the end of this notebook you will be able to:
# 
# - **Configure an LLM** via a config list (e.g., from a JSON file or environment variables).
# - **Create an AssistantAgent** that uses the LLM to generate responses.
# - **Create a UserProxyAgent** that represents the user and can optionally execute code.
# - **Initiate a chat** between the user proxy and the assistant to complete a task.
# 
# ---
# 
# ## Basic workflow (four steps)
# 
# 1. **Define configuration list** — Specify which model(s) and API keys the agent will use.
# 2. **Define the assistant** — An LLM-backed agent that reasons and responds (and can suggest code).
# 3. **Define the user proxy** — Represents the human user; can run code and relay results back.
# 4. **Initiate chat** — Start a conversation with a task message and let the agents collaborate.

# %%
# ---------------------------------------------------------------------------
# Optional: Install AutoGen in your environment (run once per environment)
# ---------------------------------------------------------------------------
# Uncomment the line below if autogen is not already installed.
# pip install autogen-agentchat==0.2.38

# %%
# ---------------------------------------------------------------------------
# Imports: Core AutoGen components and utilities
# ---------------------------------------------------------------------------
import autogen
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
from autogen.coding import LocalCommandLineCodeExecutor
from pathlib import Path
from IPython.display import Image, Markdown

# AssistantAgent: LLM-powered agent that generates replies (and can suggest code).
# UserProxyAgent: Represents the user; can execute code and return results.
# config_list_from_json: Loads LLM config (model, API key) from a JSON file or env var.
# LocalCommandLineCodeExecutor: Runs code in a local folder (used later for code execution).

# Verify the installed AutoGen version (helps match tutorial and docs).
autogen.__version__

# %% [markdown]
# ### Step 1: LLM configuration
# 
# AutoGen needs to know **which LLM to use** and **how to authenticate**. This is done via a **config list**: a list of model configurations (model name, API key, and optional settings).
# 
# **Recommended approach:** Use a `CONFIG_LIST.json` file (or an environment variable holding the same JSON) so that API keys stay out of your notebook and out of version control.
# 
# - **Reference:** [LLM configuration (AutoGen docs)](https://microsoft.github.io/autogen/0.2/docs/topics/llm_configuration/)
# 
# **Example `CONFIG_LIST.json` format:**
# 
# ```json
# [
#   {
#     "model": "gpt-4o-mini",
#     "api_key": "<your-key-here>"
#   }
# ]
# ```
# 
# For Groq, you would use something like `"model": "llama-3.3-70b-versatile"` and `"api_type": "groq"`, with the key from your environment.
# 
# **Security:** Keep API keys out of git (e.g., add `CONFIG_LIST.json` to `.gitignore`) and rotate them regularly.

# %%
# ---------------------------------------------------------------------------
# Load environment variables from a .env file (optional but recommended)
# ---------------------------------------------------------------------------
# This allows you to store API keys in .env (e.g. OPENAI_API_KEY, GROQ_API_KEY)
# and reference them in CONFIG_LIST.json via env vars, keeping secrets out of code.
from dotenv import load_dotenv
import os

load_dotenv()  # Loads .env from the current directory (or parent directories)

# %%
# ---------------------------------------------------------------------------
# Steps 2–4: Load config, create assistant and user proxy, then initiate chat
# ---------------------------------------------------------------------------

# Step 2a: Load model configuration from CONFIG_LIST.json (or from env var of same name).
# Keep this file out of source control; use .env or env vars for the actual API key.
config_list = config_list_from_json(env_or_file="CONFIG_LIST.json")

# Step 2b: Create the AssistantAgent — the LLM-backed agent that will answer questions.
# - name: Used in conversation logs.
# - system_message: Instructions that shape the assistant’s behavior.
# - llm_config: Tells the agent which model(s) to use (from config_list).
assistant = AssistantAgent(
    name="groq_assistant",
    system_message="You are a helpful AI assistant.",
    llm_config={"config_list": config_list},
    human_input_mode="NEVER"
)

# Step 3: Create the UserProxyAgent — represents the human user in the conversation.
# - code_execution_config=False: This example is Q&A only; no code will be run.
# - human_input_mode="NEVER": no prompts; proxy auto-replies so the chat runs unattended.
# - max_consecutive_auto_reply=1: stop after one assistant reply (avoids endless back-and-forth).
user_proxy = UserProxyAgent(
    name="user_proxy",
    code_execution_config=False,
    human_input_mode="NEVER",
    max_consecutive_auto_reply=1,
)

# Step 4: Start the conversation. The user_proxy sends the first message to the assistant.
# The assistant replies; in this setup you may be prompted for input (e.g. type 'exit' to stop).
user_proxy.initiate_chat(
    assistant,
    message="What are the key benefits of using Groq for AI apps?")

# %%
# ---------------------------------------------------------------------------
# Alternative: Build config list in code (instead of CONFIG_LIST.json)
# ---------------------------------------------------------------------------
# Use this when you prefer to pass the API key from the environment (e.g. .env).
# Validate GROQ_API_KEY so we fail fast with a clear message instead of at API call time.
api_key = os.environ.get("GROQ_API_KEY")
if not api_key or not api_key.strip():
    raise ValueError(
        "GROQ_API_KEY is not set or is empty. "
        "Set it in your environment or in a .env file (e.g. GROQ_API_KEY=gsk_...). "
        "Then re-run the cell that runs load_dotenv() and this cell."
    )

config_list = [{
    "model": "llama-3.3-70b-versatile",
    "api_key": api_key,
    "api_type": "groq"
}]

# Create assistant and user proxy (same as the main example above).
assistant = AssistantAgent(
    name="groq_assistant",
    system_message="You are a helpful AI assistant.",
    llm_config={"config_list": config_list}
)

# human_input_mode="NEVER": no prompts for input; proxy auto-replies so the chat runs unattended.
# max_consecutive_auto_reply: cap auto-replies so the conversation stops after the assistant replies.
user_proxy = UserProxyAgent(
    name="user_proxy",
    code_execution_config=False,
    human_input_mode="NEVER",
    max_consecutive_auto_reply=1,
)

# Initiate the same demo chat.
user_proxy.initiate_chat(
    assistant,
    message="What are the key benefits of using Groq for AI apps?")

# %% [markdown]
# ---
# ## Code execution: assistant + user proxy with an executor
# 
# In the next section we switch to a **code-execution** setup: the assistant can suggest code, and the user proxy will run it in a local folder and return the results. This enables tasks like plotting charts or writing files.

# %% [markdown]
# The sample `CONFIG_LIST.json` format is shown in the **LLM configuration** section above. Ensure your JSON file exists in the notebook directory (or that the corresponding environment variable is set) before loading the config.

# %% [markdown]
# ### Build the assistant and user proxy (with code execution)
# 
# Here we create:
# 
# 1. **AssistantAgent** — Same as before, but it will now suggest **code** to accomplish tasks (e.g., plotting, file I/O).
# 2. **UserProxyAgent** — Configured with a **code executor** so it can run the assistant’s code in a local directory and send the output (or errors) back. The conversation continues until the task is done or you stop it.

# %%
# ---------------------------------------------------------------------------
# Create assistant and user proxy with code execution enabled
# ---------------------------------------------------------------------------
# Assistant: same as before; it will now propose code to solve tasks.
assistant = AssistantAgent("assistant", llm_config={"config_list": config_list})

# Executor: runs code in the "coding" subdirectory. All generated scripts
# (e.g. .py files) and outputs (e.g. .png) will appear there.
work_dir = Path("coding")
code_executor = LocalCommandLineCodeExecutor(work_dir=work_dir)

# User proxy: when the assistant sends code, the proxy runs it via code_executor
# and returns stdout/stderr/exit code to the assistant for the next turn.
user_proxy = UserProxyAgent("user_proxy", code_execution_config={"executor": code_executor})

# %%
# ---------------------------------------------------------------------------
# Task 1: Stock chart — assistant writes code, user proxy runs it
# ---------------------------------------------------------------------------
# Wrapper to send a natural-language task to the assistant. The user proxy
# starts the chat; the assistant may respond with code; the proxy executes it
# and sends results back until the task is done (or you type 'exit').
def execute_agent(prompt: str):
    """Send a task to the assistant via the user proxy."""
    return user_proxy.initiate_chat(assistant, message=prompt)

# Example 1: Data + plotting. The assistant will typically suggest a Python script
# (e.g. using yfinance + matplotlib), the proxy runs it in the "coding" folder,
# and the plot is saved as stock_price_change.png there.
execute_agent(
    """Plot a chart of Tesla stock price change and compare it to Nvidia's
    stock price change in the same period and save it as stock_price_change.png"""
)

# %%
# ---------------------------------------------------------------------------
# Display the chart produced by the agent (saved in the "coding" work directory)
# ---------------------------------------------------------------------------
Image("coding/stock_price_change.png")

# %% [markdown]
# ### Task 2: Research report generation
# 
# The same assistant + user proxy can handle more involved tasks. In this example, the agent is asked to fetch papers (e.g. from arXiv), summarize them, and write a single Markdown report. The conversation may take several turns (install deps, run scripts, retry on errors). When running in a notebook you may need to press Enter to allow auto-reply or type `exit` to stop.

# %%
# ---------------------------------------------------------------------------
# Task 2: Research report — fetch papers, summarize, write one Markdown file
# ---------------------------------------------------------------------------
# The assistant may use arXiv API, write Python scripts to fetch/summarize,
# and iterate if dependencies are missing or execution fails. Can take several minutes.
execute_agent(
    """Fetch 2 papers about using small language models and
    summarize them into a one single research report file named research-report-llms-productivity.md"""
)

# %%
# ---------------------------------------------------------------------------
# Task 2: Research report — fetch papers, summarize, write one Markdown file
# ---------------------------------------------------------------------------
# The assistant may use arXiv API, write Python scripts to fetch/summarize,
# and iterate if dependencies are missing or execution fails. Can take several minutes.
execute_agent(
    """Fetch 3 papers about using large language models to augment human productivity and
    summarize them into a one single research report file named research-report-llms-productivity.md"""
)

# %%
# ---------------------------------------------------------------------------
# Display the generated research report (run after Task 2 completes)
# ---------------------------------------------------------------------------
# The report is saved in the "coding" folder; we read it and render as Markdown.
with open("coding/research-report-llms-productivity.md") as f:
    report = f.read()

Markdown(report)

# %% [markdown]
# ---
# 
# ## Summary
# 
# You have seen the core AutoGen workflow:
# 
# | Step | What you did |
# |------|----------------|
# | **Config** | Loaded an LLM config from `CONFIG_LIST.json` (or built it in code). |
# | **Assistant** | Created an `AssistantAgent` with a system message and `llm_config`. |
# | **User proxy** | Created a `UserProxyAgent`; with `code_execution_config` it can run code in a folder. |
# | **Chat** | Used `user_proxy.initiate_chat(assistant, message=...)` to run Q&A or code-based tasks. |
# 
# **Next steps:** Explore multi-agent patterns (e.g. group chat), custom tools, or different executors. See the [AutoGen documentation](https://microsoft.github.io/autogen/) for more.


