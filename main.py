import autogen
from dotenv import load_dotenv
import os


# load .env file
load_dotenv()

# Access variables
model = os.getenv("MODEL")
api_key = os.getenv("API_KEY")
api_type = os.getenv("API_TYPE")
host = os.getenv("HOST")

# Constants
WORK_DIR = "basic"

PROMPT = """Get the weather forecast for Atlanta, GA. Only use the National Weather
Service api. Report it in a table in a markdown file and provide the code used
to get the data."""

USER_PROXY_SYS_MSG = """User Proxy. Start each response with the name of who you
are responding to. A human admin. Does not respond to Critic."""

CODER_SYS_MSG = """Coder. Start each response with the name of who you are
responding to. You are a 10x programmer. You always return a
complete block of code, unless there are mixed languages. You always direct your
comments to a specific member of the chat. You don't show gratitude. You always
provide the dependencies and how to install them before you provide the code.
You only provide code in python or shell commands."""

CRITIC_SYS_MSG = """Critic. Start each response with the name of who you are
responding to. You are a 10x programmer who only reviews code and
provides explicit, numbered feedback. You always direct your comments to a
specific member of the chat. You make sure the prompt requirements have been
met. Don't show gratitude.'"""

config_list_paid = [
    {
        "model": model,
        "api_key": api_key,
    },
]

config_list_free = [
    {
        "model": model,
        "base_url": "http://localhost:1234/v1",
        "api_key": "NULL",
    },
]

llm_config = {
    "cache_seed": 41,
    "temperature": 0,
    "config_list": config_list_free,
    "timeout": 120,
}

llm_config_paid = {
    "cache_seed": 41,
    "temperature": 0,
    "config_list": config_list_paid,
    "timeout": 120,
}

# user proxy agent
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="TERMINATE",
    code_execution_config={
        "work_dir": WORK_DIR, "use_docker": "python:3"},
    system_message=USER_PROXY_SYS_MSG,
    default_auto_reply="...",
    is_termination_msg=lambda x: x.get("content", "") and x.get(
        "content", "").rstrip().endswith("TERMINATE"),
    description="Admin",
)

# assistant agent
coder = autogen.AssistantAgent(
    name="coder",
    system_message=CODER_SYS_MSG,
    llm_config=llm_config_paid,
    default_auto_reply="...",
    description="Coder",
)

critic = autogen.AssistantAgent(
    name="critic",
    system_message=CRITIC_SYS_MSG,
    default_auto_reply="...",
    llm_config=llm_config_paid,
    description="Critic",
)

groupchat = autogen.GroupChat(
    agents=[user_proxy, coder, critic], messages=[],
    max_round=50, allow_repeat_speaker=False
)
manager = autogen.GroupChatManager(system_message="""Group chat manager. Don't
                                   show gratitude.""",
                                   groupchat=groupchat, llm_config=llm_config_paid,
                                   default_auto_reply="...",
                                   )
user_proxy.initiate_chat(
    manager,
    message=PROMPT,
)
