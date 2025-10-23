from deepagents import create_deep_agent
from dotenv import load_dotenv
from textwrap import dedent
from langchain.chat_models import init_chat_model
from tools.internet_search import InternetSearchTool
from utils.subagent_factory import make_all_subagents

RESEARCH_TEMPLATE = '''\
You are an expert researcher. Your job is to conduct thorough research and
then write a polished report.

You can delegate to the following agents as needed: 
    {subagent_descriptions}
'''

def main():
    load_dotenv()

    params = {
        "topic": "AI in healthcare",
        "audience": "policy makers",
        "word_limit": 120,
    }
    subagents = make_all_subagents(params, "./subagents.yaml")
    subagent_descriptions = [sa["name"] + ": " + sa["description"] for sa in subagents]

    # System prompt to steer the agent to be an expert researcher
    research_instructions = RESEARCH_TEMPLATE.format(subagent_descriptions="\n    ".join(subagent_descriptions))

    print(research_instructions)

    model = init_chat_model(
        model="openai:gpt-5",
    )

    agent = create_deep_agent(
        model=model,
        tools=[],
        subagents=subagents,
        system_prompt=research_instructions
    )

    result = agent.invoke({"messages": [{"role": "user", "content": "Write a brief (120 words) report on AI in healthcare."}]})

    # Print the agent's response
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
