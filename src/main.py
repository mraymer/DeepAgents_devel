from deepagents import create_deep_agent
from dotenv import load_dotenv
from textwrap import dedent
from datetime import datetime
from langchain.chat_models import init_chat_model
from tools.internet_search import InternetSearchTool
from tools.venue_lookup import VenueLookupTool
from langchain_tavily import TavilyExtract

from utils.agent_factory import make_all_subagents

COORDINATOR_BACKGROUND = '''\
You are an experienced managing editor.  You are skilled at weighing evidence
and drawing measured conclusions.  You are frank and honest about your confidence
in your conclusions.
'''

COORDINATOR_TASK = '''\
Write a demographic report about {author}, who publishes editorial pieces for {venue}.
Find each of the following pieces of information about the author: {bio_fields}
In determining the author's age, note that the current year is {current_year}.
For each piece of information, provide evidence in the form of information that supports
your conclusion, and a rationale for how you deduced the demographic information from
the evidence.
If no information on full name is found, use the name: {author} and confidence "medium"
If the gender of the author is not found, take your best guess based on the author's name and report it
as low or medium confidence, depending on how certain you are of your guess.  DO NOT report "Not found" for
gender.
If the location of residence is not found, look up the location of the venue
where they publish ({venue}) and report it as medium confidence.
DO NOT report "Not found" for location of current residence.
If ethnicity is not found, you may use any ethnicity associated with their publishing venue.
If you do not know these details about the venue, you should look them up.
If ANY OTHER item is missing, enter "Not found" for that item.  For each item, include your
confidence that the reported data is correct, based on all the evidence you have available.
Confidence should ALWAYS be high, medium, or low, unless value is "Not found", in which case
confidence should be "N/A".

You can delegate to the following agents, who can provide more information:
    {subagent_descriptions}
'''

COORDINATOR_FORMAT_INSTRUCTIONS ='''
Return exactly one JSON object. The keys should be: {json_keys}. For each key, the value should be another
JSON object with the following keys: "value", "evidence_source", "evidence_excerpt", "rationale", "confidence".
The "value" field should contain the value for that demographic item (or "Not found" if applicable).
The "evidence_source" field should contain the URL, dossier, or tool where the information was found (or "Not found" if applicable).
The "evidence_excerpt" field should contain a brief (up to a few lines) excerpt from the source including the text supporting your finding
(or "Not found" if applicable). Do not prepend any explanation text. Do not wrap in code-fences. Output no extra keys.
Do not wrap the JSON in quotes or any other characters. Ensure the output is valid JSON.

Example output:
{{
    "full_name": {{
    "value": "Jane Doe",
    "evidence_source": "https://example.com/jane-doe-bio",
    "evidence_excerpt": "Jane Doe is a renowned journalist known for her work in...",
    "rationale": "The excerpt clearly states the full name of the author.",
    "confidence": "high"
    }},
    "gender": {{
    "value": "Female",
    "evidence_source": "https://example.com/jane-doe-interview",
    "evidence_excerpt": "In a recent interview, Jane Doe discussed her experiences as a mother...",
    "rationale": "The excerpt uses 'mother', indicating female gender.",
    "confidence": "medium"
    }},
    ...
}} 
'''

bio_fields = [
            'Full name',
            'Gender',
            'Ethnicity',
            'Birth location',
            'Location of current residence',
            'Age'
        ],

json_keys = [
            'full_name',
            'gender',
            'ethnicity',
            'birth_location',
            'current_residence',
            'age'
        ],

current_year = datetime.now().year

def main():
    load_dotenv()

    params = {
        "author": "Michael Raymer",
        "venue": "werken.cl",
        "bio_url": "https://birg.cs.wright.edu/assets/RaymerCV.html",
        "bio_fields": bio_fields,
        "json_keys": json_keys,
        "current_year": current_year
    }

    subagents = make_all_subagents(params, "./config/subagents.yaml")
    subagent_descriptions = '\n    '.join([sa["name"] + ": " + sa["description"] for sa in subagents])

    # System prompt to steer the agent to be an expert researcher
    system_template = COORDINATOR_BACKGROUND
    system_prompt = system_template.format(**params, subagent_descriptions=subagent_descriptions)
    user_template = COORDINATOR_TASK + COORDINATOR_FORMAT_INSTRUCTIONS
    user_prompt = user_template.format(**params, subagent_descriptions=subagent_descriptions)

    model = init_chat_model(
        model="openai:gpt-4o-mini",
    )

    agent = create_deep_agent(
        model=model,
        tools=[VenueLookupTool()],
        subagents=subagents,
        system_prompt=system_prompt
    )

    result = agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})

    # Print the agent's response
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
