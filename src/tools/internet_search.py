from typing import Literal, Type
from langchain.tools import BaseTool
from tavily import TavilyClient
from dotenv import load_dotenv
from pydantic import BaseModel, Field, PrivateAttr
import os

class InternetSearchToolInputs(BaseModel):
    '''Inputs for the Internet Search Tool'''
    query: str = Field(..., description="The search query string.")
    max_results: int = Field(5, description="Maximum number of results to return.")
    topic: Literal["general", "news", "finance"] = Field('general', description="The topic of the search.")
    include_raw_content: bool = Field(False, description="Whether to include raw content in the results.")


class InternetSearchTool(BaseTool):
    name: str = "internet_search"
    description: str = (
        "Use this to run an internet search for a given query. You can specify "
        "the max number of results to return, the topic, and whether raw content "
        "should be included."
    )
    args_schema: Type[BaseModel] = InternetSearchToolInputs

    _client: TavilyClient = PrivateAttr()

    def __init__(self):
        super().__init__()
        load_dotenv()
        self._client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    def _run(
        self,
        query: str,
        max_results: int = 5,
        topic: Literal["general", "news", "finance"] = "general",
        include_raw_content: bool = False,
    ) -> str:
        """Run a web search"""
        print("Searching internet, query: " + query)
        return self._client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        )

