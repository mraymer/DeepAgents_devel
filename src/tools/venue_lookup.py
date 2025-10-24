from langchain.tools import BaseTool
from typing import Sequence, Type, List
from pydantic import BaseModel, Field
from urllib.parse import urlparse
from functools import lru_cache
import json

class VenueLookupToolInput(BaseModel):
    '''Input schema for country lookup'''
    url: str = Field(..., description="The base url of the publisher to look up.")

class VenueLookupTool(BaseTool):
    '''Lookup publication venue details given a URL.'''
    name: str = "venue_lookup"
    description: str = (
        "Useful for information on a publication venue given its URL."
    )
    args_schema: Type[BaseModel] = VenueLookupToolInput

    sources: dict = None

    def _load_sources(self, json_path: str = 'sources.json'):
        '''Load and cache the sources.json file, decoding UTF-8 and Unicode escapes.'''
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                sources_list = json.load(f)
                self.sources = {}
            for source in sources_list:
                domain = self._normalize_url(source['root_url'])
                self.sources[domain] = source
        except Exception as e:
            raise RuntimeError(f'Error reading {json_path}: {e}')

    def _normalize_url(self, u: str) -> str:
        '''Normalize a URL for comparison (strip www., trailing slashes, etc.).'''
        u = u.lower()
        if u.startswith('http://') or u.startswith('https://'):
            u = urlparse(u).netloc
        if u.startswith('www.'):
            u = u[4:]
        return u.rstrip('/')

    def _run(self, url: str) -> str:
        '''
        Given a URL, return the associated country of publication from sources.json.
        Uses cached JSON data and handles Unicode and URL variations.
        '''
        print("Looking up venue: " + url)

        # TODO:  Brittle:  clean this up
        json_path = './config/sources.json'
        if self.sources is None:
            self._load_sources(json_path)
        input_domain = self._normalize_url(url)

        # If the input URL domain matches or contains the root domain
        if input_domain in self.sources:
            return self.sources[input_domain]
        else:
            return 'Unknown'
       