from dotenv import load_dotenv
from os import getenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import BraveSearchLoader
from pydantic import BaseModel, Field
from typing import List

load_dotenv()


class Source(BaseModel):
    """
    Args:
        BaseModel (_type_): _description_
    """

    name: str = Field(..., description="The name of the source")
    url: str = Field(..., description="The URL of the source")


class AgentResponse(BaseModel):
    """
    Args:
        BaseModel (_type_): _description_
    """

    answer: str = Field(..., description="The answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="The sources used to answer the query"
    )


@tool
def search(query: str) -> str:
    """
    Tool that searches over internet
    Args:
        query: The query to search for
    Returns:
        The search result
    """
    print(f"Searching for: {query}")
    loader = BraveSearchLoader(query=query, api_key=getenv("BRAVE_SEARCH_API_KEY"))
    results = loader.load()
    return "\n".join([result.page_content for result in results])


llm = ChatOllama(model="ministral-3:latest")
tools = [search]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Hello from langchian!")
    result = agent.invoke(
        {
            "messages": HumanMessage(
                content="search for 3 job postings for an ai engineer using langchain in Berlin and list their details"
            )
        }
    )
    print(result)


if __name__ == "__main__":
    main()
