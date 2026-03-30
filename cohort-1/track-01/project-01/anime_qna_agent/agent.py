import os
import logging
import google.cloud.logging
from dotenv import load_dotenv

from google.adk import Agent
from google.adk.agents import SequentialAgent
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.langchain_tool import LangchainTool

from langchain_community.tools import WikipediaQueryRun, GoogleSearchRun
from langchain_community.utilities import WikipediaAPIWrapper, GoogleSearchAPIWrapper

import requests

# --- Setup Logging and Environment ---

try:
    cloud_logging_client = google.cloud.logging.Client()
    cloud_logging_client.setup_logging()
except Exception:
    logging.basicConfig(level=logging.INFO)
    logging.info("Running locally — using standard logging.")

load_dotenv()

model_name = os.getenv("MODEL")

# --- Custom Tools ---

def add_prompt_to_state(
    tool_context: ToolContext, prompt: str
) -> dict[str, str]:
    """Saves the user's anime question to the state."""
    tool_context.state["PROMPT"] = prompt
    logging.info(f"[State updated] Added to PROMPT: {prompt}")
    return {"status": "success"}


def search_anime_fandom(query: str) -> str:
    """Searches the Anime Fandom wiki for information about anime series, characters, episodes, and lore."""
    try:
        url = "https://anime.fandom.com/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 3,
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("query", {}).get("search", [])

        if not results:
            return "No results found on Anime Fandom wiki."

        output = []
        for r in results:
            title = r.get("title", "")
            snippet = (
                r.get("snippet", "")
                .replace('<span class="searchmatch">', "")
                .replace("</span>", "")
            )
            output.append(f"Title: {title}\nSnippet: {snippet}")

        return "\n\n".join(output)
    except Exception as e:
        logging.error(f"Fandom search error: {e}")
        return f"Error searching Anime Fandom: {str(e)}"


# --- Langchain Tools ---

wikipedia_tool = LangchainTool(
    tool=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
)

google_search_tool = LangchainTool(
    tool=GoogleSearchRun(api_wrapper=GoogleSearchAPIWrapper())
)

# --- Agents ---

# 1. Anime Researcher Agent
anime_researcher = Agent(
    name="anime_researcher",
    model=model_name,
    description="Researches anime questions using Wikipedia and Fandom.",
    instruction="""
    You are an expert anime researcher. Your goal is to fully answer the user's PROMPT
    about anime using all available tools.

    You have access to three tools:
    1. Wikipedia - for general anime knowledge, plot summaries, character info, and production details.
    2. search_anime_fandom - for detailed anime lore, episode guides, character backstories, and fan-curated info.

    Research strategy:
    - Start by searching Wikipedia for a broad overview.
    - Then search Anime Fandom for more detailed or specific information.
    - Combine and synthesize the results from all tools into comprehensive research data.

    PROMPT:
    { PROMPT }
    """,
    tools=[wikipedia_tool, search_anime_fandom],
    output_key="research_data",
)

# 2. Response Formatter Agent
response_formatter = Agent(
    name="response_formatter",
    model=model_name,
    description="Formats the anime research into a structured response.",
    instruction="""
    You are the response formatter for the Anime Q&A Agent. Your task is to take the
    RESEARCH_DATA and the original PROMPT, then present the answer in the following
    STRICT format:

    Anime: <anime name that the question is about>
    Query: <the original question the user asked>
    Answer: <a clear, detailed, and accurate answer based on the research data>

    Rules:
    - Extract the anime name from the context of the question and research.
    - The Query must be the user's original question as-is.
    - The Answer should be comprehensive yet concise, synthesizing all research findings.
    - If the research covers multiple anime, list each one separately in the same format.
    - Be accurate. Only include information supported by the research data.

    PROMPT:
    { PROMPT }

    RESEARCH_DATA:
    { research_data }
    """,
)

# 3. Anime Q&A Workflow
anime_qna_workflow = SequentialAgent(
    name="anime_qna_workflow",
    description="The workflow for researching and answering anime questions.",
    sub_agents=[
        anime_researcher,
        response_formatter,
    ],
)

# --- Root Agent ---

root_agent = Agent(
    name="greeter",
    model=model_name,
    description="The main entry point for the Anime Q&A Agent.",
    instruction="""
    You are the Anime Q&A Agent! Your job is to help users with questions about anime.

    When a user first interacts with you:
    - Greet them warmly.
    - Introduce yourself as the Anime Q&A Agent.
    - Tell them they can ask you any question about any anime — characters, plot, episodes,
      recommendations, trivia, and more.

    When the user asks a question:
    - First, determine if the question is related to anime.
    - If the question is NOT about anime, politely tell the user:
      "That question is outside my area of expertise. I'm here to help with anime-related
      questions only! Please ask me something about anime."
      Do NOT use the 'add_prompt_to_state' tool for non-anime questions.
    - If the question IS about anime, use the 'add_prompt_to_state' tool to save their question.
      After using the tool, transfer control to the 'anime_qna_workflow' agent.
    """,
    tools=[add_prompt_to_state],
    sub_agents=[anime_qna_workflow],
)
