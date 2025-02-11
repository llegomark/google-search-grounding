"""
Research Tool using Google's Gemini AI with Search Grounding and Chat
A tool that leverages Gemini AI and Google Search to research topics and maintain conversations.
"""

import os
import pytz
import signal
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


@dataclass
class ResearchResult:
    topic: str
    content: str
    timestamp: datetime
    model: str
    search_metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'timestamp': self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        }


class ResearchChat:
    def __init__(self, client: genai.Client, model: str):
        search_config = types.GoogleSearch()

        config = types.GenerateContentConfig(
            temperature=1,
            top_p=0.95,
            top_k=40,
            max_output_tokens=8192,
            tools=[types.Tool(google_search=search_config)],
        )

        self.chat = client.chats.create(model=model, config=config)
        self.history = []

    def send_message(self, message: str) -> str:
        try:
            print("\nGenerating response (streaming)...")
            print("-" * 80 + "\n")

            full_response = ""
            response_metadata = None

            for response in self.chat.send_message_stream(message):
                if hasattr(response, 'text') and response.text:
                    print(response.text, end='', flush=True)
                    full_response += response.text

                if hasattr(response, 'candidates') and response.candidates:
                    for candidate in response.candidates:
                        if hasattr(candidate, 'grounding_metadata'):
                            response_metadata = candidate.grounding_metadata

            print("\n" + "-" * 80)

            self.history.append({
                "role": "user",
                "content": message,
                "timestamp": datetime.now(pytz.UTC).isoformat()
            })

            search_info = self._extract_search_metadata(response_metadata)

            self.history.append({
                "role": "assistant",
                "content": full_response,
                "search_metadata": search_info,
                "timestamp": datetime.now(pytz.UTC).isoformat()
            })

            return full_response

        except Exception as e:
            raise Exception(f"Chat response failed: {str(e)}") from e

    def _extract_search_metadata(self, metadata: Optional[Any]) -> Dict[str, Any]:
        if not metadata:
            return {"search_used": False}

        search_info = {
            "search_used": True,
            "searches": [],
            "sources": [],
            "timestamp": datetime.now(pytz.UTC).isoformat()
        }

        try:
            if hasattr(metadata, 'groundingSearchResults'):
                for result in metadata.groundingSearchResults:
                    search_info["searches"].append({
                        "query": result.searchQuery,
                        "timestamp": result.searchTime,
                        "results_count": len(result.searchResults)
                    })

                    for search_result in result.searchResults:
                        search_info["sources"].append({
                            "title": search_result.title,
                            "url": search_result.url,
                            "snippet": search_result.snippet,
                            "timestamp": search_result.get('timestamp', None)
                        })
        except AttributeError as e:
            search_info["error"] = f"Error extracting search metadata: {str(e)}"

        return search_info

    def get_history(self) -> List[Dict[str, Any]]:
        return self.history


class ResearchTool:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError(
                "API key not found. Please provide via:\n"
                "1. Parameter: ResearchTool(api_key='your_key')\n"
                "2. Environment variable: GOOGLE_API_KEY\n"
                "3. .env file: GOOGLE_API_KEY=your_key"
            )

        self.client = genai.Client(
            api_key=self.api_key,
        )
        self.model = 'gemini-2.0-flash'

    def create_research_prompt(self, topic: str) -> str:
        manila_tz = pytz.timezone('Asia/Manila')
        manila_time = datetime.now(manila_tz).strftime("%Y-%m-%d %I:%M %p")

        prompt = f"""
                USE GOOGLE SEARCH: {topic}

                Current Date and Time (Manila, GMT+8): {manila_time}

                Instructions:
                **1. YOU MUST USE GOOGLE SEARCH to find accurate and up-to-date information.**
                2. For each search result used, include:
                - The source title and URL
                - Publication date when available
                - Brief summary of relevant information
                3. Focus on:
                - Recent and reliable sources
                - Verified information
                - Multiple perspectives when relevant
                4. Structure the response with:
                - Main findings
                - Supporting details
                - Sources used
                5. Maintain accuracy while being concise

                Please provide a comprehensive but clear overview of the topic, **making sure to use Google Search as instructed.**
                """
        return prompt.strip()

    def start_chat(self) -> ResearchChat:
        return ResearchChat(self.client, self.model)

    def save_chat_results(self, topic: str, chat_history: List[Dict[str, Any]],
                          reason: str = "normal") -> None:
        try:
            output_dir = Path("research_results")
            output_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_topic = "".join(
                c for c in topic if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_topic = safe_topic.replace(' ', '_')
            truncated_topic = safe_topic[:50] if len(
                safe_topic) > 50 else safe_topic

            filename = f"{timestamp}_{truncated_topic}.txt"
            output_path = output_dir / filename

            manila_tz = pytz.timezone('Asia/Manila')
            manila_time = datetime.now(manila_tz).strftime("%Y-%m-%d %I:%M %p")

            content = f"""Research Results
================

Topic: {topic}
Timestamp (Manila, GMT+8): {manila_time}
Model: {self.model}
End Reason: {reason}

Conversation History:
-------------------
"""

            for msg in chat_history:
                role = msg["role"].title()
                timestamp = datetime.fromisoformat(msg["timestamp"]).strftime(
                    "%Y-%m-%d %I:%M:%S %p %Z")

                content += f"\n[{timestamp}] {role}:\n"
                content += "-" * (len(timestamp) + len(role) + 4) + "\n"
                content += f"{msg['content']}\n"

                if role == "Assistant" and "search_metadata" in msg:
                    metadata = msg["search_metadata"]
                    content += "\nSearch Information:\n"
                    content += f"Search Used: {metadata['search_used']}\n"

                    if metadata['search_used']:
                        content += "\nSearches Performed:\n"
                        for search in metadata.get('searches', []):
                            content += f"- Query: {search['query']}\n"
                            content += f"  Time: {search['timestamp']}\n"
                            content += f"  Results: {search['results_count']}\n"

                        content += "\nSources Used:\n"
                        for source in metadata.get('sources', []):
                            content += f"- Title: {source['title']}\n"
                            content += f"  URL: {source['url']}\n"
                            content += f"  Snippet: {source['snippet']}\n"
                            if source['timestamp']:
                                content += f"  Published: {source['timestamp']}\n"
                            content += "\n"

            output_path.write_text(content, encoding='utf-8')
            print(f"\nResearch results saved to: {output_path}")

        except Exception as e:
            raise Exception(f"Error saving research results: {str(e)}") from e


def signal_handler(signum, frame):
    print('\n\nInterrupted by user. Saving research results...')
    if 'chat' in globals() and 'researcher' in globals() and 'topic' in globals():
        researcher.save_chat_results(topic, chat.get_history(), "interrupted")
    print("Research session ended. Results have been saved.")
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, signal_handler)

    global researcher, chat, topic

    try:
        researcher = ResearchTool()

        print("\n=== AI Research Assistant ===")
        print("Powered by Gemini AI with Google Search Grounding")
        print("\nCommands:")
        print("- Type 'quit' or 'exit' to end the session")
        print("- Type 'save' to save the current results")
        print("- Press Ctrl+C to interrupt and save")

        topic = input("\nWhat would you like to research?: ").strip()
        while not topic:
            topic = input("Please enter a research topic: ").strip()

        chat = researcher.start_chat()

        initial_prompt = researcher.create_research_prompt(topic)
        chat.send_message(initial_prompt)

        while True:
            try:
                user_input = input(
                    "\nYour follow-up question (or command): ").strip()

                if user_input.lower() in ['quit', 'exit']:
                    researcher.save_chat_results(
                        topic, chat.get_history(), "normal")
                    print("\nResearch session ended. Results have been saved.")
                    break

                if user_input.lower() == 'save':
                    researcher.save_chat_results(
                        topic, chat.get_history(), "saved")
                    print("\nCurrent results saved. You can continue researching.")
                    continue

                if user_input:
                    chat.send_message(user_input)

            except EOFError:
                researcher.save_chat_results(
                    topic, chat.get_history(), "interrupted")
                print("\nResearch session ended. Results have been saved.")
                break

    except Exception as e:
        if 'chat' in locals() and 'researcher' in locals() and 'topic' in locals():
            researcher.save_chat_results(
                topic, chat.get_history(), f"error: {str(e)}")
        print(f"\nError: {str(e)}")
        raise


if __name__ == "__main__":
    main()
