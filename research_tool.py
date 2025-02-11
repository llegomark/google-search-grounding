"""
Research Tool using Google's Gemini AI with Search Grounding
A simple tool that leverages Gemini AI and Google Search to research any topic.
"""

import os
import json
import random
import pytz
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()


@dataclass
class ResearchResult:
    """Data class to store research results and metadata."""
    topic: str
    content: str
    timestamp: datetime
    model: str
    search_verification: Dict[str, bool]

    def to_dict(self) -> Dict[str, Any]:
        """Convert the research result to a dictionary format."""
        return {
            **asdict(self),
            'timestamp': self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        }


class ResearchTool:
    """A tool for conducting AI-powered research using Gemini and Google Search."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the research tool.

        Args:
            api_key: Optional API key for Gemini. If not provided, will look for
                    GOOGLE_API_KEY in environment variables.

        Raises:
            ValueError: If no API key is found.
        """
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError(
                "API key not found. Please provide via:\n"
                "1. Parameter: ResearchTool(api_key='your_key')\n"
                "2. Environment variable: GOOGLE_API_KEY\n"
                "3. .env file: GOOGLE_API_KEY=your_key"
            )
        self.client = genai.Client(api_key=self.api_key)
        self.model = 'gemini-2.0-flash'

    def create_research_prompt(self, topic: str) -> str:
        """
        Create a simple, universal research prompt for any topic with Manila time.

        Args:
            topic: The research topic to investigate

        Returns:
            str: A conversational research prompt
        """
        # Get current Manila time
        manila_tz = pytz.timezone('Asia/Manila')
        manila_time = datetime.now(manila_tz).strftime("%Y-%m-%d %I:%M %p")

        prompt = f"""
        Tell me everything important about {topic}.
        
        Current Date and Time (Manila, GMT+8): {manila_time}

        Please:
        - Use Google Search to find current information
        - Start each new topic with [Search: what you searched for]
        - Include recent dates and sources when sharing information
        - Keep it simple and easy to understand
        - Share interesting facts and stories
        - Tell me what's happening now
        - Explain why it matters
        - Share what experts think
        - Tell me what might happen next

        Make it conversational and engaging, like you're explaining to a friend.
        Focus on what's most interesting and important.
        """
        return prompt.strip()

    def conduct_research(self, topic: str) -> ResearchResult:
        """
        Conduct research on the given topic using Gemini with Google Search grounding.

        Args:
            topic: The research topic to investigate

        Returns:
            ResearchResult: Object containing research results and metadata

        Raises:
            Exception: If research generation fails
        """
        try:
            print(f"\nResearching: {topic}")
            print("Using Google Search for current information...")

            prompt = self.create_research_prompt(topic)

            # Configure generation parameters
            config = types.GenerateContentConfig(
                temperature=0.7,
                top_p=0.95,
                top_k=40,
                max_output_tokens=8192,
                tools=[types.Tool(google_search=types.GoogleSearch())],
            )

            # Generate content
            print("Generating response...")
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config
            )

            # Verify search usage
            search_verification = self._verify_search_usage(response.text)

            # Print verification results
            self._print_verification_results(search_verification)

            # Create research result
            result = ResearchResult(
                topic=topic,
                content=response.text,
                timestamp=datetime.now(),
                model=self.model,
                search_verification=search_verification
            )

            # Automatically save results
            self._save_results(result)

            return result

        except Exception as e:
            raise Exception(f"Research generation failed: {str(e)}") from e

    def _verify_search_usage(self, content: str) -> Dict[str, bool]:
        """
        Verify if search was used in the response.

        Args:
            content: The generated research content

        Returns:
            Dict[str, bool]: Dictionary containing verification results
        """
        search_indicators = {
            "explicit_tags": "[Search:" in content,
            "recent_references": any(
                indicator in content for indicator in [
                    "According to recent",
                    "As of",
                    "In recent news",
                    "Latest data shows",
                    "Recent studies"
                ]
            )
        }
        return search_indicators

    def _print_verification_results(self, verification: Dict[str, bool]) -> None:
        """
        Print the search verification results.

        Args:
            verification: Dictionary containing verification results
        """
        print("\nSearch grounding verification:")
        for key, value in verification.items():
            print(f"{'✓' if value else '✗'} {key.replace('_', ' ').title()}")

    def _save_results(self, result: ResearchResult) -> None:
        """
        Save research results to a file with truncated topic name and random suffix.

        Args:
            result: ResearchResult object containing the research data

        Raises:
            Exception: If saving results fails
        """
        try:
            # Create research_results directory if it doesn't exist
            output_dir = Path("research_results")
            output_dir.mkdir(exist_ok=True)

            # Generate random number suffix (4 digits)
            random_suffix = str(random.randint(1000, 9999))

            # Create filename with timestamp and random suffix
            timestamp = datetime.now().strftime("%Y%m%d")

            # Truncate and clean topic name (limit to 20 characters)
            safe_topic = "".join(
                c for c in result.topic if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_topic = safe_topic.replace(' ', '_')
            truncated_topic = safe_topic[:20] if len(
                safe_topic) > 20 else safe_topic

            filename = f"{timestamp}_{truncated_topic}_{random_suffix}.txt"

            output_path = output_dir / filename

            content = f"""Research Results
=================

Metadata:
{json.dumps(result.to_dict(), indent=2)}

Results:
{result.content}
"""

            output_path.write_text(content, encoding='utf-8')
            print(f"\nResults saved to: {output_path}")

        except Exception as e:
            raise Exception(f"Error saving results: {str(e)}") from e


def main():
    """Main function to run the research tool."""
    try:
        # Initialize the research tool
        researcher = ResearchTool()

        # Get user input for topic only
        print("\n=== Research Tool ===")
        print("This tool will research any topic using Gemini AI with Google Search")

        topic = input("\nWhat would you like to learn about?: ").strip()
        while not topic:
            topic = input("Please enter a topic: ").strip()

        # Conduct research
        result = researcher.conduct_research(topic)

        # Print results
        print("\nResearch Results:")
        print("=" * 80)
        print(result.content)

    except Exception as e:
        print(f"Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
