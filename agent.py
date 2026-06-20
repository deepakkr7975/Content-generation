"""
LangGraph Content Generation Agent for ReinforcedX.

Automates content generation using a state graph:
    START → generate → validate → save → END

Every run is traced in LangSmith for observability.

Usage:
    python3 agent.py "prompt-engineering-techniques" --tag ai-systems
    python3 agent.py "build-rag-chatbot" --tag how-to
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from pydantic import ValidationError
from typing_extensions import TypedDict

# ── Setup ────────────────────────────────────────────────────────────────────

# Load .env from project root (one level up from Content_generation/)
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(ENV_PATH)

# Add current dir to path so we can import models
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import TopicContent  # noqa: E402

HERE = Path(__file__).resolve().parent

# Output directory for generated content
OUTPUT_DIR = HERE / "test-output"

# Source files (used for style examples)
SOURCE_FILES = {
    "ai-systems": HERE / "ai-systems-internal.json",
    "how-to": HERE / "how-to-internal.json",
}

# Output files
OUTPUT_FILES = {
    "ai-systems": OUTPUT_DIR / "ai-systems-internal.json",
    "how-to": OUTPUT_DIR / "how-to-internal.json",
}

MAX_RETRIES = 2

# ── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert technical content writer for ReinforcedX, a developer education platform.
You produce deeply technical, opinionated, production-focused content about AI/ML engineering.

Writing style rules:
- Write like a senior engineer explaining to a competent peer, not a tutorial for beginners.
- Be opinionated and specific — say "do X" not "you might consider X".
- Use concrete numbers, thresholds, and real tool/technique names when possible.
- Each paragraph should be 2–4 sentences, dense with insight.
- Takeaways should be standalone — each one useful even without reading the full article.
- FAQs should answer the question directly in the first sentence, then elaborate.
- Section IDs should be short slugs (e.g., "why", "architecture", "eval").
- directAnswer should be a single comprehensive sentence (40–80 words).
"""


# ── Helper Functions ─────────────────────────────────────────────────────────


def get_example(tag: str) -> str:
    """Load the first topic from the source JSON file as a style example."""
    file_path = SOURCE_FILES[tag]
    if not file_path.exists():
        return "{}"
    data = json.loads(file_path.read_text())
    first_key = next(iter(data))
    example = data[first_key]
    trimmed = {
        "directAnswer": example["directAnswer"],
        "takeaways": example["takeaways"],
        "sections": example["sections"][:2],
        "faqs": example["faqs"][:2],
    }
    if "steps" in example and example["steps"]:
        trimmed["steps"] = example["steps"][:2]
    return json.dumps(trimmed, indent=2)


def build_prompt(topic_slug: str, tag: str, example_json: str) -> str:
    """Build the generation prompt based on the tag type."""
    topic_display = topic_slug.replace("-", " ").title()

    if tag == "how-to":
        return f"""\
Generate a comprehensive how-to guide for the topic: "{topic_display}"
Topic slug: "{topic_slug}"

This is a HOW-TO guide, so you MUST include:
- directAnswer: A single sentence starting with "To [verb]..." explaining the high-level steps
- takeaways: 5 key takeaways (practical, actionable)
- steps: 5-7 ordered steps (each with a short imperative name and a detailed text paragraph)
- sections: 5-7 in-depth sections with detailed paragraphs. Some sections should have bullets.
- faqs: 5 frequently asked questions with direct answers

Here is an example of an existing topic in the same format for style/depth reference:
{example_json}
"""
    else:  # ai-systems
        return f"""\
Generate a comprehensive technical deep-dive for the topic: "{topic_display}"
Topic slug: "{topic_slug}"

This is a CONCEPTUAL deep-dive (NOT a how-to), so:
- directAnswer: A single sentence defining what this is and why it matters
- takeaways: 5 key takeaways (insightful, opinionated)
- Do NOT include steps (this is not a how-to guide)
- sections: 5-7 in-depth sections with detailed paragraphs. Some sections should have bullets.
- faqs: 5 frequently asked questions with direct answers

Here is an example of an existing topic in the same format for style/depth reference:
{example_json}
"""


# ── State ────────────────────────────────────────────────────────────────────


class ContentState(TypedDict):
    topic_slug: str
    tag: str
    generated_content: Optional[dict]
    validation_error: Optional[str]
    retry_count: int
    saved_path: Optional[str]


# ── Nodes ────────────────────────────────────────────────────────────────────


def generate_node(state: ContentState) -> dict:
    """Generate content using Gemini with structured output (TopicContent schema)."""

    topic_slug = state["topic_slug"]
    tag = state["tag"]
    retry = state.get("retry_count", 0)

    print(f"\n🚀 Generating '{topic_slug}' (tag: {tag})"
          f"{f' [retry {retry}]' if retry > 0 else ''}...")

    # Load style example from source files
    example_json = get_example(tag)

    # Build the prompt
    prompt = build_prompt(topic_slug, tag, example_json)

    # Initialize Gemini with structured output via LangChain
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
    )
    structured_llm = llm.with_structured_output(TopicContent, method="json_schema")

    # Generate
    result = structured_llm.invoke(
        [
            ("system", SYSTEM_PROMPT),
            ("human", prompt),
        ]
    )

    # Convert Pydantic model to dict
    content_dict = result.model_dump(exclude_none=True)

    print(f"Gemini returned structured content")

    return {
        "generated_content": content_dict,
        "validation_error": None,
    }


def validate_node(state: ContentState) -> dict:
    """Validate the generated content against the Pydantic TopicContent model."""

    content = state.get("generated_content")
    tag = state["tag"]
    topic_slug = state["topic_slug"]

    if content is None:
        return {
            "validation_error": "No content was generated",
            "retry_count": state.get("retry_count", 0) + 1,
        }

    try:
        # Re-validate through Pydantic
        topic = TopicContent.model_validate(content)

        # For ai-systems, strip steps
        if tag == "ai-systems" and topic.steps is not None:
            topic.steps = None
            content = topic.model_dump(exclude_none=True)

        # Quick sanity checks
        checks = []
        if len(topic.takeaways) < 3:
            checks.append(f"Only {len(topic.takeaways)} takeaways (need 3+)")
        if len(topic.sections) < 3:
            checks.append(f"Only {len(topic.sections)} sections (need 3+)")
        if len(topic.faqs) < 3:
            checks.append(f"Only {len(topic.faqs)} FAQs (need 3+)")
        if tag == "how-to" and (not topic.steps or len(topic.steps) < 3):
            checks.append("How-to guide needs 3+ steps")
        if len(topic.directAnswer) < 50:
            checks.append("directAnswer too short (need 50+ chars)")

        if checks:
            error_msg = "; ".join(checks)
            print(f"   ⚠️  Validation issues: {error_msg}")
            return {
                "generated_content": content,
                "validation_error": error_msg,
                "retry_count": state.get("retry_count", 0) + 1,
            }

        print(f"      Validation passed")
        print(f"      directAnswer: {topic.directAnswer[:80]}...")
        print(f"      takeaways:    {len(topic.takeaways)}")
        print(f"      steps:        {len(topic.steps) if topic.steps else 'N/A'}")
        print(f"      sections:     {len(topic.sections)}")
        print(f"      faqs:         {len(topic.faqs)}")

        return {
            "generated_content": content,
            "validation_error": None,
        }

    except ValidationError as e:
        error_msg = str(e)
        print(f"    Pydantic validation failed: {error_msg[:200]}")
        return {
            "validation_error": error_msg,
            "retry_count": state.get("retry_count", 0) + 1,
        }


def save_node(state: ContentState) -> dict:
    """Save the validated topic to the test-output JSON file (appends if exists)."""

    topic_slug = state["topic_slug"]
    tag = state["tag"]
    content = state["generated_content"]

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    file_path = OUTPUT_FILES[tag]

    # Load existing data or start fresh
    if file_path.exists():
        data = json.loads(file_path.read_text())
    else:
        data = {}

    # Append the new topic
    data[topic_slug] = content

    # Write back
    file_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")

    total = len(data)
    print(f"\n Saved to: {file_path}")
    print(f"   Total topics in file: {total}")

    return {"saved_path": str(file_path)}


# ── Routing ──────────────────────────────────────────────────────────────────


def should_retry_or_save(state: ContentState) -> str:
    """Route after validation: retry generation or proceed to save."""

    if state.get("validation_error") is None:
        return "save"

    retry_count = state.get("retry_count", 0)
    if retry_count <= MAX_RETRIES:
        print(f"    Retrying generation (attempt {retry_count}/{MAX_RETRIES})...")
        return "generate"

    print(f"    Max retries ({MAX_RETRIES}) exceeded. Skipping topic.")
    return END


# ── Graph Construction ───────────────────────────────────────────────────────


def build_graph() -> StateGraph:
    """Build and compile the LangGraph content generation agent."""

    graph = StateGraph(ContentState)

    # Add nodes
    graph.add_node("generate", generate_node)
    graph.add_node("validate", validate_node)
    graph.add_node("save", save_node)

    # Define edges
    graph.set_entry_point("generate")
    graph.add_edge("generate", "validate")
    graph.add_conditional_edges("validate", should_retry_or_save)
    graph.add_edge("save", END)

    return graph.compile()


# ── Compiled graph (for langgraph dev / LangSmith Studio) ────────────────────

graph = build_graph()


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="LangGraph Content Generation Agent for ReinforcedX"
    )
    parser.add_argument(
        "topic",
        help='Topic slug, e.g. "prompt-engineering-techniques"',
    )
    parser.add_argument(
        "--tag",
        choices=["ai-systems", "how-to"],
        required=True,
        help="Content type: ai-systems (conceptual) or how-to (guide)",
    )
    args = parser.parse_args()

    # Build the graph
    app = build_graph()

    # Initial state
    initial_state: ContentState = {
        "topic_slug": args.topic,
        "tag": args.tag,
        "generated_content": None,
        "validation_error": None,
        "retry_count": 0,
        "saved_path": None,
    }

    print(f"\n{'='*60}")
    print(f"  🤖 ReinforcedX Content Agent (LangGraph)")
    print(f"  Topic: {args.topic}")
    print(f"  Tag:   {args.tag}")
    print(f"{'='*60}")

    # Run the graph
    final_state = app.invoke(initial_state)

    # Summary
    print(f"\n{'='*60}")
    if final_state.get("saved_path"):
        print(f"  ✅ Done! Content saved to: {final_state['saved_path']}")
    else:
        print(f"  ❌ Failed to generate content after {MAX_RETRIES} retries.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
