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
import re
import sys
import time
from datetime import datetime
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
    # Scheduling fields (optional — only used in scheduled mode)
    topics: Optional[list]             # Full topics list from topics.json
    topics_file: Optional[str]         # Path to topics.json
    interval_seconds: Optional[int]    # Seconds between generations
    current_index: Optional[int]       # Index of current topic in list


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

    if content is None:
        return {
            "validation_error": "No content was generated",
            "retry_count": state.get("retry_count", 0) + 1,
        }

    try:
        topic = TopicContent.model_validate(content)

        if tag == "ai-systems" and topic.steps is not None:
            topic.steps = None
            content = topic.model_dump(exclude_none=True)

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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    file_path = OUTPUT_FILES[tag]

    if file_path.exists():
        data = json.loads(file_path.read_text())
    else:
        data = {}

    data[topic_slug] = content
    file_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")

    total = len(data)
    print(f"\n Saved to: {file_path}")
    print(f"   Total topics in file: {total}")

    # If in scheduled mode, mark topic as done in topics.json
    topics = state.get("topics")
    idx = state.get("current_index")
    if topics is not None and idx is not None:
        topics[idx]["status"] = "done"
        topics[idx]["completed_at"] = datetime.now().isoformat()
        topics_file = Path(state["topics_file"])
        save_topics(topics_file, topics)

    return {"saved_path": str(file_path)}


def schedule_node(state: ContentState) -> dict:
    """Wait for the scheduled interval, then load the next pending topic."""

    topics = state.get("topics")
    interval = state.get("interval_seconds", 0)

    if topics is None:
        # Single-topic mode — no scheduling, just end
        return {}

    # Find next pending topic
    start = state.get("current_index", -1) + 1
    for i in range(start, len(topics)):
        if topics[i].get("status") not in ("done", "failed", "error"):
            # Wait for interval before next topic
            if interval > 0:
                interval_str = f"{interval // 3600}h" if interval >= 3600 else f"{interval // 60}m"
                now = datetime.now().strftime("%H:%M:%S")
                print(f"\n  [{now}] Waiting {interval_str} before next topic...")
                time.sleep(interval)

            slug = topics[i]["slug"]
            tag = topics[i]["tag"]
            now = datetime.now().strftime("%H:%M:%S")
            print(f"\n  [{now}] Next: '{slug}' ({tag})")

            return {
                "current_index": i,
                "topic_slug": slug,
                "tag": tag,
                "generated_content": None,
                "validation_error": None,
                "retry_count": 0,
            }

    print(f"\n  All topics processed!")
    return {}


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


def should_schedule_or_end(state: ContentState) -> str:
    """Route after save: go to schedule (if more pending topics) or END."""

    topics = state.get("topics")
    if topics is None:
        return END  # single-topic mode → done

    # Check if there are more pending topics after the current one
    start = state.get("current_index", -1) + 1
    for i in range(start, len(topics)):
        if topics[i].get("status") not in ("done", "failed", "error"):
            return "schedule"  # more topics → go to schedule node

    return END  # all topics processed


# ── Graph Construction ───────────────────────────────────────────────────────


def build_graph():
    """Build the content generation graph with optional scheduling node.

    Graph structure:
        START → generate → validate →[retry?]→ generate (loop)
                                    →[ok?]→ save
        save →[schedule mode + more topics?]→ schedule → generate (loop)
            →[single topic / all done?]→ END
    """

    g = StateGraph(ContentState)

    g.add_node("generate", generate_node)
    g.add_node("validate", validate_node)
    g.add_node("save", save_node)
    g.add_node("schedule", schedule_node)

    g.set_entry_point("generate")
    g.add_edge("generate", "validate")
    g.add_conditional_edges("validate", should_retry_or_save, {
        "save": "save",
        "generate": "generate",
        END: END,
    })
    g.add_conditional_edges("save", should_schedule_or_end, {
        "schedule": "schedule",
        END: END,
    })
    g.add_edge("schedule", "generate")   # schedule always loops back

    return g.compile()


# ── Compiled graph (for langgraph dev / LangSmith Studio) ────────────────────

graph = build_graph()


# ── Scheduler ────────────────────────────────────────────────────────────────

TOPICS_FILE = HERE / "topics.json"


def parse_interval(interval_str: str) -> int:
    """Parse interval string like '2h', '30m', '10h' into seconds."""
    match = re.match(r"^(\d+)\s*(h|hr|hrs|hours?|m|min|mins|minutes?)$", interval_str.lower().strip())
    if not match:
        raise ValueError(
            f"Invalid interval '{interval_str}'. "
            f"Use format like: 30m, 1h, 2h, 10h"
        )
    value = int(match.group(1))
    unit = match.group(2)[0]  # 'h' or 'm'
    return value * 3600 if unit == "h" else value * 60


def load_topics(topics_file: Path) -> list:
    """Load topics list from JSON file."""
    if not topics_file.exists():
        print(f"  Topics file not found: {topics_file}")
        sys.exit(1)
    data = json.loads(topics_file.read_text())
    return data.get("topics", [])


def save_topics(topics_file: Path, topics: list) -> None:
    """Save updated topics list back to JSON file."""
    data = {"topics": topics}
    topics_file.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def run_scheduled(interval_str: str, topics_file: Path) -> None:
    """Run the agent on a schedule using the graph's schedule node."""

    interval_secs = parse_interval(interval_str)
    topics = load_topics(topics_file)

    pending = [t for t in topics if t.get("status") != "done"]

    if not pending:
        print(f"\n  All topics in {topics_file.name} are already done!")
        return

    # Find first pending topic
    first_slug, first_tag, first_idx = "", "", 0
    for i, t in enumerate(topics):
        if t.get("status") not in ("done", "failed", "error"):
            first_slug = t["slug"]
            first_tag = t["tag"]
            first_idx = i
            break

    print(f"\n{'='*60}")
    print(f"  ReinforcedX Content Agent — Scheduled Mode")
    print(f"  Interval:      {interval_str} ({interval_secs}s)")
    print(f"  Topics file:   {topics_file.name}")
    print(f"  Total topics:  {len(topics)}")
    print(f"  Pending:       {len(pending)}")
    print(f"{'='*60}")

    app = build_graph()

    # Run the graph — the schedule node handles the loop + waiting
    initial_state: ContentState = {
        "topic_slug": first_slug,
        "tag": first_tag,
        "generated_content": None,
        "validation_error": None,
        "retry_count": 0,
        "saved_path": None,
        "topics": topics,
        "topics_file": str(topics_file),
        "interval_seconds": interval_secs,
        "current_index": first_idx,
    }

    try:
        final_state = app.invoke(initial_state)
    except KeyboardInterrupt:
        print(f"\n  Scheduler stopped by user. Progress saved to {topics_file.name}.")
        return

    # Final summary
    done = sum(1 for t in topics if t.get("status") == "done")
    failed = sum(1 for t in topics if t.get("status") in ("failed", "error"))
    print(f"\n{'='*60}")
    print(f"  Schedule complete!")
    print(f"  Done:   {done}/{len(topics)}")
    if failed:
        print(f"  Failed: {failed}/{len(topics)}")
    print(f"{'='*60}\n")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="LangGraph Content Generation Agent for ReinforcedX"
    )

    # Single topic mode
    parser.add_argument(
        "topic",
        nargs="?",
        default=None,
        help='Topic slug, e.g. "prompt-engineering-techniques"',
    )
    parser.add_argument(
        "--tag",
        choices=["ai-systems", "how-to"],
        help="Content type: ai-systems (conceptual) or how-to (guide)",
    )

    # Scheduled mode
    parser.add_argument(
        "--schedule",
        metavar="INTERVAL",
        help='Run in scheduled mode. Interval like "30m", "1h", "2h", "10h"',
    )
    parser.add_argument(
        "--topics-file",
        default=str(TOPICS_FILE),
        help=f"Path to topics JSON file (default: {TOPICS_FILE.name})",
    )

    args = parser.parse_args()

    # ── Scheduled mode ──
    if args.schedule:
        topics_file = Path(args.topics_file)
        run_scheduled(args.schedule, topics_file)
        return

    # ── Single topic mode ──
    if not args.topic:
        parser.error("Provide a topic slug, or use --schedule for scheduled mode.")
    if not args.tag:
        parser.error("--tag is required for single topic mode.")

    app = build_graph()

    initial_state: ContentState = {
        "topic_slug": args.topic,
        "tag": args.tag,
        "generated_content": None,
        "validation_error": None,
        "retry_count": 0,
        "saved_path": None,
    }

    print(f"\n{'='*60}")
    print(f"  ReinforcedX Content Agent (LangGraph)")
    print(f"  Topic: {args.topic}")
    print(f"  Tag:   {args.tag}")
    print(f"{'='*60}")

    final_state = app.invoke(initial_state)

    print(f"\n{'='*60}")
    if final_state.get("saved_path"):
        print(f"   Done! Content saved to: {final_state['saved_path']}")
    else:
        print(f"  Failed to generate content after {MAX_RETRIES} retries.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
