"""
Pydantic models for AI-Zoned content generation JSONs.

Covers two content shapes:
  • "ai-systems-internal.json" — conceptual topics with sections (no steps)
  • "how-to-internal.json"     — how-to topics with steps AND sections

Both share: directAnswer, takeaways, sections, faqs.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ── Leaf models ──────────────────────────────────────────────────────────────


class FAQ(BaseModel):
    """A single frequently-asked question with its answer."""

    q: str = Field(..., description="The question text")
    a: str = Field(..., description="The answer text")


class Section(BaseModel):
    """
    A content section within a topic.

    Every section has an id, heading, and at least one paragraph.
    Bullets are optional (only some sections include them).
    """

    id: str = Field(..., description="Short slug identifier for the section")
    heading: str = Field(..., description="Display heading")
    paras: List[str] = Field(
        ..., min_length=1, description="One or more body paragraphs"
    )
    bullets: Optional[List[str]] = Field(
        None, description="Optional bullet-point list"
    )


class Step(BaseModel):
    """
    A single step in a how-to guide.

    Steps appear only in how-to content (e.g. how-to-internal.json).
    """

    name: str = Field(..., description="Short imperative step title")
    text: str = Field(..., description="Detailed step description")


# ── Topic model ──────────────────────────────────────────────────────────────


class TopicContent(BaseModel):
    """
    A complete topic entry.

    • Conceptual topics  → steps is None
    • How-to topics      → steps is a non-empty list
    """

    directAnswer: str = Field(
        ...,
        description="A concise, direct answer to the topic's core question",
    )
    takeaways: List[str] = Field(
        ...,
        min_length=1,
        description="Key takeaways / bullet summary",
    )
    steps: Optional[List[Step]] = Field(
        None,
        description="Ordered how-to steps (present only in how-to guides)",
    )
    sections: List[Section] = Field(
        ...,
        min_length=1,
        description="In-depth content sections",
    )
    faqs: List[FAQ] = Field(
        ...,
        min_length=1,
        description="Frequently asked questions for the topic",
    )


# ── Root type ────────────────────────────────────────────────────────────────

# Each JSON file is a dict mapping topic slugs to TopicContent.
ContentFile = Dict[str, TopicContent]
