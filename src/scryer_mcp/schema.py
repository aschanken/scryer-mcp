"""schema.py — Scryer request/response models.

Single source of truth for the skill's interface.
All calling agents and internal scripts import from here.
Pydantic v2. Python >=3.10 required.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ---- Enums & Constants ----

class Tier(str, Enum):
    INSTANT = "instant"
    FAST = "fast"
    AUTO = "auto"
    DEEP = "deep"
    DEEP_REASONING = "deep-reasoning"


class Category(str, Enum):
    COMPANY = "company"
    PEOPLE = "people"
    NEWS = "news"
    RESEARCH = "research"
    PAPER = "paper"
    CODE = "code"
    GENERAL = "general"


# ---- Response Models ----

class StructuredItem(BaseModel):
    """A single discovered structured content object from a web page.

    Automatically extracted from HTML during fetch — no prior knowledge required.
    """
    type: str = Field(..., description="One of: json_ld, open_graph, microdata, table")
    name: Optional[str] = Field(default=None, description="Human-readable label for the item")
    data: dict | list | str | int | float | bool | None = Field(
        default=None, description="The extracted structured data"
    )


class SearchResult(BaseModel):
    """A single search result."""
    title: str
    url: str
    published_date: Optional[str] = None
    author: Optional[str] = None
    snippet: str                                  # Always present (from search)
    highlights: Optional[str] = None              # Present if highlights=True
    full_text: Optional[str] = None               # Present if full_text=True
    score: float = Field(default=1.0, ge=0.0, le=1.0)
    category: Optional[Category] = None
    structured_items: list[StructuredItem] = Field(
        default_factory=list,
        description="Automatically-discovered structured content (JSON-LD, OG, microdata, tables)",
    )


class StructuredData(BaseModel):
    """Populated when structured_output_schema is provided."""
    items: list[dict] = Field(default_factory=list)
    schema_used: Optional[dict] = None
    extraction_errors: int = 0                    # Count of failed extractions


class TierTrace(BaseModel):
    """Transparent breakdown of what each tier did."""
    tier: str
    searches_performed: int
    urls_fetched: int
    urls_skipped_cache: int
    urls_failed: int                              # Dead links encountered
    wall_time_ms: int
    tokens_consumed: int


class ScryerResponse(BaseModel):
    """Response shape — what the calling agent receives."""
    results: list[SearchResult]
    search_time_ms: int
    tier_used: str
    trace: TierTrace
    structured_data: Optional[StructuredData] = None
    grounded_answer: Optional[str] = None
    citations: list[str] = Field(default_factory=list)
    cot: Optional[str] = None
    verifications: Optional[list[dict]] = None
    error: Optional[str] = None                  # Non-empty only on partial failure


# ---- Request Model ----

class ScryerRequest(BaseModel):
    """Request shape — what the calling agent submits."""
    query: str = Field(..., min_length=1)           # Required, non-empty
    tier: Tier = Tier.AUTO
    num_results: int = Field(default=10, ge=1, le=50)
    category: Optional[Category] = None
    highlights: bool = True                       # Token-efficient extracts
    full_text: bool = False                       # Full page content
    livecrawl: bool = True                        # False = snippets only
    structured_output_schema: Optional[dict] = None  # JSON Schema for extraction
    max_tokens: Optional[int] = None              # Soft cap on output
    prompt: Optional[str] = Field(
        default=None,
        description="Instructional prompt passed to the LLM when synthesizing, "
                    "extracting, or processing content. Additive — does not "
                    "replace built-in instructions.",
    )

    @field_validator("num_results")
    @classmethod
    def cap_by_tier(cls, v: int, info) -> int:
        """Auto-adjust num_results to tier-appropriate maximum."""
        tier = info.data.get("tier", Tier.AUTO)
        caps: dict[Tier, int] = {
            Tier.INSTANT: 20,
            Tier.FAST: 20,
            Tier.AUTO: 30,
            Tier.DEEP: 40,
            Tier.DEEP_REASONING: 50,
        }
        return min(v, caps.get(tier, 50))


# ---- Helper: JSON Schema Validation ----

def validate_extraction_schema(schema: dict) -> Optional[str]:
    """Validate a user-provided JSON Schema.

    Returns None if valid, error string if not.
    """
    try:
        import jsonschema
        jsonschema.Draft7Validator.check_schema(schema)
        return None
    except ImportError:
        return "jsonschema library not available — install with: pip install jsonschema"
    except Exception as e:
        return str(e)
