"""Shared, strictly validated schemas for code-review results."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class OverallRating(str, Enum):
    """Allowed overall assessments for a code review."""

    GOOD = "good"
    NEEDS_WORK = "needs_work"
    CRITICAL_ISSUES = "critical_issues"


class FindingCategory(str, Enum):
    """Areas that the reviewer evaluates."""

    BUG = "bug"
    SECURITY = "security"
    PERFORMANCE = "performance"
    STYLE = "style"
    IMPROVEMENT = "improvement"


class Severity(str, Enum):
    """Impact level of an individual finding."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ReviewFinding(BaseModel):
    """One actionable issue found during review."""

    model_config = ConfigDict(extra="forbid")

    category: FindingCategory = Field(
        description="The review area this finding belongs to."
    )
    severity: Severity = Field(description="The impact of the finding.")
    title: str = Field(description="A short, specific title.", min_length=1)
    description: str = Field(
        description="Why this is an issue and when it matters.",
        min_length=1,
    )
    line_number: int | None = Field(
        default=None,
        description="The relevant 1-based line number, when known.",
        ge=1,
    )
    suggestion: str = Field(
        description="A concrete way to address the finding.",
        min_length=1,
    )


class CodeReview(BaseModel):
    """Validated, machine-readable result returned by the reviewer."""

    model_config = ConfigDict(extra="forbid")

    overall_rating: OverallRating
    summary: str = Field(description="A concise overall assessment.", min_length=1)
    findings: list[ReviewFinding] = Field(
        default_factory=list,
        description="Concrete findings; use an empty list when no issues are found.",
    )
