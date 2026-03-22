#!/usr/bin/env python3
"""Intelligent Bajaj Finserv Blu bot testing system.

This script:
1. Logs into the Bajaj Finserv Blu bot with Playwright.
2. Scrapes a source-of-truth webpage for all relevant content.
3. Extracts structured knowledge from the page without hardcoding domain fields.
4. Generates natural user-like questions.
5. Applies multiple personas to diversify phrasing.
6. Asks the bot each generated question.
7. Evaluates correctness using numeric, lexical, and optional semantic matching.
8. Generates CSV/Excel reports with aggregate metrics.

Usage example:
    python bot_tester.py \
        --product "business loan" \
        --source-url "https://www.example.com/business-loan" \
        --max-facts 25 \
        --max-questions-per-fact 3

Environment variables:
    OPENAI_API_KEY      Optional, enables semantic similarity scoring.
    OPENAI_MODEL        Optional, default: gpt-4.1-mini

Prerequisites:
    pip install playwright beautifulsoup4 pandas requests lxml
    playwright install chromium
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import urlparse

import csv



DEFAULT_BOT_URL = "https://www.bajajfinserv.in/blu/?jid=service"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "to", "of", "for",
    "in", "on", "at", "by", "with", "and", "or", "as", "from", "that", "this", "these", "those",
    "it", "its", "if", "into", "your", "you", "can", "i", "me", "my", "we", "our", "they",
    "their", "them", "what", "when", "where", "how", "which", "who", "whom", "why", "about",
    "tell", "please", "could", "would", "should", "there", "than", "then", "also", "any", "all",
    "more", "most", "up", "per", "each", "such", "other", "will", "may", "might", "do", "does",
}
CATEGORY_PATTERNS = {
    "rate": [r"\brate\b", r"interest", r"apr", r"percentage", r"p\.a\.?"],
    "fee": [r"fee", r"charge", r"charges", r"cost", r"penalty", r"gst", r"foreclosure"],
    "eligibility": [r"eligible", r"eligibility", r"age", r"income", r"cibil", r"employment"],
    "tenure": [r"tenure", r"duration", r"months?", r"years?", r"repayment", r"term"],
    "document": [r"document", r"kyc", r"proof", r"aadhaar", r"pan", r"bank statement"],
    "benefit": [r"benefit", r"feature", r"advantage", r"why choose", r"instant", r"quick"],
    "process": [r"apply", r"application", r"steps?", r"process", r"approval"],
    "limit": [r"limit", r"amount", r"loan amount", r"coverage", r"sum insured", r"up to"],
    "restriction": [r"not eligible", r"except", r"condition", r"subject to", r"restriction"],
}
IRRELEVANT_PHRASES = {
    "cookie", "privacy policy", "javascript", "subscribe", "download app", "share", "follow us",
    "social media", "advertisement", "copyright", "all rights reserved",
}
BOT_READY_SELECTORS = [
    "textarea",
    "input[type='text']",
    "[contenteditable='true']",
    "input[placeholder*='Ask']",
    "textarea[placeholder*='Ask']",
]
CHAT_INPUT_SELECTORS = [
    "textarea",
    "input[type='text']",
    "[contenteditable='true']",
]
SEND_BUTTON_SELECTORS = [
    "button:has-text('Send')",
    "button[aria-label*='send' i]",
    "button[type='submit']",
    "button svg",
]
BOT_RESPONSE_SELECTORS = [
    "[data-testid*='message']",
    ".bot-message",
    ".message",
    ".chat-message",
    ".markdown",
    "article",
]
NUMERIC_PATTERN = re.compile(
    r"(?:₹|rs\.?|inr\s*)?\d+(?:,\d{2,3})*(?:\.\d+)?(?:\s*(?:%|percent|percentage|years?|months?|days?|hrs?|hours?|lakhs?|crores?))?",
    re.IGNORECASE,
)
RANGE_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(%|years?|months?|days?)?\s*(?:to|-|–)\s*(\d+(?:\.\d+)?)\s*(%|years?|months?|days?)?",
    re.IGNORECASE,
)
KEY_VALUE_PATTERN = re.compile(
    r"^(?P<key>[A-Za-z][A-Za-z0-9 /()&,'-]{2,80})\s*[:\-–]\s*(?P<value>.+)$"
)


@dataclass
class RawContentItem:
    content_type: str
    text: str
    heading_context: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeItem:
    category: str
    text: str
    source_type: str
    heading_context: str
    keywords: list[str]
    numeric_values: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QuestionCandidate:
    knowledge_item: KnowledgeItem
    canonical_question: str
    variants: list[str]


@dataclass
class PersonaQuestion:
    persona_name: str
    persona_style: str
    question: str
    canonical_question: str
    knowledge_item: KnowledgeItem


@dataclass
class EvaluationResult:
    result: str
    confidence_score: int
    exact_match_score: float
    keyword_score: float
    numeric_score: float
    semantic_score: float
    notes: str


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def sentence_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", normalize_whitespace(text))
    return [p.strip(" -•\t") for p in parts if len(p.strip(" -•\t")) > 20]


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[A-Za-z0-9%]+", text.lower()) if t not in STOPWORDS and len(t) > 1]


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def jaccard_score(a: Iterable[str], b: Iterable[str]) -> float:
    set_a, set_b = set(a), set(b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        normalized = normalize_whitespace(item).lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            output.append(normalize_whitespace(item))
    return output


def categorize_text(text: str) -> str:
    lowered = text.lower()
    for category, patterns in CATEGORY_PATTERNS.items():
        if any(re.search(pattern, lowered) for pattern in patterns):
            return category
    if RANGE_PATTERN.search(text):
        return "range"
    if any(token in lowered for token in ["required", "must", "need to", "mandatory"]):
        return "requirement"
    return "general"


def is_probably_irrelevant(text: str) -> bool:
    lowered = text.lower()
    if len(lowered) < 25:
        return True
    if any(phrase in lowered for phrase in IRRELEVANT_PHRASES):
        return True
    alpha_ratio = sum(char.isalpha() for char in lowered) / max(1, len(lowered))
    return alpha_ratio < 0.45


def extract_numeric_values(text: str) -> list[str]:
    return dedupe_preserve_order(match.group(0) for match in NUMERIC_PATTERN.finditer(text))


def build_requests_session():
    import requests

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def scrape_full_content(source_url: str, timeout: int = 30) -> tuple[list[RawContentItem], str]:
    """Fetch and extract broad webpage content, preserving structure."""
    session = build_requests_session()
    response = session.get(source_url, timeout=timeout)
    response.raise_for_status()
    html = response.text
    from bs4 import BeautifulSoup, Tag

    soup = BeautifulSoup(html, "lxml")

    for tag_name in ["script", "style", "noscript", "svg", "footer", "form"]:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    items: list[RawContentItem] = []
    current_heading = ""

    def add_item(content_type: str, text: str, metadata: Optional[dict[str, Any]] = None) -> None:
        cleaned = normalize_whitespace(text)
        if not cleaned or is_probably_irrelevant(cleaned):
            return
        items.append(
            RawContentItem(
                content_type=content_type,
                text=cleaned,
                heading_context=current_heading,
                metadata=metadata or {},
            )
        )

    body = soup.body or soup
    for element in body.descendants:
        if not isinstance(element, Tag):
            continue

        if element.name in {"h1", "h2", "h3"}:
            heading_text = normalize_whitespace(element.get_text(" ", strip=True))
            if heading_text:
                current_heading = heading_text
                add_item(element.name, heading_text, {"level": element.name})

        elif element.name == "p":
            add_item("paragraph", element.get_text(" ", strip=True))

        elif element.name in {"ul", "ol"}:
            list_items = [normalize_whitespace(li.get_text(" ", strip=True)) for li in element.find_all("li", recursive=False)]
            for item in list_items:
                add_item("list_item", item)

        elif element.name == "table":
            rows = []
            for tr in element.find_all("tr"):
                cells = [normalize_whitespace(cell.get_text(" ", strip=True)) for cell in tr.find_all(["th", "td"])]
                cells = [cell for cell in cells if cell]
                if cells:
                    rows.append(cells)
            if rows:
                table_text = " | ".join(" ; ".join(row) for row in rows)
                add_item("table", table_text, {"rows": rows})
                if len(rows) > 1:
                    headers = rows[0]
                    for row in rows[1:]:
                        if len(row) == len(headers):
                            kv_pairs = [f"{header}: {value}" for header, value in zip(headers, row)]
                            add_item("table_row", "; ".join(kv_pairs), {"headers": headers, "row": row})

        elif element.name in {"div", "span"}:
            text = normalize_whitespace(element.get_text(" ", strip=True))
            if KEY_VALUE_PATTERN.match(text) and len(text) < 250:
                add_item("key_value", text)

    unique_items: list[RawContentItem] = []
    seen = set()
    for item in items:
        key = (item.content_type, item.text.lower())
        if key not in seen:
            seen.add(key)
            unique_items.append(item)
    return unique_items, html


def extract_structured_data(raw_items: list[RawContentItem], product_name: str) -> list[KnowledgeItem]:
    """Convert raw webpage content into structured knowledge entries."""
    knowledge_items: list[KnowledgeItem] = []

    for item in raw_items:
        sentences = sentence_split(item.text) or [item.text]
        for sentence in sentences:
            sentence = normalize_whitespace(sentence)
            if len(sentence) < 20:
                continue
            category = categorize_text(sentence)
            keywords = dedupe_preserve_order(tokenize(sentence)[:10])
            numeric_values = extract_numeric_values(sentence)
            metadata: dict[str, Any] = {}
            kv_match = KEY_VALUE_PATTERN.match(sentence)
            if kv_match:
                metadata["key"] = kv_match.group("key")
                metadata["value"] = kv_match.group("value")
            if product_name.lower() not in sentence.lower() and item.heading_context:
                metadata["qualified_text"] = f"{item.heading_context}: {sentence}"
            knowledge_items.append(
                KnowledgeItem(
                    category=category,
                    text=sentence,
                    source_type=item.content_type,
                    heading_context=item.heading_context,
                    keywords=keywords,
                    numeric_values=numeric_values,
                    metadata=metadata,
                )
            )

    scored_items: list[tuple[float, KnowledgeItem]] = []
    product_tokens = set(tokenize(product_name))
    for item in knowledge_items:
        score = 0.0
        score += 1.0 if item.category != "general" else 0.2
        score += min(len(item.numeric_values) * 0.5, 1.5)
        score += min(len(item.keywords) * 0.08, 0.8)
        score += 0.5 if item.source_type in {"table_row", "key_value", "list_item"} else 0.0
        score += 0.5 if item.heading_context else 0.0
        score += 0.7 * jaccard_score(product_tokens, item.keywords)
        scored_items.append((score, item))

    sorted_items = [item for _, item in sorted(scored_items, key=lambda pair: pair[0], reverse=True)]
    unique_items: list[KnowledgeItem] = []
    seen_texts: list[str] = []
    for item in sorted_items:
        lowered = item.text.lower()
        if any(similarity(lowered, existing) > 0.86 for existing in seen_texts):
            continue
        seen_texts.append(lowered)
        unique_items.append(item)
    return unique_items


QUESTION_PREFIXES = {
    "rate": [
        "What is the {product} rate?",
        "Can you tell me the interest rate for {product}?",
        "What rate range applies here?",
    ],
    "fee": [
        "What fees are charged for {product}?",
        "How much is the processing fee?",
        "Are there any extra charges I should know about?",
    ],
    "eligibility": [
        "Who is eligible for {product}?",
        "What are the eligibility requirements?",
        "What age or income conditions apply?",
    ],
    "tenure": [
        "What is the tenure for {product}?",
        "How long can the repayment period be?",
        "What is the minimum and maximum term?",
    ],
    "document": [
        "What documents are required?",
        "Which KYC proofs do I need?",
        "Can you list the documents needed for {product}?",
    ],
    "benefit": [
        "What are the main benefits of {product}?",
        "Why should someone choose this option?",
        "What features stand out here?",
    ],
    "process": [
        "How do I apply for {product}?",
        "What is the application process?",
        "Can you explain the steps to get started?",
    ],
    "limit": [
        "What amount can I get under {product}?",
        "What is the minimum or maximum limit?",
        "How much coverage or loan amount is available?",
    ],
    "restriction": [
        "Are there any restrictions or exclusions?",
        "When would someone not qualify?",
        "What conditions should I be careful about?",
    ],
    "range": [
        "What is the full range mentioned here?",
        "Can you share the exact minimum and maximum values?",
        "What does the range go from and to?",
    ],
    "requirement": [
        "What are the mandatory requirements?",
        "What do I need to meet before applying?",
        "Can you tell me the must-have conditions?",
    ],
    "general": [
        "Can you explain this {product} detail?",
        "What should I know about this point?",
        "Can you clarify this information for me?",
    ],
}

PERSONAS = {
    "Curious User": {
        "style": "Asks short, direct, beginner-friendly questions.",
        "transforms": [
            lambda q, item, product: q,
            lambda q, item, product: q.replace("Can you tell me", "Tell me").replace("What is the", "What's the"),
        ],
    },
    "Detailed User": {
        "style": "Asks complete, explicit, context-rich questions.",
        "transforms": [
            lambda q, item, product: f"For {product}, {q[0].lower() + q[1:]} Please include the exact details.",
            lambda q, item, product: f"Can you give me the complete information on this: {q.rstrip('?')}?",
        ],
    },
    "Confused User": {
        "style": "Uses vague, natural, slightly indirect phrasing.",
        "transforms": [
            lambda q, item, product: vagueify_question(q, item, product),
            lambda q, item, product: f"I'm a bit confused — {vagueify_question(q, item, product).rstrip('?')}?",
        ],
    },
    "Comparison User": {
        "style": "Frames questions in a market-comparison mindset.",
        "transforms": [
            lambda q, item, product: comparisonify_question(q, item, product),
            lambda q, item, product: f"Compared with similar options, {comparisonify_question(q, item, product).rstrip('?')}?",
        ],
    },
    "Edge Case User": {
        "style": "Tests conditions, exceptions, and borderline scenarios.",
        "transforms": [
            lambda q, item, product: edge_case_question(q, item, product),
            lambda q, item, product: f"What happens in a borderline case — {edge_case_question(q, item, product).rstrip('?')}?",
        ],
    },
}


def vagueify_question(question: str, item: KnowledgeItem, product_name: str) -> str:
    if item.category == "fee":
        return "How much would I actually need to pay here?"
    if item.category == "rate":
        return "What kind of rate does this usually come with?"
    if item.category == "eligibility":
        return "Would someone like me even qualify for this?"
    if item.category == "tenure":
        return "How long does this usually run for?"
    return f"How does this work for {product_name}?"


def comparisonify_question(question: str, item: KnowledgeItem, product_name: str) -> str:
    if item.category == "rate":
        return f"Is the {product_name} interest rate competitive compared with other options?"
    if item.category == "fee":
        return f"Are the charges for {product_name} on the higher side or not?"
    if item.category == "benefit":
        return f"What makes {product_name} better than similar products?"
    return f"How does this {product_name} detail compare with similar offerings?"


def edge_case_question(question: str, item: KnowledgeItem, product_name: str) -> str:
    if item.category == "eligibility":
        age_match = re.search(r"(\d{2})\s*years?", item.text, re.IGNORECASE)
        if age_match:
            edge_age = max(18, int(age_match.group(1)) - 2)
            return f"What if I am {edge_age} years old — would I still be eligible for {product_name}?"
        return f"What if I barely miss one eligibility condition for {product_name}?"
    if item.category == "tenure":
        return f"Can I choose a shorter or longer tenure than the standard range for {product_name}?"
    if item.category == "fee":
        return f"If I close {product_name} early, do any extra charges apply?"
    return f"What happens if my case does not neatly fit the stated conditions for {product_name}?"


def generate_questions(
    knowledge_items: list[KnowledgeItem],
    product_name: str,
    max_facts: int = 25,
    max_questions_per_fact: int = 3,
) -> list[QuestionCandidate]:
    """Generate natural question variants for each structured data point."""
    selected_items = knowledge_items[:max_facts]
    candidates: list[QuestionCandidate] = []

    for item in selected_items:
        templates = QUESTION_PREFIXES.get(item.category, QUESTION_PREFIXES["general"])
        seed_questions: list[str] = []
        for template in templates:
            question = template.format(product=product_name)
            seed_questions.append(question)

        if item.metadata.get("key"):
            key = item.metadata["key"]
            seed_questions.extend([
                f"What is the {key.lower()} for {product_name}?",
                f"Can you tell me about {key.lower()}?",
            ])

        if item.heading_context:
            seed_questions.append(f"Under {item.heading_context.lower()}, what does it say for {product_name}?")

        if item.numeric_values:
            seed_questions.append(f"What is the exact value mentioned for this {product_name} detail?")
            if len(item.numeric_values) >= 2:
                seed_questions.append("Can you share the minimum and maximum values exactly?")

        variations = dedupe_preserve_order(seed_questions)[: max_questions_per_fact + 2]
        canonical_question = variations[0]
        candidates.append(
            QuestionCandidate(
                knowledge_item=item,
                canonical_question=canonical_question,
                variants=variations[:max_questions_per_fact],
            )
        )
    return candidates


def apply_personas(question_candidates: list[QuestionCandidate], product_name: str) -> list[PersonaQuestion]:
    """Expand generated questions across multiple realistic user personas."""
    persona_questions: list[PersonaQuestion] = []
    seen: set[tuple[str, str]] = set()

    for candidate in question_candidates:
        base_variants = candidate.variants or [candidate.canonical_question]
        for persona_name, config in PERSONAS.items():
            transforms = config["transforms"]
            for idx, transform in enumerate(transforms):
                base = base_variants[idx % len(base_variants)]
                question = normalize_whitespace(transform(base, candidate.knowledge_item, product_name))
                key = (persona_name, question.lower())
                if key in seen:
                    continue
                seen.add(key)
                persona_questions.append(
                    PersonaQuestion(
                        persona_name=persona_name,
                        persona_style=config["style"],
                        question=question,
                        canonical_question=candidate.canonical_question,
                        knowledge_item=candidate.knowledge_item,
                    )
                )
    return persona_questions


class SemanticMatcher:
    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.enabled = bool(self.api_key)

    def similarity(self, expected: str, actual: str) -> float:
        if not self.enabled:
            return 0.0
        try:
            import requests  # local import to keep optional path self-contained

            payload = {
                "model": self.model,
                "input": [
                    {
                        "role": "system",
                        "content": (
                            "Return only a number between 0 and 1 representing semantic similarity "
                            "between the expected factual answer and the actual bot answer."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Expected: {expected}\nActual: {actual}",
                    },
                ],
            }
            response = requests.post(
                "https://api.openai.com/v1/responses",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=25,
            )
            response.raise_for_status()
            data = response.json()
            text = ""
            for item in data.get("output", []):
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        text += content.get("text", "")
            value = float(re.search(r"0(?:\.\d+)?|1(?:\.0+)?", text).group(0))
            return max(0.0, min(1.0, value))
        except Exception:
            return 0.0


def compare_numeric_values(expected_values: list[str], response_text: str) -> float:
    if not expected_values:
        return 0.0
    response_values = extract_numeric_values(response_text)
    if not response_values:
        return 0.0

    normalized_expected = {value.lower().replace(",", "") for value in expected_values}
    normalized_response = {value.lower().replace(",", "") for value in response_values}
    overlap = len(normalized_expected & normalized_response)
    if overlap == len(normalized_expected) and overlap > 0:
        return 1.0
    if overlap > 0:
        return overlap / len(normalized_expected)

    range_match_expected = RANGE_PATTERN.search(" ".join(expected_values))
    range_match_response = RANGE_PATTERN.search(response_text)
    if range_match_expected and range_match_response:
        exp_low, exp_high = float(range_match_expected.group(1)), float(range_match_expected.group(3))
        res_low, res_high = float(range_match_response.group(1)), float(range_match_response.group(3))
        if exp_low == res_low and exp_high == res_high:
            return 1.0
        if exp_low == res_low or exp_high == res_high:
            return 0.6
    return 0.0


REFUSAL_PATTERNS = [
    r"I (?:cannot|can't|am unable to)",
    r"I do not have that information",
    r"please visit",
    r"contact customer care",
    r"not available",
    r"sorry",
]


def evaluate_response(
    expected_item: KnowledgeItem,
    bot_response: str,
    semantic_matcher: Optional[SemanticMatcher] = None,
) -> EvaluationResult:
    """Evaluate bot response against extracted knowledge."""
    response = normalize_whitespace(bot_response)
    expected = normalize_whitespace(expected_item.text)
    refusal = any(re.search(pattern, response, re.IGNORECASE) for pattern in REFUSAL_PATTERNS)

    exact_match_score = similarity(expected, response)
    expected_tokens = tokenize(expected)
    response_tokens = tokenize(response)
    keyword_score = jaccard_score(expected_tokens, response_tokens)
    numeric_score = compare_numeric_values(expected_item.numeric_values, response)
    semantic_score = semantic_matcher.similarity(expected, response) if semantic_matcher else 0.0

    combined = (
        exact_match_score * 0.35
        + keyword_score * 0.25
        + numeric_score * 0.30
        + semantic_score * 0.10
    )

    if refusal and combined < 0.35:
        return EvaluationResult(
            result="Irrelevant",
            confidence_score=10,
            exact_match_score=exact_match_score,
            keyword_score=keyword_score,
            numeric_score=numeric_score,
            semantic_score=semantic_score,
            notes="Bot response appears to refuse or redirect instead of answering.",
        )

    if exact_match_score >= 0.85 or (numeric_score == 1.0 and keyword_score >= 0.45):
        result = "Correct"
        confidence = max(90, int(combined * 100))
        notes = "Expected values and supporting context are present."
    elif numeric_score >= 0.5 or combined >= 0.45:
        result = "Partially Correct"
        confidence = min(80, max(50, int(combined * 100)))
        notes = "Bot captured only part of the expected detail or omitted some qualifiers."
    elif combined >= 0.2:
        result = "Incorrect"
        confidence = min(30, max(15, int(combined * 100)))
        notes = "Response overlaps slightly with the topic but misses the expected answer."
    else:
        result = "Irrelevant"
        confidence = 0 if not response else 10
        notes = "Response is not meaningfully aligned with the expected information."

    return EvaluationResult(
        result=result,
        confidence_score=confidence,
        exact_match_score=round(exact_match_score, 3),
        keyword_score=round(keyword_score, 3),
        numeric_score=round(numeric_score, 3),
        semantic_score=round(semantic_score, 3),
        notes=notes,
    )


class BotTester:
    def __init__(self, headless: bool = False, slow_mo: int = 0) -> None:
        self.headless = headless
        self.slow_mo = slow_mo
        self.play = None
        self.browser = None
        self.page = None

    def __enter__(self) -> "BotTester":
        from playwright.sync_api import sync_playwright

        self.play = sync_playwright().start()
        self.browser = self.play.chromium.launch(headless=self.headless, slow_mo=self.slow_mo)
        context = self.browser.new_context(viewport={"width": 1440, "height": 960}, user_agent=USER_AGENT)
        self.page = context.new_page()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.browser:
            self.browser.close()
        if self.play:
            self.play.stop()

    def _locate_first(self, selectors: list[str], timeout_ms: int = 5000):
        assert self.page is not None
        for selector in selectors:
            locator = self.page.locator(selector)
            try:
                locator.first.wait_for(state="visible", timeout=timeout_ms)
                return locator.first
            except Exception:
                continue
        return None

    def login_bot(self, mobile_number: Optional[str] = None, bot_url: str = DEFAULT_BOT_URL) -> None:
        """Open Blu bot, handle onboarding, and wait for chat readiness."""
        assert self.page is not None
        self.page.goto(bot_url, wait_until="domcontentloaded", timeout=60_000)
        self.page.wait_for_timeout(3000)

        mobile_locator = self._locate_first([
            "input[type='tel']",
            "input[name*='mobile' i]",
            "input[placeholder*='mobile' i]",
            "input[placeholder*='phone' i]",
        ], timeout_ms=4000)
        if mobile_locator:
            if not mobile_number:
                mobile_number = input("Enter mobile number for Blu bot login: ").strip()
            mobile_locator.fill(mobile_number)

        checkbox = self._locate_first([
            "input[type='checkbox']",
            "label:has-text('Terms') input",
            "label:has-text('terms')",
        ], timeout_ms=2500)
        if checkbox:
            try:
                checkbox.check(force=True)
            except Exception:
                try:
                    checkbox.click(force=True)
                except Exception:
                    pass

        submit = self._locate_first([
            "button:has-text('Continue')",
            "button:has-text('Proceed')",
            "button:has-text('Verify')",
            "button:has-text('Submit')",
            "button[type='submit']",
        ], timeout_ms=2000)
        if submit:
            try:
                submit.click()
            except Exception:
                pass

        otp_locator = self._locate_first([
            "input[name*='otp' i]",
            "input[placeholder*='otp' i]",
            "input[inputmode='numeric']",
        ], timeout_ms=15_000)
        if otp_locator:
            otp = input("Enter OTP received for Blu bot login: ").strip()
            otp_locator.fill(otp)
            otp_submit = self._locate_first([
                "button:has-text('Verify')",
                "button:has-text('Continue')",
                "button:has-text('Submit')",
                "button[type='submit']",
            ], timeout_ms=3000)
            if otp_submit:
                try:
                    otp_submit.click()
                except Exception:
                    pass

        ready_locator = self._locate_first(BOT_READY_SELECTORS, timeout_ms=40_000)
        if not ready_locator:
            raise RuntimeError("Could not detect a ready chat input after onboarding.")

    def ask_bot(self, question: str, response_wait_ms: int = 12000, retries: int = 2) -> str:
        """Send question to bot and capture the latest bot response."""
        assert self.page is not None
        last_error: Optional[Exception] = None

        for _ in range(retries + 1):
            try:
                input_locator = self._locate_first(CHAT_INPUT_SELECTORS, timeout_ms=3000)
                if not input_locator:
                    raise RuntimeError("Chat input field not found.")

                previous_messages = self._collect_response_candidates()
                input_locator.click()
                try:
                    input_locator.fill(question)
                except Exception:
                    input_locator.press("Control+A")
                    input_locator.type(question, delay=15)

                send_button = self._locate_first(SEND_BUTTON_SELECTORS, timeout_ms=1500)
                if send_button:
                    try:
                        send_button.click()
                    except Exception:
                        input_locator.press("Enter")
                else:
                    input_locator.press("Enter")

                self.page.wait_for_timeout(response_wait_ms)
                response = self._get_new_response(previous_messages)
                if response:
                    return response
            except Exception as exc:
                last_error = exc
                self.page.wait_for_timeout(2000)
        raise RuntimeError(f"Unable to capture bot response. Last error: {last_error}")

    def _collect_response_candidates(self) -> list[str]:
        assert self.page is not None
        texts: list[str] = []
        for selector in BOT_RESPONSE_SELECTORS:
            locator = self.page.locator(selector)
            count = min(locator.count(), 30)
            for idx in range(count):
                try:
                    text = normalize_whitespace(locator.nth(idx).inner_text())
                    if len(text) > 2:
                        texts.append(text)
                except Exception:
                    continue
        return dedupe_preserve_order(texts)

    def _get_new_response(self, previous_messages: list[str]) -> str:
        current_messages = self._collect_response_candidates()
        previous_set = {message.lower() for message in previous_messages}
        new_messages = [message for message in current_messages if message.lower() not in previous_set]
        if new_messages:
            return new_messages[-1]
        return current_messages[-1] if current_messages else ""


ask_bot = BotTester.ask_bot
login_bot = BotTester.login_bot


def _build_summary_from_rows(rows: list[dict[str, Any]], source_url: str) -> dict[str, Any]:
    score_map = {"Correct": 100, "Partially Correct": 65, "Incorrect": 20, "Irrelevant": 0}
    enriched_rows = []
    for row in rows:
        cloned = dict(row)
        cloned["Source URL"] = source_url
        cloned["Assigned Score"] = score_map.get(cloned.get("Result"), cloned.get("Confidence Score", 0))
        enriched_rows.append(cloned)

    overall_accuracy = round(sum(row["Assigned Score"] for row in enriched_rows) / (len(enriched_rows) * 100) * 100, 2)

    persona_buckets: dict[str, list[float]] = defaultdict(list)
    category_buckets: dict[str, list[float]] = defaultdict(list)
    result_breakdown: Counter[str] = Counter()
    for row in enriched_rows:
        persona_buckets[str(row.get("Persona", ""))].append(float(row["Assigned Score"]))
        category_buckets[str(row.get("Category", ""))].append(float(row["Assigned Score"]))
        result_breakdown[str(row.get("Result", "Unknown"))] += 1

    summary = {
        "overall_accuracy_percent": overall_accuracy,
        "total_test_cases": int(len(enriched_rows)),
        "persona_summary": [
            {"Persona": persona, "Average Score": round(sum(scores) / len(scores), 2)}
            for persona, scores in sorted(persona_buckets.items())
        ],
        "category_summary": [
            {"Category": category, "Average Score": round(sum(scores) / len(scores), 2)}
            for category, scores in sorted(category_buckets.items())
        ],
        "result_breakdown": dict(result_breakdown),
        "source_domain": urlparse(source_url).netloc,
    }
    return {"rows": enriched_rows, "summary": summary}


def generate_report(
    rows: list[dict[str, Any]],
    output_prefix: str,
    source_url: str,
) -> tuple[Path, Optional[Path], dict[str, Any]]:
    """Generate detailed CSV/Excel reports plus aggregate summary metrics."""
    if not rows:
        raise ValueError("No test rows available for report generation.")

    packaged = _build_summary_from_rows(rows, source_url)
    enriched_rows = packaged["rows"]
    summary = packaged["summary"]

    output_dir = Path(output_prefix).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(f"{output_prefix}.csv").resolve()
    xlsx_path = Path(f"{output_prefix}.xlsx").resolve()
    summary_path = Path(f"{output_prefix}_summary.json").resolve()

    fieldnames = list(enriched_rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(enriched_rows)

    try:
        import pandas as pd

        df = pd.DataFrame(enriched_rows)
        persona_summary = pd.DataFrame(summary["persona_summary"])
        category_summary = pd.DataFrame(summary["category_summary"])
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Detailed Results", index=False)
            persona_summary.to_excel(writer, sheet_name="Persona Summary", index=False)
            category_summary.to_excel(writer, sheet_name="Category Summary", index=False)
            pd.DataFrame([summary]).to_excel(writer, sheet_name="Overview", index=False)
    except ModuleNotFoundError:
        xlsx_path = None

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return csv_path, xlsx_path, summary


def run_test_flow(
    product_name: str,
    source_url: str,
    bot_url: str = DEFAULT_BOT_URL,
    mobile_number: Optional[str] = None,
    headless: bool = False,
    max_facts: int = 25,
    max_questions_per_fact: int = 3,
    output_prefix: str = "reports/bot_evaluation_report",
) -> dict[str, Any]:
    raw_items, _html = scrape_full_content(source_url)
    knowledge_items = extract_structured_data(raw_items, product_name)
    question_candidates = generate_questions(
        knowledge_items,
        product_name=product_name,
        max_facts=max_facts,
        max_questions_per_fact=max_questions_per_fact,
    )
    persona_questions = apply_personas(question_candidates, product_name)
    semantic_matcher = SemanticMatcher()

    report_rows: list[dict[str, Any]] = []
    with BotTester(headless=headless) as tester:
        tester.login_bot(mobile_number=mobile_number, bot_url=bot_url)
        for persona_question in persona_questions:
            response = tester.ask_bot(persona_question.question)
            evaluation = evaluate_response(
                expected_item=persona_question.knowledge_item,
                bot_response=response,
                semantic_matcher=semantic_matcher,
            )
            report_rows.append(
                {
                    "Persona": persona_question.persona_name,
                    "Persona Style": persona_question.persona_style,
                    "Question": persona_question.question,
                    "Canonical Question": persona_question.canonical_question,
                    "Bot Response": response,
                    "Expected Data": persona_question.knowledge_item.text,
                    "Category": persona_question.knowledge_item.category,
                    "Result": evaluation.result,
                    "Confidence Score": evaluation.confidence_score,
                    "Exact Match Score": evaluation.exact_match_score,
                    "Keyword Score": evaluation.keyword_score,
                    "Numeric Score": evaluation.numeric_score,
                    "Semantic Score": evaluation.semantic_score,
                    "Notes": evaluation.notes,
                }
            )

    csv_path, xlsx_path, summary = generate_report(report_rows, output_prefix, source_url)
    return {
        "csv_path": str(csv_path),
        "xlsx_path": str(xlsx_path) if xlsx_path else None,
        "summary": summary,
        "raw_items_count": len(raw_items),
        "knowledge_items_count": len(knowledge_items),
        "question_candidates_count": len(question_candidates),
        "persona_questions_count": len(persona_questions),
    }


SAMPLE_RESPONSES = [
    (
        KnowledgeItem(
            category="rate",
            text="Interest rate ranges from 14% to 23% per annum.",
            source_type="paragraph",
            heading_context="Interest Rates",
            keywords=["interest", "rate", "14", "23", "annum"],
            numeric_values=["14%", "23%"],
        ),
        "The interest rate starts from 14% and can go up to 23% per annum depending on eligibility.",
    ),
    (
        KnowledgeItem(
            category="eligibility",
            text="Minimum age required is 24 years at the time of application.",
            source_type="list_item",
            heading_context="Eligibility",
            keywords=["minimum", "age", "24", "application"],
            numeric_values=["24 years"],
        ),
        "Applicants should be at least 24 years old when applying.",
    ),
    (
        KnowledgeItem(
            category="fee",
            text="Processing fee is up to 3.5% of the sanctioned amount.",
            source_type="table_row",
            heading_context="Fees and Charges",
            keywords=["processing", "fee", "3.5", "sanctioned", "amount"],
            numeric_values=["3.5%"],
        ),
        "I am sorry, please visit the website for fee details.",
    ),
]


def run_sample_execution() -> None:
    matcher = SemanticMatcher()
    rows = []
    for idx, (item, bot_response) in enumerate(SAMPLE_RESPONSES, start=1):
        evaluation = evaluate_response(item, bot_response, matcher)
        rows.append(
            {
                "Persona": "Sample Persona",
                "Persona Style": "Offline simulation",
                "Question": f"Sample question {idx}",
                "Canonical Question": f"Sample question {idx}",
                "Bot Response": bot_response,
                "Expected Data": item.text,
                "Category": item.category,
                "Result": evaluation.result,
                "Confidence Score": evaluation.confidence_score,
                "Exact Match Score": evaluation.exact_match_score,
                "Keyword Score": evaluation.keyword_score,
                "Numeric Score": evaluation.numeric_score,
                "Semantic Score": evaluation.semantic_score,
                "Notes": evaluation.notes,
            }
        )

    csv_path, xlsx_path, summary = generate_report(rows, "sample_report", "https://example.com/business-loan")
    print("Sample execution complete")
    print(json.dumps({"csv": str(csv_path), "xlsx": str(xlsx_path) if xlsx_path else None, "summary": summary}, indent=2))


sample_test_execution = run_sample_execution


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Intelligent Bajaj Finserv Blu bot testing system")
    parser.add_argument("--product", help="Product name to test, e.g. 'business loan'")
    parser.add_argument("--source-url", help="Source-of-truth webpage URL")
    parser.add_argument("--bot-url", default=DEFAULT_BOT_URL, help="Bajaj Blu bot URL")
    parser.add_argument("--mobile-number", help="Optional mobile number for login automation")
    parser.add_argument("--headless", action="store_true", help="Run Playwright in headless mode")
    parser.add_argument("--max-facts", type=int, default=25, help="Maximum extracted knowledge items to test")
    parser.add_argument(
        "--max-questions-per-fact",
        type=int,
        default=3,
        help="Maximum base question variants per knowledge item",
    )
    parser.add_argument(
        "--output-prefix",
        default="reports/bot_evaluation_report",
        help="Output path prefix for report files",
    )
    parser.add_argument(
        "--sample-run",
        action="store_true",
        help="Run offline sample evaluation/report generation without browser login",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.sample_run:
        run_sample_execution()
        return 0

    product = args.product or input("Enter product name: ").strip()
    source_url = args.source_url or input("Enter source-of-truth URL: ").strip()
    if not product or not source_url:
        print("Product name and source URL are required.", file=sys.stderr)
        return 1

    result = run_test_flow(
        product_name=product,
        source_url=source_url,
        bot_url=args.bot_url,
        mobile_number=args.mobile_number,
        headless=args.headless,
        max_facts=args.max_facts,
        max_questions_per_fact=args.max_questions_per_fact,
        output_prefix=args.output_prefix,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
