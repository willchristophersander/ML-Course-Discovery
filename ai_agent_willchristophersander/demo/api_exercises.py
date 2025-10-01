"""Part 2 demo exercises for working with public APIs."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import requests


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("api_exercises")
CAT_FACTS_ENDPOINT = "https://catfact.ninja/fact"
HOLIDAYS_ENDPOINT = "https://date.nager.at/api/v3/PublicHolidays/{year}/{country}"


def fetch_cat_facts(count: int = 5, *, timeout: int = 10) -> List[str]:
    """Fetch unique cat facts from the public API."""

    facts: List[str] = []
    attempts = 0
    while len(facts) < count and attempts < count * 3:
        attempts += 1
        try:
            response = requests.get(CAT_FACTS_ENDPOINT, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            fact = payload.get("fact")
            if fact and fact not in facts:
                facts.append(fact)
        except requests.RequestException as exc:
            LOGGER.warning("Cat fact request failed (%s/%s): %s", attempts, count * 3, exc)
    if len(facts) < count:
        raise RuntimeError(f"Only gathered {len(facts)} facts after {attempts} attempts")
    return facts


def save_facts_to_json(facts: List[str], output_path: Path) -> None:
    """Persist facts to disk with timestamps."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "facts": facts,
        "count": len(facts),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def fetch_public_holidays(country: str, year: int) -> List[Dict[str, str]]:
    """Fetch public holidays for a country and year."""

    url = HOLIDAYS_ENDPOINT.format(country=country.upper(), year=year)
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected holidays response")
    return payload


def summarise_holiday_counts(countries: List[str], year: int) -> Dict[str, int]:
    """Return a dictionary mapping country codes to holiday counts."""

    summary: Dict[str, int] = {}
    for country in countries:
        try:
            holidays = fetch_public_holidays(country, year)
        except requests.RequestException as exc:
            LOGGER.error("Failed to fetch holidays for %s: %s", country, exc)
            summary[country.upper()] = 0
            continue
        summary[country.upper()] = len(holidays)
        LOGGER.info("%s has %s public holidays in %s", country.upper(), len(holidays), year)
    return summary


def main() -> None:
    """Entry point for manual execution."""

    try:
        facts = fetch_cat_facts()
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.error("Unable to fetch cat facts: %s", exc)
        return
    target_path = Path(__file__).resolve().parent / "cat_facts.json"
    save_facts_to_json(facts, target_path)

    countries = ["US", "GB", "CA"]
    try:
        holiday_summary = summarise_holiday_counts(countries, year=datetime.now().year)
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.error("Unable to summarise holidays: %s", exc)
        return
    LOGGER.info("Holiday summary: %s", holiday_summary)


if __name__ == "__main__":
    from datetime import datetime

    main()
