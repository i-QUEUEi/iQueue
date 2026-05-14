import re
from pathlib import Path

# Mapping of 3-letter month abbreviations to month numbers
# Used to parse holiday calendar entries like "Jan 1 : New Year's Day"
MONTH_MAP = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


def load_ph_holiday_month_days(calendar_path: Path):
    """Parse a Philippine holiday calendar CSV and return a set of (month, day) tuples.

    The function reads a text file where holidays are listed as lines like:
        "Jan 1 : New Year's Day"
    It uses a regex to extract the month abbreviation and day number,
    then converts them to (month_number, day_number) tuples stored in a set.

    Using a set ensures O(1) lookup time — checking "is this date a holiday?"
    is instant regardless of how many holidays exist.

    Args:
        calendar_path: Path to the holiday calendar CSV file.

    Returns:
        A set of (month, day) tuples, e.g., {(1, 1), (6, 12), (12, 25), ...}.
        Returns an empty set if the file doesn't exist.
    """
    holiday_md = set()

    # If the calendar file is missing, return empty set (no holidays)
    if not calendar_path.exists():
        return holiday_md

    # Read the entire file as plain text
    text = calendar_path.read_text(encoding="utf-8", errors="ignore")

    # Process each line looking for holiday entries
    for line in text.splitlines():
        # Regex breakdown:
        #   \b([A-Za-z]{3})  → capture exactly 3 letters (month abbreviation like "Jan")
        #   \s+(\d{1,2})     → one or more spaces, then 1-2 digit day number
        #   \s*:\s*           → optional spaces around a colon (separator before holiday name)
        match = re.search(r"\b([A-Za-z]{3})\s+(\d{1,2})\s*:\s*", line)
        if not match:
            continue  # Skip lines that don't match the holiday format

        # Extract and convert the month and day
        month_name = match.group(1).title()   # Normalize case: "jan" → "Jan"
        day = int(match.group(2))             # Convert string to integer: "1" → 1
        month = MONTH_MAP.get(month_name)     # Look up month number: "Jan" → 1

        # Only add if the month abbreviation was recognized
        if month:
            holiday_md.add((month, day))       # Store as tuple: (1, 1) for January 1st

    return holiday_md
