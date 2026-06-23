"""
Summary mutation utilities.
"""

import re


def move_active_to_resolved_in_summary(chat_summary: str) -> str:
    """
    Mutate a markdown chat summary to move all 'Active Issues' to 'Resolved Issues'.

    Called when a conversation is resolved by a human support agent.  Clears the
    active issues section, merges those issues into resolved, and sets
    `Escalate to Human` to False.

    Only operates on summaries that contain both `### Active Issues` and
    `### Resolved Issues` sections.  Returns the original string unchanged if
    the expected structure is not found.

    Args:
        chat_summary: The markdown chat summary string from the graph state.

    Returns:
        The updated summary string, or the original if it cannot be parsed.
    """
    if not chat_summary:
        return chat_summary

    # ── 1. Parse section headers and their content lines ─────────────────────
    lines = chat_summary.split("\n")
    sections: dict = {}
    current_section = None

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("### "):
            current_section = stripped[4:].strip()
            sections[current_section] = []
        elif current_section is not None and stripped:
            sections[current_section].append(stripped)

    # ── 2. Guard: both sections must exist ────────────────────────────────────
    if "Active Issues" not in sections or "Resolved Issues" not in sections:
        return chat_summary

    # ── 3. Extract non-empty active issues ────────────────────────────────────
    active_issues = [
        l for l in sections["Active Issues"] if l.startswith("-") and l != "- None"
    ]
    if not active_issues:
        return chat_summary

    # ── 4. Strip the "[Turns Active: N]" prefix before moving to resolved ─────
    cleaned_active = [
        re.sub(r"^-\s*\[Turns Active:\s*\d+\]\s*", "- ", issue)
        for issue in active_issues
    ]

    # ── 5. Merge into resolved issues ─────────────────────────────────────────
    resolved_issues = [
        l for l in sections["Resolved Issues"] if l.startswith("-") and l != "- None"
    ]
    sections["Active Issues"] = ["- None"]
    sections["Resolved Issues"] = resolved_issues + cleaned_active
    if "Escalate to Human" in sections:
        sections["Escalate to Human"] = ["- False"]

    # ── 6. Reconstruct the summary string ─────────────────────────────────────
    new_lines = []
    for sec_name, sec_lines in sections.items():
        new_lines.append(f"### {sec_name}")
        new_lines.extend(sec_lines)
        new_lines.append("")  # blank line between sections

    return "\n".join(new_lines).strip()
