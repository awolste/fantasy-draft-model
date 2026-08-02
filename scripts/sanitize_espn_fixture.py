"""Produce a committable ESPN fixture with all personal data removed.

The raw ESPN payload identifies real people: every league member's ESPN
account GUID (which for the authenticated user is the SWID cookie itself),
their real first and last names, their ESPN usernames, and their team names.
None of that belongs in git.

This script reads the gitignored raw cache and writes a structurally
identical fixture with every identifier replaced by a deterministic fake, so
parser tests still exercise real response shape, pick counts, and internal
consistency without carrying anyone's identity.

Usage:
    .venv/bin/python scripts/sanitize_espn_fixture.py <raw.json> <out.json>
"""

import json
import re
import sys
from pathlib import Path

GUID_PATTERN = re.compile(r"\{[0-9A-Fa-f-]{30,40}\}")


def _fake_guid(index: int) -> str:
    """Deterministic placeholder preserving ESPN's GUID shape."""
    return f"{{{index:08X}-0000-0000-0000-000000000000}}"


def sanitize(payload: dict) -> dict:
    text = json.dumps(payload)

    # Map every distinct GUID to a stable fake, in sorted order so repeated
    # runs produce identical output.
    guids = sorted(set(GUID_PATTERN.findall(text)))
    for index, guid in enumerate(guids, start=1):
        text = text.replace(guid, _fake_guid(index))

    data = json.loads(text)

    for index, member in enumerate(data.get("members", []), start=1):
        member["firstName"] = f"Manager{index}"
        member["lastName"] = "Anonymized"
        member["displayName"] = f"manager{index}"

    for index, team in enumerate(data.get("teams", []), start=1):
        if "name" in team:
            team["name"] = f"Team {index}"
        if "location" in team:
            team["location"] = "Team"
        if "nickname" in team:
            team["nickname"] = str(index)
        team.pop("logo", None)

    return data


def assert_clean(data: dict, forbidden: list[str]) -> None:
    """Fail loudly if any known-sensitive string survived."""
    text = json.dumps(data)
    leaked = [value for value in forbidden if value and value in text]
    if leaked:
        raise SystemExit(f"Sanitization failed; {len(leaked)} sensitive value(s) still present.")
    remaining = [g for g in GUID_PATTERN.findall(text) if not g.endswith("-000000000000}")]
    if remaining:
        raise SystemExit(f"Sanitization failed; unreplaced GUIDs remain: {len(remaining)}")


def main() -> None:
    raw_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    raw = json.loads(raw_path.read_text())

    original_text = json.dumps(raw)
    forbidden = sorted(set(GUID_PATTERN.findall(original_text)))
    for member in raw.get("members", []):
        forbidden += [member.get("firstName"), member.get("lastName"), member.get("displayName")]
    for team in raw.get("teams", []):
        forbidden += [team.get("name"), team.get("nickname")]

    clean = sanitize(raw)
    assert_clean(clean, [f for f in forbidden if f and len(f) > 3])
    out_path.write_text(json.dumps(clean))

    picks = len(clean.get("draftDetail", {}).get("picks", []))
    print(f"wrote {out_path} — teams={len(clean.get('teams', []))} picks={picks}")


if __name__ == "__main__":
    main()
