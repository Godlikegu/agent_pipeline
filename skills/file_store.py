"""
File-backed skill storage with Claude-compatible SKILL.md output.

Categories:
  - knowledge_general     : cross-task reusable patterns
  - knowledge_task_specific: per-task validated strategies or failure lessons
  - code                  : verified code snippets from cleaned repositories

Tiers:
  - draft     : newly distilled, not yet validated
  - permanent : promoted after successful task usage
"""
from __future__ import annotations

import json
import logging
import shutil
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

VALID_CATEGORIES = {"knowledge_general", "knowledge_task_specific", "code"}
VALID_TIERS = {"draft", "permanent"}
VALID_SCOPES = {"General", "Planner", "Coder"}


def slugify(value: str) -> str:
    text = (value or "").strip().lower()
    allowed: list[str] = []
    last_dash = False
    for char in text:
        if char.isalnum():
            allowed.append(char)
            last_dash = False
        else:
            if not last_dash:
                allowed.append("-")
                last_dash = True
    slug = "".join(allowed).strip("-")
    return slug or f"skill-{uuid.uuid4().hex[:8]}"


def now_ts() -> int:
    return int(time.time())


@dataclass
class SkillRecord:
    """Skill record with draft/permanent tier and Planner/Coder/General scope."""
    id: str
    slug: str
    title: str
    description: str
    category: str                             # knowledge_general | knowledge_task_specific | code
    tier: str = "draft"                       # draft | permanent
    scope: str = "General"                    # General | Planner | Coder
    status: str = "active"                    # active | archived (for soft-delete)

    instructions: str = ""                    # Full markdown body (Claude SKILL.md content)

    tags: List[str] = field(default_factory=list)
    source_tasks: List[str] = field(default_factory=list)
    task_origin: str = ""                     # The task that first created this skill
    code_snippet_path: Optional[str] = None
    fingerprint: str = ""

    created_at: int = field(default_factory=now_ts)
    updated_at: int = field(default_factory=now_ts)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillRecord":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


def _migrate_record(data: dict) -> dict:
    """Migrate old-format records (with score/agent_scope/etc.) to new format."""
    # --- tier migration: old 'status' mapped to 'tier' ---
    if "tier" not in data:
        old_status = data.get("status", "draft")
        if old_status == "active":
            data["tier"] = "permanent"
        else:
            data["tier"] = "draft"
        data["status"] = "active"

    # --- scope migration: old 'agent_scope' mapped to 'scope' ---
    if "scope" not in data and "agent_scope" in data:
        old_scope = data.pop("agent_scope", "General")
        if old_scope in VALID_SCOPES:
            data["scope"] = old_scope
        else:
            data["scope"] = "General"

    # --- remove deprecated fields ---
    for key in ("score", "usage_count", "success_count", "failure_count",
                "corroboration_count", "agent_scope"):
        data.pop(key, None)

    # --- ensure task_origin ---
    if "task_origin" not in data or not data["task_origin"]:
        sources = data.get("source_tasks", [])
        data["task_origin"] = sources[0] if sources else ""

    return data


class FileSkillStore:
    """Registry-backed store that writes Claude-compatible SKILL.md files."""

    def __init__(
        self,
        active_dir: str,
        draft_dir: str,
        registry_path: str,
        code_pool_dir: str,
    ) -> None:
        self.active_dir = Path(active_dir).expanduser().resolve()
        self.draft_dir = Path(draft_dir).expanduser().resolve()
        self.registry_path = Path(registry_path).expanduser().resolve()
        self.code_pool_dir = Path(code_pool_dir).expanduser().resolve()
        for d in (self.active_dir, self.draft_dir, self.code_pool_dir, self.registry_path.parent):
            d.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self._write_registry([])

    # ---- registry I/O ----
    def _read_registry(self) -> List[SkillRecord]:
        if not self.registry_path.exists():
            return []
        try:
            raw = json.loads(self.registry_path.read_text(encoding="utf-8") or "[]")
        except (json.JSONDecodeError, OSError):
            logger.warning("Failed to read registry, returning empty list")
            return []
        migrated = [_migrate_record(item) for item in raw]
        return [SkillRecord.from_dict(item) for item in migrated]

    def _write_registry(self, items: List[SkillRecord]) -> None:
        self.registry_path.write_text(
            json.dumps([item.to_dict() for item in items], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ---- queries ----
    def list_records(self, *, status: Optional[str] = None, category: Optional[str] = None) -> List[SkillRecord]:
        return [
            r for r in self._read_registry()
            if (status is None or r.status == status)
            and (category is None or r.category == category)
        ]

    def list_records_by_tier(self, tier: str) -> List[SkillRecord]:
        """List active records filtered by tier (draft or permanent)."""
        return [r for r in self._read_registry() if r.tier == tier and r.status == "active"]

    def list_draft_records_for_task(self, task_origin: str) -> List[SkillRecord]:
        """List all active draft records originating from a specific task."""
        return [
            r for r in self._read_registry()
            if r.tier == "draft" and r.task_origin == task_origin and r.status == "active"
        ]

    def get_by_ids(self, ids: List[str]) -> List[SkillRecord]:
        target = set(ids)
        return [r for r in self._read_registry() if r.id in target]

    def find_by_fingerprint(self, fingerprint: str, category: str) -> Optional[SkillRecord]:
        for r in self._read_registry():
            if r.category == category and r.fingerprint == fingerprint and r.status == "active":
                return r
        return None

    # ---- mutations ----
    def upsert(self, record: SkillRecord) -> SkillRecord:
        records = self._read_registry()
        updated = False
        for i, existing in enumerate(records):
            if existing.id == record.id:
                record.created_at = existing.created_at
                record.updated_at = now_ts()
                records[i] = record
                updated = True
                break
        if not updated:
            record.updated_at = now_ts()
            records.append(record)
        self._write_registry(records)
        self._write_skill_file(record)
        return record

    def promote_to_permanent(self, record_id: str) -> Optional[SkillRecord]:
        """Promote a draft skill to permanent tier."""
        records = self._read_registry()
        target: Optional[SkillRecord] = None
        for i, r in enumerate(records):
            if r.id == record_id:
                r.tier = "permanent"
                r.updated_at = now_ts()
                records[i] = r
                target = r
                break
        if target:
            self._write_registry(records)
            self._write_skill_file(target)
        return target

    def delete_draft_skills_for_task(self, task_origin: str, exclude_ids: set = None) -> int:
        """Archive all remaining draft skills for a task. Returns count deleted.

        Args:
            task_origin: Task name whose drafts to clean up.
            exclude_ids: Set of skill IDs to exclude from deletion (e.g., newly distilled).
        """
        exclude = exclude_ids or set()
        records = self._read_registry()
        count = 0
        for r in records:
            if (r.tier == "draft" and r.task_origin == task_origin
                    and r.status == "active" and r.id not in exclude):
                r.status = "archived"
                r.updated_at = now_ts()
                count += 1
        if count:
            self._write_registry(records)
        return count

    def overwrite_record(self, record_id: str, updated: SkillRecord) -> SkillRecord:
        """Overwrite the content of an existing record with merged content."""
        records = self._read_registry()
        for i, r in enumerate(records):
            if r.id == record_id:
                updated.id = r.id
                updated.created_at = r.created_at
                updated.updated_at = now_ts()
                records[i] = updated
                self._write_registry(records)
                self._write_skill_file(updated)
                return updated
        raise ValueError(f"Record {record_id} not found")

    def add_code_snippet(self, slug: str, content: str) -> str:
        path = self.code_pool_dir / f"{slug}.py"
        path.write_text(content.strip() + "\n", encoding="utf-8")
        return str(path)

    # ---- SKILL.md rendering ----
    def _write_skill_file(self, record: SkillRecord) -> None:
        if record.status == "archived":
            # Remove files for archived records
            for base in (self.active_dir, self.draft_dir):
                target = base / record.slug
                if target.exists():
                    shutil.rmtree(target)
            return

        base = self.active_dir if record.tier == "permanent" else self.draft_dir
        skill_dir = base / record.slug
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(self.render_markdown(record), encoding="utf-8")

        # Clean up the other tier's directory if it exists
        other_base = self.draft_dir if record.tier == "permanent" else self.active_dir
        other_dir = other_base / record.slug
        if other_dir.exists() and other_dir != skill_dir:
            shutil.rmtree(other_dir)

    def render_markdown(self, record: SkillRecord) -> str:
        frontmatter = {
            "name": record.slug,
            "description": record.description,
            "category": record.category,
            "scope": record.scope,
            "tier": record.tier,
            "user-invocable": False,
            "disable-model-invocation": False,
        }
        lines = [
            "---",
            yaml.safe_dump(frontmatter, sort_keys=False).strip(),
            "---",
            "",
            f"# {record.title}",
            "",
        ]
        if record.instructions:
            lines.append(record.instructions.strip())
            lines.append("")
        if record.code_snippet_path:
            lines.append(f"**Code reference**: `{record.code_snippet_path}`")
            lines.append("")
        return "\n".join(lines)

    def export_prompt_payload(self, record: SkillRecord) -> Dict[str, Any]:
        return {
            "id": record.id,
            "name": record.title,
            "category": record.category,
            "tier": record.tier,
            "scope": record.scope,
            "instructions": record.instructions,
            "code_snippet_path": record.code_snippet_path,
            "tags": list(record.tags),
        }
