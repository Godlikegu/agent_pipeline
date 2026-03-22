"""
File-backed skill storage with Claude-compatible SKILL.md output.

Categories:
  - knowledge_general     : cross-task reusable patterns
  - knowledge_task_specific: per-task validated strategies or failure lessons
  - code                  : verified code snippets from cleaned repositories
"""
from __future__ import annotations

import json
import shutil
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


DEFAULT_SCORE = 1.0
VALID_CATEGORIES = {"knowledge_general", "knowledge_task_specific", "code"}


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
    """Simplified skill record — one `instructions` field instead of many lists."""
    id: str
    slug: str
    title: str
    description: str
    category: str                             # knowledge_general | knowledge_task_specific | code
    status: str = "draft"                     # active | draft
    agent_scope: str = "General"              # General | Planner | Architect | Coder | Judge

    instructions: str = ""                    # Full markdown body (Claude SKILL.md content)

    tags: List[str] = field(default_factory=list)
    score: float = DEFAULT_SCORE
    usage_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    corroboration_count: int = 0
    source_tasks: List[str] = field(default_factory=list)
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
        raw = json.loads(self.registry_path.read_text(encoding="utf-8") or "[]")
        return [SkillRecord.from_dict(item) for item in raw]

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

    def get_by_ids(self, ids: List[str]) -> List[SkillRecord]:
        target = set(ids)
        return [r for r in self._read_registry() if r.id in target]

    def find_by_fingerprint(self, fingerprint: str, category: str) -> Optional[SkillRecord]:
        for r in self._read_registry():
            if r.category == category and r.fingerprint == fingerprint:
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

    def promote(self, record_id: str) -> Optional[SkillRecord]:
        records = self._read_registry()
        target: Optional[SkillRecord] = None
        for i, r in enumerate(records):
            if r.id == record_id:
                r.status = "active"
                r.updated_at = now_ts()
                records[i] = r
                target = r
                break
        if target:
            self._write_registry(records)
            self._write_skill_file(target)
        return target

    def record_feedback(
        self,
        knowledge_ids: List[str],
        *,
        success: bool,
        success_delta: float,
        failure_delta: float,
    ) -> None:
        if not knowledge_ids:
            return
        target = set(knowledge_ids)
        records = self._read_registry()
        changed = False
        for r in records:
            if r.id not in target:
                continue
            r.usage_count += 1
            if success:
                r.success_count += 1
                r.score += success_delta
            else:
                r.failure_count += 1
                r.score += failure_delta
            r.updated_at = now_ts()
            changed = True
        if changed:
            self._write_registry(records)

    def add_code_snippet(self, slug: str, content: str) -> str:
        path = self.code_pool_dir / f"{slug}.py"
        path.write_text(content.strip() + "\n", encoding="utf-8")
        return str(path)

    # ---- SKILL.md rendering ----
    def _write_skill_file(self, record: SkillRecord) -> None:
        base = self.active_dir if record.status == "active" else self.draft_dir
        skill_dir = base / record.slug
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(self.render_markdown(record), encoding="utf-8")

        if record.status == "active":
            draft = self.draft_dir / record.slug
            if draft.exists() and draft != skill_dir:
                shutil.rmtree(draft)
        else:
            active = self.active_dir / record.slug
            if active.exists():
                shutil.rmtree(active)

    def render_markdown(self, record: SkillRecord) -> str:
        frontmatter = {
            "name": record.slug,
            "description": record.description,
            "category": record.category,
            "agent-scope": record.agent_scope,
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
            "status": record.status,
            "agent_scope": record.agent_scope,
            "instructions": record.instructions,
            "code_snippet_path": record.code_snippet_path,
            "tags": list(record.tags),
            "credit_score": record.score,
        }
