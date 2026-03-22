"""
File-backed skill manager.

Skill categories:
  - knowledge_general      – reusable cross-task patterns (promoted after N corroborations)
  - knowledge_task_specific – per-task validated strategies or failure lessons
  - code                   – verified code snippets linked to the code pool
"""
from __future__ import annotations

import hashlib
import logging
import math
import random
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from .file_store import DEFAULT_SCORE, FileSkillStore, SkillRecord, slugify

logger = logging.getLogger(__name__)


class FileSkillManager:
    """Workflow-facing API: retrieve, format, distill, score."""

    def __init__(
        self,
        client: Optional[Any] = None,
        model_name: str = "",
        config: Optional[dict] = None,
    ) -> None:
        cfg = config or {}
        paths_cfg = cfg.get("_paths", {})
        retrieval_cfg = cfg.get("retrieval", {})
        credit_cfg = cfg.get("credit", {})
        embedding_cfg = cfg.get("embedding", {})

        self.client = client
        self.model_name = model_name

        self.max_items = retrieval_cfg.get("max_items", 5)
        self.similarity_threshold = retrieval_cfg.get("similarity_threshold", 0.3)
        self.max_token_budget = retrieval_cfg.get("max_token_budget", 2000)
        self.rank_sim_w = retrieval_cfg.get("rank_weights", {}).get("similarity", 0.7)
        self.rank_credit_w = retrieval_cfg.get("rank_weights", {}).get("credit", 0.3)

        self.credit_initial = credit_cfg.get("initial_score", DEFAULT_SCORE)
        self.credit_success_delta = credit_cfg.get("success_delta", 0.1)
        self.credit_failure_delta = credit_cfg.get("failure_delta", -0.2)
        self.general_promotion_threshold = cfg.get("general_promotion_threshold", 2)

        root = Path(__file__).resolve().parent.parent
        self.store = FileSkillStore(
            active_dir=paths_cfg.get("active_dir", str(root / ".claude" / "skills")),
            draft_dir=paths_cfg.get("draft_dir", str(root / "skills" / "library" / "drafts")),
            registry_path=paths_cfg.get("registry", str(root / "skills" / "library" / "registry.json")),
            code_pool_dir=paths_cfg.get("code_pool", str(root / "skills" / "library" / "code_pool")),
        )

        self.embedder = None
        self.target_dim = embedding_cfg.get("dimension", 384)
        model_dir = Path(
            paths_cfg.get(
                "embedding_model_dir",
                root / "skills" / "embeddings" / embedding_cfg.get("model_name", "all-MiniLM-L6-v2"),
            )
        )
        if SentenceTransformer is not None and model_dir.exists():
            try:
                self.embedder = SentenceTransformer(str(model_dir))
            except Exception as exc:
                logger.warning("Failed to load embedder: %s", exc)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve_knowledge(
        self,
        task_desc: str,
        agent_role: str = "General",
        top_k: int = 3,
    ) -> Dict[str, List[Dict]]:
        records = self.store.list_records(status="active")
        if not records:
            return {"knowledge_general": [], "knowledge_task_specific": [], "code": []}

        filtered = [
            r for r in records
            if r.agent_scope in {"General", agent_role} or agent_role == "General"
        ] or records

        query = (task_desc or "").strip()
        texts = [self._record_text(r) for r in filtered]
        tfidf = self._tfidf_similarity(query, texts) if query else [0.0] * len(filtered)
        embed = self._embed_similarity(query, filtered) if query else [0.0] * len(filtered)

        ranked: list[tuple[float, SkillRecord]] = []
        for i, r in enumerate(filtered):
            sim = max(tfidf[i], embed[i])
            credit = max(0.0, min(2.0, r.score)) / 2.0
            score = sim * self.rank_sim_w + credit * self.rank_credit_w
            if score >= self.similarity_threshold:
                ranked.append((score, r))
        ranked.sort(key=lambda x: x[0], reverse=True)

        limits = {
            "knowledge_general": 2,
            "knowledge_task_specific": max(1, min(top_k, self.max_items)),
            "code": 2 if agent_role in {"Coder", "Architect", "General"} else 0,
        }
        results: Dict[str, List[Dict]] = {k: [] for k in limits}
        budget = self.max_token_budget

        for _, r in ranked:
            cat = r.category
            if limits.get(cat, 0) <= 0:
                continue
            payload = self.store.export_prompt_payload(r)
            est = max(60, len(payload.get("instructions", "")) // 4)
            if budget - est < 0:
                continue
            budget -= est
            results[cat].append(payload)
            limits[cat] -= 1

        return results

    def format_knowledge_for_prompt(self, knowledge: Dict[str, List[Dict]]) -> str:
        if not any(knowledge.values()):
            return ""
        sections: list[str] = []

        general = knowledge.get("knowledge_general", [])
        if general:
            lines = ["### General Knowledge Skills"]
            for i, item in enumerate(general, 1):
                lines.append(f"\n**{i}. {item['name']}**\n")
                lines.append(item.get("instructions", "").strip())
            sections.append("\n".join(lines))

        specific = knowledge.get("knowledge_task_specific", [])
        if specific:
            lines = ["### Task-Specific Knowledge Skills"]
            for i, item in enumerate(specific, 1):
                lines.append(f"\n**{i}. {item['name']}**\n")
                lines.append(item.get("instructions", "").strip())
            sections.append("\n".join(lines))

        code = knowledge.get("code", [])
        if code:
            lines = ["### Validated Code Skills"]
            for i, item in enumerate(code, 1):
                lines.append(f"\n**{i}. {item['name']}**\n")
                lines.append(item.get("instructions", "").strip())
                if item.get("code_snippet_path"):
                    lines.append(f"\nCode reference: `{item['code_snippet_path']}`")
            sections.append("\n".join(lines))

        return "\n\n".join(sections) + "\n"

    def get_knowledge_details(self, knowledge_ids: List[str]) -> List[Dict]:
        return [self.store.export_prompt_payload(r) for r in self.store.get_by_ids(knowledge_ids)]

    def update_scores(self, knowledge_ids: List[str], success: bool) -> None:
        self.store.record_feedback(
            knowledge_ids,
            success=success,
            success_delta=self.credit_success_delta,
            failure_delta=self.credit_failure_delta,
        )
        self._promote_ready_drafts()

    # ------------------------------------------------------------------
    # Distillation — extract skills from completed trajectories
    # ------------------------------------------------------------------
    def distill_and_store(self, trajectory: Dict[str, Any]) -> Dict[str, int]:
        candidates = self._extract_candidates(trajectory)
        stats = {"knowledge_general": 0, "knowledge_task_specific": 0, "code": 0}
        for c in candidates:
            cat = c["category"]
            existing = self.store.find_by_fingerprint(c["fingerprint"], cat)
            if existing:
                record = self._merge(existing, c)
            else:
                record = SkillRecord(
                    id=c["id"],
                    slug=c["slug"],
                    title=c["title"],
                    description=c["description"],
                    category=cat,
                    status=c["status"],
                    agent_scope=c["agent_scope"],
                    instructions=c["instructions"],
                    tags=c["tags"],
                    score=self.credit_initial,
                    source_tasks=c["source_tasks"],
                    corroboration_count=c.get("corroboration_count", 1),
                    fingerprint=c["fingerprint"],
                )
            self.store.upsert(record)
            if record.status == "active":
                stats[cat] = stats.get(cat, 0) + 1
        self._promote_ready_drafts()
        return stats

    def store_code_skill(
        self,
        *,
        title: str,
        description: str,
        instructions: str,
        code_snippet: str,
        repo_name: str,
    ) -> SkillRecord:
        slug = slugify(title)
        snippet_path = self.store.add_code_snippet(slug, code_snippet)
        fp = hashlib.sha256(f"{repo_name}:{title}".encode()).hexdigest()
        existing = self.store.find_by_fingerprint(fp, "code")
        if existing:
            existing.description = description
            existing.instructions = instructions
            existing.code_snippet_path = snippet_path
            existing.status = "active"
            record = existing
        else:
            record = SkillRecord(
                id=f"code-{uuid.uuid4().hex[:12]}",
                slug=slug,
                title=title,
                description=description,
                category="code",
                status="active",
                agent_scope="Coder",
                instructions=instructions,
                tags=["code", slugify(repo_name)],
                source_tasks=[repo_name],
                code_snippet_path=snippet_path,
                corroboration_count=1,
                fingerprint=fp,
            )
        self.store.upsert(record)
        return record

    # ------------------------------------------------------------------
    # Candidate extraction from trajectories
    # ------------------------------------------------------------------
    def _extract_candidates(self, trajectory: Dict[str, Any]) -> List[Dict[str, Any]]:
        task_name = trajectory.get("task_name", "unknown")
        task_desc = trajectory.get("task_desc", "")
        plan = trajectory.get("final_plan", "")
        outcome = trajectory.get("outcome", "failure")
        metrics = trajectory.get("final_reward") or {}
        candidates: list[dict] = []

        if outcome == "success":
            instructions = self._build_success_instructions(task_name, plan, metrics)
            candidates.append({
                "id": f"ts-{uuid.uuid4().hex[:12]}",
                "slug": slugify(f"validated-{task_name}"),
                "title": f"Validated strategy for {task_name}",
                "description": f"Task-specific validated strategy from successful run of {task_name}.",
                "category": "knowledge_task_specific",
                "status": "active",
                "agent_scope": "General",
                "instructions": instructions,
                "tags": ["task-specific", slugify(task_name)],
                "source_tasks": [task_name],
                "corroboration_count": 1,
                "fingerprint": f"task-specific:{slugify(task_name)}",
            })

            algo = self._extract_algorithm(f"{task_desc} {plan}")
            if algo:
                candidates.append({
                    "id": f"kg-{uuid.uuid4().hex[:12]}",
                    "slug": slugify(f"{algo}-general"),
                    "title": f"General pattern: {algo}",
                    "description": f"Reusable guidance for {algo}-style workflows.",
                    "category": "knowledge_general",
                    "status": "draft",
                    "agent_scope": "General",
                    "instructions": self._build_general_instructions(algo, plan),
                    "tags": ["general", algo],
                    "source_tasks": [task_name],
                    "corroboration_count": 1,
                    "fingerprint": f"general:{algo}",
                })
        else:
            failure = self._extract_failure(trajectory)
            if failure:
                candidates.append({
                    "id": f"fl-{uuid.uuid4().hex[:12]}",
                    "slug": slugify(f"avoid-{failure['ticket']}-{task_name}"),
                    "title": f"Failure lesson: {failure['ticket']} in {task_name}",
                    "description": f"Draft failure lesson for {failure['ticket']} issues.",
                    "category": "knowledge_task_specific",
                    "status": "draft",
                    "agent_scope": failure["ticket"],
                    "instructions": failure["instructions"],
                    "tags": ["failure-draft", slugify(failure["ticket"])],
                    "source_tasks": [task_name],
                    "corroboration_count": 1,
                    "fingerprint": f"failure:{slugify(failure['ticket'])}:{slugify(task_name)}",
                })
        return candidates

    def _build_success_instructions(self, task_name: str, plan: str, metrics: Any) -> str:
        lines = [
            "## When to use",
            f"- Task closely matches `{task_name}` or its assets.",
            "",
            "## Key strategy",
        ]
        for line in (plan or "").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and len(stripped) > 10:
                lines.append(f"- {stripped[:180]}")
                if len(lines) > 8:
                    break
        if metrics:
            lines.extend(["", f"## Validated metrics", f"- {metrics}"])
        lines.extend([
            "",
            "## Constraints",
            "- Adapt constants and file paths to the current task.",
            "- Prefer the validated plan unless Judge feedback proves regression.",
        ])
        return "\n".join(lines)

    def _build_general_instructions(self, algo: str, plan: str) -> str:
        lines = [
            "## When to use",
            f"- Plan relies on `{algo}` or a closely related method.",
            "",
            "## Guidance",
        ]
        for line in (plan or "").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and len(stripped) > 10:
                lines.append(f"- {stripped[:180]}")
                if len(lines) > 7:
                    break
        lines.extend([
            "",
            "## Constraints",
            "- Only apply when the forward model and data assumptions match.",
            "- Keep as draft until multiple successful trajectories corroborate.",
        ])
        return "\n".join(lines)

    def _merge(self, existing: SkillRecord, candidate: Dict[str, Any]) -> SkillRecord:
        existing.instructions = candidate.get("instructions") or existing.instructions
        existing.tags = list(dict.fromkeys([*existing.tags, *candidate.get("tags", [])]))
        existing.source_tasks = list(dict.fromkeys([*existing.source_tasks, *candidate.get("source_tasks", [])]))
        existing.corroboration_count += candidate.get("corroboration_count", 1)
        if existing.category == "knowledge_general" and existing.corroboration_count >= self.general_promotion_threshold:
            existing.status = "active"
        elif candidate.get("status") == "active":
            existing.status = "active"
        return existing

    def _promote_ready_drafts(self) -> None:
        for r in self.store.list_records(status="draft"):
            if r.category == "knowledge_general" and r.corroboration_count >= self.general_promotion_threshold:
                self.store.promote(r.id)

    def _extract_algorithm(self, text: str) -> Optional[str]:
        lowered = (text or "").lower()
        for kw in ["admm", "fista", "primal-dual", "gradient descent",
                    "plug-and-play", "unet", "wiener", "fft", "ista",
                    "conjugate gradient", "total variation", "deconvolution"]:
            if kw in lowered:
                return kw.replace(" ", "-")
        return None

    def _extract_failure(self, trajectory: Dict[str, Any]) -> Optional[Dict[str, str]]:
        for step in reversed(trajectory.get("steps", [])):
            if step.get("role") != "Judge":
                continue
            out = step.get("output", {}) or {}
            ticket = str(out.get("ticket") or out.get("error_category") or "General")
            analysis = str(out.get("full_judgement_analysis") or out.get("analysis") or "")[:300]
            feedback = str(out.get("feedback") or "")[:300]
            instructions = "\n".join([
                "## When to use",
                f"- Diagnosing failures assigned to `{ticket}`.",
                "",
                "## Failure analysis",
                f"- {analysis}",
                "",
                "## Suggested fix",
                f"- {feedback}",
                "",
                "## Constraints",
                "- Failure-derived knowledge stays draft until corroborated by a later success.",
            ])
            return {"ticket": ticket, "instructions": instructions}
        return None

    # ------------------------------------------------------------------
    # Similarity helpers
    # ------------------------------------------------------------------
    def _record_text(self, r: SkillRecord) -> str:
        return f"{r.description} {r.instructions} {' '.join(r.tags)}"

    def _embed_similarity(self, query: str, records: List[SkillRecord]) -> List[float]:
        if not query or self.embedder is None:
            return [0.0] * len(records)
        try:
            q_vec = np.array(self.embedder.encode(query, normalize_embeddings=True), dtype=np.float32)
        except Exception:
            return [0.0] * len(records)
        scores: list[float] = []
        for r in records:
            try:
                r_vec = np.array(self.embedder.encode(self._record_text(r), normalize_embeddings=True), dtype=np.float32)
                denom = (np.linalg.norm(q_vec) * np.linalg.norm(r_vec)) or 1e-8
                scores.append(float(np.dot(q_vec, r_vec) / denom))
            except Exception:
                scores.append(0.0)
        return scores

    def _tfidf_similarity(self, query: str, docs: Iterable[str]) -> List[float]:
        docs = list(docs)
        if not docs:
            return []
        q_tok = self._tokenize(query)
        d_toks = [self._tokenize(d) for d in docs]
        if not q_tok:
            return [0.0] * len(docs)

        df: dict[str, int] = {}
        for toks in [q_tok, *d_toks]:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        n = len(d_toks) + 1

        def vec(tokens: list[str]) -> dict[str, float]:
            cnt: dict[str, int] = {}
            for t in tokens:
                cnt[t] = cnt.get(t, 0) + 1
            return {t: (c / max(len(tokens), 1)) * (math.log((n + 1) / (df.get(t, 0) + 1)) + 1) for t, c in cnt.items()}

        qv = vec(q_tok)
        results: list[float] = []
        for toks in d_toks:
            dv = vec(toks)
            common = set(qv) & set(dv)
            if not common:
                results.append(0.0)
                continue
            dot = sum(qv[t] * dv[t] for t in common)
            nq = math.sqrt(sum(v * v for v in qv.values()))
            nd = math.sqrt(sum(v * v for v in dv.values()))
            results.append(dot / ((nq * nd) or 1e-8))
        return results

    def _tokenize(self, text: str) -> list[str]:
        clean = (text or "").lower()
        tokens: list[str] = []
        cur: list[str] = []
        for c in clean:
            if c.isalnum() or c == "_":
                cur.append(c)
            else:
                if cur:
                    tok = "".join(cur)
                    if len(tok) > 2:
                        tokens.append(tok)
                    cur = []
        if cur:
            tok = "".join(cur)
            if len(tok) > 2:
                tokens.append(tok)
        return tokens
