"""
File-backed skill manager.

Runtime: retrieve existing skills (read-only) for Planner and Coder.
Post-task: distill new skills from trajectories via LLM, manage draft/permanent lifecycle.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from .file_store import FileSkillStore, SkillRecord, slugify, now_ts

logger = logging.getLogger(__name__)


class FileSkillManager:
    """Workflow-facing API: retrieve (runtime) and distill/promote/cleanup (post-task)."""

    def __init__(
        self,
        client: Optional[Any] = None,
        model_name: str = "",
        config: Optional[dict] = None,
    ) -> None:
        cfg = config or {}
        paths_cfg = cfg.get("_paths", {})
        retrieval_cfg = cfg.get("retrieval", {})
        embedding_cfg = cfg.get("embedding", {})
        learning_cfg = cfg.get("learning", {})
        self._generator_cfg = cfg.get("generator", {})

        self.client = client
        self.model_name = model_name

        # Switches
        self.retrieval_enabled = cfg.get("retrieval_enabled", True)
        self.learning_enabled = cfg.get("learning_enabled", False)

        # Retrieval params
        self.max_items = retrieval_cfg.get("max_items", 5)
        self.similarity_threshold = retrieval_cfg.get("similarity_threshold", 0.6)
        self.max_token_budget = retrieval_cfg.get("max_token_budget", 2000)
        self.top_k_planner = retrieval_cfg.get("top_k_planner", 3)
        self.top_k_coder = retrieval_cfg.get("top_k_coder", 3)
        self.max_permanent = retrieval_cfg.get("max_permanent", 3)
        self.max_draft = retrieval_cfg.get("max_draft", 5)

        # Learning params
        self.merge_similarity_threshold = learning_cfg.get("merge_similarity_threshold", 0.75)
        self.max_skills_per_distillation = learning_cfg.get("max_skills_per_distillation", 10)

        # Store
        root = Path(__file__).resolve().parent.parent
        self.store = FileSkillStore(
            active_dir=paths_cfg.get("active_dir", str(root / ".claude" / "skills")),
            draft_dir=paths_cfg.get("draft_dir", str(root / "skills" / "library" / "drafts")),
            registry_path=paths_cfg.get("registry", str(root / "skills" / "library" / "registry.json")),
            code_pool_dir=paths_cfg.get("code_pool", str(root / "skills" / "library" / "code_pool")),
        )

        # Embedder
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

        # Skills generator agent (lazy init, only for learning)
        self._skills_generator = None

    # ------------------------------------------------------------------
    # Lazy init for SkillsGeneratorAgent
    # ------------------------------------------------------------------
    @property
    def skills_generator(self):
        if self._skills_generator is None:
            from agents.skills_generator import SkillsGeneratorAgent
            self._skills_generator = SkillsGeneratorAgent(
                self.client,
                self.model_name,
                temperature=self._generator_cfg.get("temperature", 0.3),
            )
        return self._skills_generator

    # ------------------------------------------------------------------
    # Runtime: Retrieval (read-only)
    # ------------------------------------------------------------------
    def retrieve_knowledge(
        self,
        task_desc: str,
        agent_role: str = "Planner",
        top_k: int = 3,
        exclude_ids: Optional[Set[str]] = None,
    ) -> Dict[str, List[Dict]]:
        """Retrieve skills for injection. Searches permanent + draft, labels each."""
        empty = {"knowledge_general": [], "knowledge_task_specific": [], "code": []}
        if not self.retrieval_enabled:
            return empty

        exclude = exclude_ids or set()
        query = (task_desc or "").strip()
        if not query:
            return empty

        # Gather all active records, filter by scope
        permanent = self.store.list_records_by_tier("permanent")
        drafts = self.store.list_records_by_tier("draft")

        def scope_filter(r: SkillRecord) -> bool:
            return r.scope in {"General", agent_role}

        permanent = [r for r in permanent if scope_filter(r) and r.id not in exclude]
        drafts = [r for r in drafts if scope_filter(r) and r.id not in exclude]

        # Rank by embedding similarity
        perm_ranked = self._rank_by_similarity(query, permanent)
        draft_ranked = self._rank_by_similarity(query, drafts)

        # Enforce separate tier limits
        perm_ranked = perm_ranked[:self.max_permanent]
        draft_ranked = draft_ranked[:self.max_draft]

        # Merge: permanent first (more reliable), then draft
        all_ranked = perm_ranked + draft_ranked

        # Category limits (use config-driven top_k so all relevant skills can be injected)
        code_limit = 2 if agent_role == "Coder" else 0
        general_limit = max(top_k, self.max_items)
        limits = {
            "knowledge_general": general_limit,
            "knowledge_task_specific": max(3, min(top_k, self.max_items)),
            "code": code_limit,
        }
        results: Dict[str, List[Dict]] = {k: [] for k in limits}
        budget = self.max_token_budget

        for sim_score, r in all_ranked:
            if sim_score < self.similarity_threshold:
                continue
            cat = r.category
            if limits.get(cat, 0) <= 0:
                continue
            payload = self.store.export_prompt_payload(r)
            est = max(60, len(payload.get("instructions", "")) // 3)  # More conservative token estimation
            if budget - est < 0:
                continue
            budget -= est
            results[cat].append(payload)
            limits[cat] -= 1

        return results

    def format_knowledge_for_prompt(self, knowledge: Dict[str, List[Dict]]) -> str:
        """Format retrieved skills for prompt injection. Labels [PERMANENT] or [DRAFT]."""
        if not any(knowledge.values()):
            return ""
        sections: list[str] = [
            "**Note**: [PERMANENT] skills have been validated across multiple task runs and are highly reliable. "
            "[DRAFT] skills are newly extracted and may need validation. "
            "Prefer following [PERMANENT] skills when they conflict with [DRAFT] ones."
        ]

        cat_labels = [
            ("knowledge_general", "General Knowledge Skills"),
            ("knowledge_task_specific", "Task-Specific Knowledge Skills"),
            ("code", "Validated Code Skills"),
        ]

        for cat_key, cat_label in cat_labels:
            items = knowledge.get(cat_key, [])
            if not items:
                continue
            lines = [f"### {cat_label}"]
            for i, item in enumerate(items, 1):
                tier_label = "[PERMANENT]" if item.get("tier") == "permanent" else "[DRAFT]"
                lines.append(f"\n**{i}. {tier_label} {item['name']}**\n")
                lines.append(item.get("instructions", "").strip())
                if item.get("code_snippet_path"):
                    lines.append(f"\nCode reference: `{item['code_snippet_path']}`")
            sections.append("\n".join(lines))

        return "\n\n".join(sections) + "\n"

    def get_knowledge_details(self, knowledge_ids: List[str]) -> List[Dict]:
        return [self.store.export_prompt_payload(r) for r in self.store.get_by_ids(knowledge_ids)]

    # ------------------------------------------------------------------
    # Post-task: Distillation via LLM
    # ------------------------------------------------------------------
    def distill_from_trajectories(
        self,
        task_name: str,
        task_desc: str,
        trajectories: List[dict],
        final_outcome: str,
        code_diff_report: str = "",
    ) -> List[SkillRecord]:
        """Post-task: use SkillsGeneratorAgent to analyze trajectories and create draft skills."""
        if not self.learning_enabled:
            return []

        if not trajectories:
            logger.info("No trajectories to distill.")
            return []

        # 1. Call SkillsGeneratorAgent
        from agents.skills_generator import SkillsGeneratorAgent

        traj_json = json.dumps(trajectories, indent=1, default=str)
        # Truncate to avoid exceeding context
        if len(traj_json) > 30000:
            traj_json = traj_json[:30000] + "\n... (truncated)"

        try:
            raw = self.skills_generator.generate({
                "task_name": task_name,
                "task_desc": task_desc,
                "trajectories_json": traj_json,
                "final_outcome": final_outcome,
                "code_diff_report": code_diff_report,
            })
        except Exception as e:
            logger.error("SkillsGenerator LLM call failed: %s", e)
            return []

        # 2. Parse output
        candidates = SkillsGeneratorAgent.parse_skills_output(raw)
        if not candidates:
            logger.warning("SkillsGenerator returned no parseable skills.")
            return []
        candidates = candidates[:self.max_skills_per_distillation]

        # 3. Process each candidate: similarity check + merge or create
        new_records: List[SkillRecord] = []
        for cand in candidates:
            try:
                record = self._process_candidate(cand, task_name)
                if record:
                    new_records.append(record)
            except Exception as e:
                logger.warning("Failed to process skill candidate: %s", e)

        return new_records

    def _process_candidate(self, candidate: dict, task_name: str) -> Optional[SkillRecord]:
        """Check similarity against existing skills, merge or create new draft."""
        cat = candidate.get("category", "knowledge_task_specific")
        if cat not in {"knowledge_general", "knowledge_task_specific", "code"}:
            cat = "knowledge_task_specific"

        scope = candidate.get("scope", "General")
        if scope not in {"General", "Planner", "Coder"}:
            scope = "General"

        title = candidate.get("title", "Untitled Skill")
        instructions = candidate.get("instructions", "")
        cand_text = f"{title} {instructions}"

        # Find similar existing active skills (any category — cross-category dedup)
        existing = self.store.list_records(status="active")

        best_sim, best_match = 0.0, None
        for r in existing:
            sim = self._compute_similarity(cand_text, f"{r.title} {r.instructions}")
            if sim > best_sim:
                best_sim = sim
                best_match = r

        if best_sim >= self.merge_similarity_threshold and best_match:
            # LLM-based merge
            return self._merge_skills_via_llm(best_match, candidate, task_name)
        else:
            # Create new draft
            fp = hashlib.sha256(cand_text.encode()).hexdigest()[:32]
            record = SkillRecord(
                id=f"sk-{uuid.uuid4().hex[:12]}",
                slug=slugify(title),
                title=title,
                description=candidate.get("description", ""),
                category=cat,
                tier="draft",
                scope=scope,
                instructions=instructions,
                tags=candidate.get("tags", []),
                source_tasks=[task_name],
                task_origin=task_name,
                fingerprint=fp,
            )
            self.store.upsert(record)
            return record

    def _merge_skills_via_llm(
        self, existing: SkillRecord, new_cand: dict, task_name: str
    ) -> SkillRecord:
        """Use SkillsGeneratorAgent to merge two similar skills."""
        from agents.skills_generator import SkillsGeneratorAgent

        existing_dict = {
            "title": existing.title,
            "instructions": existing.instructions,
            "source_tasks": existing.source_tasks,
        }
        new_dict = {
            "title": new_cand.get("title", ""),
            "instructions": new_cand.get("instructions", ""),
            "source_tasks": [task_name],
        }

        try:
            raw = self.skills_generator.generate_merge(existing_dict, new_dict)
            merged = SkillsGeneratorAgent.parse_merge_output(raw)
        except Exception as e:
            logger.warning("Merge LLM call failed: %s, keeping existing skill.", e)
            return existing

        if not merged:
            return existing

        # Update existing record with merged content
        existing.title = merged.get("title", existing.title)
        existing.instructions = merged.get("instructions", existing.instructions)
        existing.description = merged.get("description", existing.description)
        existing.tags = list(dict.fromkeys([*existing.tags, *merged.get("tags", [])]))
        existing.source_tasks = list(dict.fromkeys([*existing.source_tasks, task_name]))
        existing.updated_at = now_ts()

        self.store.overwrite_record(existing.id, existing)
        return existing

    # ------------------------------------------------------------------
    # Post-task: Promotion and Cleanup
    # ------------------------------------------------------------------
    def promote_used_skills(self, used_ids: Set[str], task_name: str) -> List[str]:
        """Promote draft skills that were used to permanent. Returns IDs promoted."""
        promoted = []
        for skill_id in used_ids:
            records = self.store.get_by_ids([skill_id])
            if not records:
                continue
            r = records[0]
            if r.tier != "draft":
                continue

            # Check if similar permanent skill exists for merge
            perm = self.store.list_records_by_tier("permanent")
            best_sim, best_perm = 0.0, None
            r_text = f"{r.title} {r.instructions}"
            for p in perm:
                sim = self._compute_similarity(r_text, f"{p.title} {p.instructions}")
                if sim > best_sim:
                    best_sim = sim
                    best_perm = p

            if best_sim >= self.merge_similarity_threshold and best_perm:
                # Merge into the permanent skill
                self._merge_skills_via_llm(best_perm, {
                    "title": r.title,
                    "instructions": r.instructions,
                    "tags": r.tags,
                }, task_name)
                # Archive the draft
                r.status = "archived"
                self.store.upsert(r)
            else:
                self.store.promote_to_permanent(skill_id)

            promoted.append(skill_id)
        return promoted

    def cleanup_draft_skills(self, task_name: str, exclude_ids: set = None) -> int:
        """Delete (archive) all remaining draft skills for a task, excluding specified IDs."""
        return self.store.delete_draft_skills_for_task(task_name, exclude_ids=exclude_ids)

    # ------------------------------------------------------------------
    # Similarity helpers (embedding-only)
    # ------------------------------------------------------------------
    def _compute_similarity(self, text_a: str, text_b: str) -> float:
        """Compute similarity between two texts. Uses embedding if available, TF-IDF fallback."""
        if self.embedder is not None:
            try:
                vec_a = np.array(self.embedder.encode(text_a, normalize_embeddings=True), dtype=np.float32)
                vec_b = np.array(self.embedder.encode(text_b, normalize_embeddings=True), dtype=np.float32)
                return float(np.dot(vec_a, vec_b))
            except Exception:
                pass
        # TF-IDF fallback
        scores = self._tfidf_similarity(text_a, [text_b])
        return scores[0] if scores else 0.0

    def _rank_by_similarity(
        self, query: str, records: List[SkillRecord]
    ) -> List[Tuple[float, SkillRecord]]:
        """Rank records by similarity. Uses embedding if available, TF-IDF fallback otherwise."""
        if not query or not records:
            return []
        texts = [self._record_text(r) for r in records]
        if self.embedder is not None:
            sims = self._embed_similarity(query, records)
        else:
            sims = self._tfidf_similarity(query, texts)
        ranked = [(sims[i], r) for i, r in enumerate(records)]
        ranked.sort(key=lambda x: x[0], reverse=True)
        return ranked

    def _record_text(self, r: SkillRecord) -> str:
        return f"{r.title} {r.description} {r.instructions} {' '.join(r.tags)}"

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
                r_vec = np.array(
                    self.embedder.encode(self._record_text(r), normalize_embeddings=True),
                    dtype=np.float32,
                )
                scores.append(float(np.dot(q_vec, r_vec)))
            except Exception:
                scores.append(0.0)
        return scores

    def _tfidf_similarity(self, query: str, docs: list) -> List[float]:
        """TF-IDF cosine similarity fallback when embedder is unavailable."""
        docs = list(docs)
        if not docs:
            return []
        q_tok = self._tokenize(query)
        d_toks = [self._tokenize(d) for d in docs]
        if not q_tok:
            return [0.0] * len(docs)

        df: dict = {}
        for toks in [q_tok, *d_toks]:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        n = len(d_toks) + 1

        def vec(tokens):
            cnt: dict = {}
            for t in tokens:
                cnt[t] = cnt.get(t, 0) + 1
            return {t: (c / max(len(tokens), 1)) * (math.log((n + 1) / (df.get(t, 0) + 1)) + 1)
                    for t, c in cnt.items()}

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

    @staticmethod
    def _tokenize(text: str) -> list:
        clean = (text or "").lower()
        tokens: list = []
        cur: list = []
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
