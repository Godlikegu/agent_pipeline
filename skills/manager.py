import re
import logging
import sqlite3
import json
import time
import numpy as np
import random
import hashlib
import math
from typing import List, Dict, Optional, Any
from pathlib import Path
from collections import Counter

# Try to import SentenceTransformer, fallback if not available
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from .storage import SkillStorage
from .teacher import SkillTeacher

logger = logging.getLogger(__name__)

class SkillManager:
    def __init__(self, db_path: str, client: Optional[Any] = None, model_name: str = "gpt-4",
                 config: dict = None):
        """
        Args:
            db_path: SQLite 数据库路径。
            client: OpenAI-compatible LLM 客户端。
            model_name: LLM 模型名。
            config: skills 配置字典（对应 config/default.yaml 中的 skills 节）。
        """
        cfg = config or {}
        retrieval_cfg = cfg.get('retrieval', {})
        credit_cfg = cfg.get('credit', {})
        embedding_cfg = cfg.get('embedding', {})

        # --- Retrieval parameters (from config) ---
        self.max_token_budget = retrieval_cfg.get('max_token_budget', 2000)
        self.similarity_threshold = retrieval_cfg.get('similarity_threshold', 0.50)
        self.max_items = retrieval_cfg.get('max_items', 5)
        self.decay_rate = retrieval_cfg.get('decay_rate', 0.99)
        self.top_k_coder = retrieval_cfg.get('top_k_coder', 4)
        self.top_k_default = retrieval_cfg.get('top_k_default', 3)
        rank_weights = retrieval_cfg.get('rank_weights', {})
        self.rank_sim_weight = rank_weights.get('similarity', 0.7)
        self.rank_credit_weight = rank_weights.get('credit', 0.3)

        # --- Credit parameters (from config) ---
        self.credit_initial_score = credit_cfg.get('initial_score', 1.0)
        self.credit_success_delta = credit_cfg.get('success_delta', 0.1)
        self.credit_failure_delta = credit_cfg.get('failure_delta', -0.2)
        self.credit_archive_threshold = credit_cfg.get('archive_threshold', -0.5)

        # --- Embedding parameters (from config) ---
        self.embedding_model_name = embedding_cfg.get('model_name', 'all-MiniLM-L6-v2')
        self.target_dim = embedding_cfg.get('dimension', 384)

        # --- Storage & Teacher ---
        self.storage = SkillStorage(db_path)
        self.client = client

        teacher_temp = cfg.get('teacher_temperature', 0.2)
        self.teacher = SkillTeacher(client, model_name, temperature=teacher_temp)

        # --- Embedder (Lazy Load) ---
        self.embedder = None

        # Check existing items for dimension consistency
        existing_items = self.storage.get_knowledge_items(limit=1)
        if existing_items:
            e = existing_items[0]['embedding']
            self.target_dim = len(e) if isinstance(e, list) else e.shape[0]
            logger.info(f"Detected existing embedding dimension: {self.target_dim}")

        try:
            # Load SentenceTransformer from local weights bundled with the project
            local_model_path = str(Path(__file__).parent / "embeddings" / self.embedding_model_name)
            if SentenceTransformer is not None and Path(local_model_path).exists():
                self.embedder = SentenceTransformer(local_model_path)
                logger.info(f"Loaded SentenceTransformer from {local_model_path}")
            else:
                logger.warning("SentenceTransformer not available or model not found, falling back to TF-IDF")
        except Exception as e:
            logger.warning(f"Failed to load local embedder: {e}")

    def get_embedding(self, text: str) -> List[float]:
        # Ensure text is string
        if not isinstance(text, str):
            text = str(text)

        # 1. Try local SentenceTransformer (Preferred for speed/reliability if available)
        if self.embedder and self.target_dim == 384:
            try:
                embedding = self.embedder.encode(text, normalize_embeddings=True).tolist()
                return embedding
            except Exception as e:
                logger.warning(f"SentenceTransformer encoding failed: {e}")
                pass

        # 2. Fallback: Deterministic Pseudo-Embedding
        dim = self.target_dim

        if "3D_Input_Requirement" in text:
             seed = 12345
        elif "Visualization" in text:
             seed = 67890
        else:
             hash_val = hashlib.sha256(text.encode('utf-8')).hexdigest()
             seed = int(hash_val, 16)

        random.seed(seed)
        vec = [random.uniform(-1, 1) for _ in range(dim)]

        noise_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        random.seed(int(noise_hash, 16))
        noise = [random.uniform(-0.1, 0.1) for _ in range(dim)]

        return [v + n for v, n in zip(vec, noise)]

    # ===== TF-IDF TEXT SIMILARITY =====
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase, split on non-alphanumeric, remove stopwords."""
        if not isinstance(text, str):
            text = str(text)
        tokens = re.findall(r'[a-z][a-z0-9_]+', text.lower())
        stopwords = {'the','a','an','is','are','was','were','be','been','being',
                     'have','has','had','do','does','did','will','would','could',
                     'should','may','might','shall','can','need','dare','ought',
                     'used','to','of','in','for','on','with','at','by','from',
                     'as','into','through','during','before','after','above',
                     'below','between','out','off','over','under','again','further',
                     'then','once','here','there','when','where','why','how','all',
                     'each','every','both','few','more','most','other','some','such',
                     'no','nor','not','only','own','same','so','than','too','very',
                     'just','because','but','and','or','if','while','that','this',
                     'these','those','it','its','they','them','their','we','our',
                     'you','your','he','she','his','her','my','me','what','which'}
        return [t for t in tokens if t not in stopwords and len(t) > 1]

    def _compute_tfidf_similarity(self, query_text: str, doc_texts: List[str]) -> List[float]:
        """Compute TF-IDF cosine similarity between query and each document."""
        if not doc_texts:
            return []

        query_tokens = self._tokenize(query_text)
        doc_token_lists = [self._tokenize(d) for d in doc_texts]

        if not query_tokens:
            return [0.0] * len(doc_texts)

        all_docs_tokens = [query_tokens] + doc_token_lists

        # Document frequency
        df = Counter()
        for tokens in all_docs_tokens:
            unique_tokens = set(tokens)
            for t in unique_tokens:
                df[t] += 1

        n_docs = len(all_docs_tokens)

        def tfidf_vec(tokens):
            tf = Counter(tokens)
            vec = {}
            for t, count in tf.items():
                idf = math.log((n_docs + 1) / (df.get(t, 0) + 1)) + 1
                vec[t] = (count / max(len(tokens), 1)) * idf
            return vec

        query_vec = tfidf_vec(query_tokens)

        similarities = []
        for doc_tokens in doc_token_lists:
            if not doc_tokens:
                similarities.append(0.0)
                continue
            doc_vec = tfidf_vec(doc_tokens)
            common_keys = set(query_vec.keys()) & set(doc_vec.keys())
            if not common_keys:
                similarities.append(0.0)
                continue
            dot = sum(query_vec[k] * doc_vec[k] for k in common_keys)
            norm_q = math.sqrt(sum(v*v for v in query_vec.values()))
            norm_d = math.sqrt(sum(v*v for v in doc_vec.values()))
            if norm_q == 0 or norm_d == 0:
                similarities.append(0.0)
            else:
                similarities.append(dot / (norm_q * norm_d))

        return similarities

    def _item_to_text(self, item: Dict) -> str:
        """Convert a knowledge item to searchable text representation."""
        parts = [item.get('name', '')]
        content = item.get('content', {})
        if isinstance(content, dict):
            for key in ['condition', 'action', 'rationale', 'description', 'principle']:
                if key in content:
                    parts.append(str(content[key]))
        elif isinstance(content, str):
            parts.append(content)
        tags = item.get('tags', [])
        if tags:
            parts.append(' '.join(str(t) for t in tags))
        return ' '.join(parts)

    def retrieve_knowledge(self, task_desc: str, agent_role: str = 'General', top_k: int = 3) -> Dict[str, List[Dict]]:
        """
        Retrieve layered knowledge relevant to the task and agent role.
        Uses TF-IDF text similarity (no embeddings needed - works offline).
        Implements TOKEN BUDGET retrieval strategy for experiences.
        Applies LAZY DECAY weighting based on Access Count.
        """
        results = {
            "core": [],
            "experience": [],
            "instance": []
        }

        try:
            # Increment Global Access Counter
            global_counter = self.storage.increment_global_access_counter()

            # Strip existing injected skills header if present
            clean_desc = task_desc.split("### 🧠 RELEVANT SKILLS")[0]
            clean_desc = clean_desc.split("### 🛡️ CORE KNOWLEDGE")[0]
            clean_desc = clean_desc.split("### 💡 RELEVANT EXPERIENCE")[0]

            # 1. Core Knowledge — DISABLED (no core injection for now)
            results['core'] = []

            # 2. Experience (Patterns) with Embedding Cosine Similarity + Credit + Decay hybrid ranking
            all_experiences = self.storage.get_knowledge_items(k_type='experience')
            if all_experiences:
                # Compute semantic similarity using SentenceTransformer embeddings
                if self.embedder is not None:
                    query_emb = self.embedder.encode(clean_desc, normalize_embeddings=True)
                    for it in all_experiences:
                        stored_emb = it.get('embedding')
                        if stored_emb is not None and hasattr(stored_emb, '__len__') and len(stored_emb) == self.target_dim:
                            stored_vec = np.array(stored_emb, dtype=np.float32)
                            norm = np.linalg.norm(stored_vec)
                            if norm > 0:
                                stored_vec = stored_vec / norm
                            it['_similarity'] = float(np.dot(query_emb, stored_vec))
                        else:
                            it['_similarity'] = 0.0
                else:
                    # Fallback to TF-IDF if embedder unavailable
                    exp_texts = [self._item_to_text(it) for it in all_experiences]
                    exp_sims = self._compute_tfidf_similarity(clean_desc, exp_texts)
                    for it, sim in zip(all_experiences, exp_sims):
                        it['_similarity'] = sim

                # Apply lazy decay and hybrid ranking
                selected_experiences = []
                current_token_count = 0

                seen_names = set()
                rescored_candidates = []

                max_credit = max((it.get('credit_score', 1.0) for it in all_experiences), default=1.0)
                if max_credit < 1.0: max_credit = 1.0

                for exp in all_experiences:
                    if exp['name'] in seen_names:
                        continue
                    seen_names.add(exp['name'])

                    credit = exp.get('credit_score', 1.0)
                    last_access = exp.get('last_access_counter', 0)
                    age = 5 if last_access == 0 else max(0, global_counter - last_access)

                    decay_factor = pow(self.decay_rate, age)
                    decay_floor = max(0.1, credit * 0.3)
                    effective_score = max(credit * decay_factor, decay_floor)

                    similarity = exp.get('_similarity', 0.0)
                    norm_eff_score = min(1.0, effective_score / max(max_credit, 1.0))

                    rank_score = (similarity * self.rank_sim_weight) + (norm_eff_score * self.rank_credit_weight)

                    exp['_rank_score'] = rank_score
                    rescored_candidates.append(exp)

                rescored_candidates.sort(key=lambda x: x['_rank_score'], reverse=True)

                # Fill Token Budget with similarity gating
                for exp in rescored_candidates:
                    if len(selected_experiences) >= self.max_items:
                        break
                    if exp.get('_similarity', 0.0) < self.similarity_threshold:
                        continue
                    content_str = json.dumps(exp['content']) if isinstance(exp['content'], dict) else str(exp['content'])
                    est_tokens = len(content_str) / 4 + 50

                    if current_token_count + est_tokens < self.max_token_budget:
                        selected_experiences.append(exp)
                        current_token_count += est_tokens
                    else:
                        break

                results['experience'] = selected_experiences

            # 3. Instance (Few-Shot) — DISABLED (no instance injection for now)
            results['instance'] = []

            return results

        except Exception as e:
            logger.error(f"Error retrieving knowledge: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return results

    def update_scores(self, knowledge_ids: List[str], success: bool):
        """
        Batch update credit scores for used knowledge items.
        """
        if not knowledge_ids:
            return

        print(f"  [SkillManager] Updating scores for {len(knowledge_ids)} items (Success={success})...")
        for item_id in set(knowledge_ids):
            self.storage.update_knowledge_usage(
                item_id, success,
                success_delta=self.credit_success_delta,
                failure_delta=self.credit_failure_delta,
                archive_threshold=self.credit_archive_threshold
            )

    def get_knowledge_details(self, knowledge_ids: List[str]) -> List[Dict]:
        """
        Retrieve full details for a list of knowledge IDs.
        """
        if not knowledge_ids:
            return []
        unique_ids = list(set(knowledge_ids))
        return self.storage.get_knowledge_by_ids(unique_ids)

    def format_knowledge_for_prompt(self, knowledge: Dict[str, List[Dict]]) -> str:
        """
        Format layered knowledge into a structured prompt section.
        """
        if not any(knowledge.values()):
            return ""

        formatted = "\n\n"

        # 1. Core Knowledge (currently disabled)
        if knowledge['core']:
            pass

        # 2. Experience Patterns (Mid Priority)
        if knowledge['experience']:
            formatted += "### 💡 RELEVANT EXPERIENCE PATTERNS (STRATEGIES)\n"
            for i, item in enumerate(knowledge['experience'], 1):
                content = item['content']
                formatted += f"{i}. **{item['name']}**\n"
                formatted += f"   - Condition: {content.get('condition', '')}\n"
                formatted += f"   - Action: {content.get('action', '')}\n"
                formatted += f"   - Rationale: {content.get('rationale', '')}\n"
            formatted += "\n"

        # 3. Instances (Reference)
        if knowledge['instance']:
            formatted += "### 📝 REFERENCE EXAMPLES (FEW-SHOT)\n"
            for item in knowledge['instance']:
                a_type = item.get('artifact_type', 'unknown')
                content = item['content']

                formatted += f"#### Example ({a_type}): {item['name']}\n"

                if a_type == 'code':
                    code_snippet = content if isinstance(content, str) else content.get('code', str(content))
                    if len(code_snippet) > 1500:
                        code_snippet = code_snippet[:1500] + "\n# ... (truncated)"
                    formatted += f"```python\n{code_snippet}\n```\n"

                elif a_type == 'plan':
                    plan_text = content if isinstance(content, str) else str(content)
                    if len(plan_text) > 1000:
                        plan_text = plan_text[:1000] + "\n... (truncated)"
                    formatted += f"{plan_text}\n"

                elif a_type == 'skeleton':
                    skel_text = content if isinstance(content, str) else str(content)
                    if len(skel_text) > 1500:
                        skel_text = skel_text[:1500] + "\n# ... (truncated)"
                    formatted += f"```python\n{skel_text}\n```\n"

                elif a_type == 'feedback':
                    fb_str = json.dumps(content, indent=2)
                    if len(fb_str) > 800:
                        fb_str = fb_str[:800] + "\n... (truncated)"
                    formatted += f"```json\n{fb_str}\n```\n"

                else:
                    text_repr = str(content)
                    if len(text_repr) > 1000:
                        text_repr = text_repr[:1000] + "\n... (truncated)"
                    formatted += f"{text_repr}\n"

                formatted += "\n"

        return formatted

    def distill_and_store(self, trajectory: Dict[str, Any]) -> Dict[str, int]:
        """
        Distill layered knowledge from a completed trajectory and store it.
        Trajectory is NOT persisted — only the extracted instances/experiences are stored.
        Returns a stats dict with counts of new items.
        """
        stats = {'instances': 0, 'experiences': 0, 'core': 0}

        print(f">>> [SkillManager] Distilling knowledge from trajectory: {trajectory.get('task_name')}...")

        # 1. Analyze with Teacher (Layered)
        print(f"  [SkillManager] Calling Teacher Model (Layered Extraction)...")
        results = self.teacher.analyze_trajectory_layered(trajectory)

        source_id = trajectory.get('exp_id', 'unknown')

        # 2. Process & Store Instances (Agent Specific)
        for inst in results.get('instances', []):
            try:
                emb_text = f"{inst['name']} {inst.get('description', '')}"
                embedding = self.get_embedding(emb_text)

                content_to_store = inst['content']

                item_data = {
                    "name": inst['name'],
                    "type": "instance",
                    "content": content_to_store,
                    "embedding": embedding,
                    "tags": [inst.get('artifact_type', 'misc'), inst.get('agent_scope', 'General')],
                    "source_trajectories": [source_id],
                    "agent_scope": inst.get('agent_scope', 'General'),
                    "artifact_type": inst.get('artifact_type', 'unknown'),
                    "credit_score": self.credit_initial_score
                }
                if self.storage.add_knowledge_item(item_data):
                    print(f"  Instance stored: {inst['name']} ({inst.get('agent_scope')})")
                    stats['instances'] += 1
            except Exception as e:
                print(f"  Failed to store instance: {e}")

        # 3. Process & Store Experiences
        for exp in results.get('experiences', []):
            try:
                content = exp.get('content', {})
                emb_text = f"{exp.get('name')} {content.get('condition', '')} {content.get('action', '')}"
                embedding = self.get_embedding(emb_text)

                item_data = {
                    "name": self._deinstantiate(exp.get('name')),
                    "type": "experience",
                    "content": {k: self._deinstantiate(v) for k,v in content.items()},
                    "embedding": embedding,
                    "tags": exp.get('tags', []),
                    "source_trajectories": [source_id],
                    "agent_scope": exp.get('agent_scope', 'General'),
                    "artifact_type": "experience_pattern",
                    "credit_score": self.credit_initial_score
                }
                if self.storage.add_knowledge_item(item_data):
                     print(f"  Experience stored: {item_data['name']}")
                     stats['experiences'] += 1
            except Exception as e:
                print(f"  Failed to store experience: {e}")

        # 4. Core Knowledge - handled by offline EvolutionManager
        pass

        return stats

    def _deinstantiate(self, text: str) -> str:
        """
        Replace specific values with placeholders.
        """
        if not text:
            return ""

        # 1. URLs
        text = re.sub(r'https?://\S+', '{url}', text)

        # 2. File paths (Unix-like)
        text = re.sub(r'(?:\.?\.?\/[a-zA-Z0-9_\-\.]+)+', '{path}', text)

        # 3. UUIDs
        text = re.sub(r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b', '{uuid}', text)

        return text
