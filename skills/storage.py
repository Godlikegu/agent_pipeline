import sqlite3
import numpy as np
import time
import json
import logging
import os
import pathlib
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class SkillStorage:
    def __init__(self, db_path: str = "skills.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Original skills table (Keep for backward compatibility or migration)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS skills (
            name TEXT PRIMARY KEY,
            principle TEXT NOT NULL,
            when_to_apply TEXT NOT NULL,
            source TEXT NOT NULL,
            embedding BLOB NOT NULL,
            create_time INTEGER NOT NULL,
            update_time INTEGER NOT NULL,
            version INTEGER NOT NULL DEFAULT 1
        )
        ''')

        # Global Metadata Table for Access Counters
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        ''')

        # Initialize global access counter if not exists
        cursor.execute("INSERT OR IGNORE INTO system_metadata (key, value) VALUES ('global_access_counter', '0')")

        # New Knowledge Layered System
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS knowledge_items (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            type TEXT NOT NULL,           -- 'instance', 'experience', 'core'
            content TEXT NOT NULL,        -- JSON content
            embedding BLOB NOT NULL,
            tags TEXT,                    -- JSON list of tags
            source_trajectories TEXT,     -- JSON list
            credit_score REAL DEFAULT 1.0,
            usage_count INTEGER DEFAULT 0,
            last_access_counter INTEGER DEFAULT 0, -- Replaces time-based access
            agent_scope TEXT,             -- 'Planner', 'Architect', 'Coder', 'Judge', 'General'
            artifact_type TEXT,           -- 'plan', 'skeleton', 'code', 'feedback', 'principle'
            status TEXT DEFAULT 'active', -- 'active', 'archived', 'hypothesis', 'verified', 'deprecated'
            create_time INTEGER NOT NULL,
            update_time INTEGER NOT NULL,
            version INTEGER NOT NULL DEFAULT 1,
            preconditions TEXT,           -- JSON list of preconditions (for conflict detection)
            contributed_to_ck_ids TEXT,   -- JSON list of Core Knowledge IDs this item contributed to (for Experience)
            source_experience_ids TEXT    -- JSON list of Experience IDs that formed this Core Knowledge (for Core)
        )
        ''')

        # Migration: Add columns if they don't exist (for existing DBs)
        try: cursor.execute("ALTER TABLE knowledge_items ADD COLUMN agent_scope TEXT")
        except sqlite3.OperationalError: pass
        try: cursor.execute("ALTER TABLE knowledge_items ADD COLUMN artifact_type TEXT")
        except sqlite3.OperationalError: pass
        try: cursor.execute("ALTER TABLE knowledge_items ADD COLUMN status TEXT DEFAULT 'active'")
        except sqlite3.OperationalError: pass
        try: cursor.execute("ALTER TABLE knowledge_items ADD COLUMN preconditions TEXT")
        except sqlite3.OperationalError: pass
        try: cursor.execute("ALTER TABLE knowledge_items ADD COLUMN contributed_to_ck_ids TEXT")
        except sqlite3.OperationalError: pass
        try: cursor.execute("ALTER TABLE knowledge_items ADD COLUMN source_experience_ids TEXT")
        except sqlite3.OperationalError: pass
        try: cursor.execute("ALTER TABLE knowledge_items ADD COLUMN last_access_counter INTEGER DEFAULT 0")
        except sqlite3.OperationalError: pass

        # NOTE: trajectories_meta table removed — trajectories are no longer persisted.
        # After distillation, instances and experiences are extracted and stored in knowledge_items.
        # The raw trajectory is discarded to save storage space.

        conn.commit()
        conn.close()

    def get_global_access_counter(self) -> int:
        """Get the current global access counter."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT value FROM system_metadata WHERE key = 'global_access_counter'")
            result = cursor.fetchone()
            return int(result[0]) if result else 0
        except Exception as e:
            logger.error(f"Error reading global access counter: {e}")
            return 0
        finally:
            conn.close()

    def increment_global_access_counter(self) -> int:
        """Increment and return the new global access counter."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("UPDATE system_metadata SET value = CAST(value AS INTEGER) + 1 WHERE key = 'global_access_counter'")
            conn.commit()
            return self.get_global_access_counter()
        except Exception as e:
            logger.error(f"Error incrementing global access counter: {e}")
            return 0
        finally:
            conn.close()

    def add_knowledge_item(self, item_data: Dict) -> bool:
        """Add a new knowledge item (Instance/Experience/Core)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            now = int(time.time())
            embedding_bytes = np.array(item_data['embedding'], dtype=np.float32).tobytes()

            # Generate UUID if not provided
            import uuid
            item_id = item_data.get('id', str(uuid.uuid4()))

            # Get current global counter for initial last_access_counter
            # Newly added items are considered "fresh" (accessed now)
            current_counter = self.get_global_access_counter()

            cursor.execute('''
            INSERT INTO knowledge_items (
                id, name, type, content, embedding, tags, source_trajectories,
                credit_score, usage_count, last_access_counter, agent_scope, artifact_type, status,
                create_time, update_time, version,
                preconditions, contributed_to_ck_ids, source_experience_ids
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                item_id,
                item_data['name'],
                item_data['type'],
                json.dumps(item_data['content']),
                embedding_bytes,
                json.dumps(item_data.get('tags', [])),
                json.dumps(item_data.get('source_trajectories', [])),
                item_data.get('credit_score', 1.0),
                item_data.get('usage_count', 0),
                current_counter, # Set to current global counter
                item_data.get('agent_scope', 'General'),
                item_data.get('artifact_type', 'unknown'),
                item_data.get('status', 'active'),
                now,
                now,
                item_data.get('version', 1),
                json.dumps(item_data.get('preconditions', [])),
                json.dumps(item_data.get('contributed_to_ck_ids', [])),
                json.dumps(item_data.get('source_experience_ids', []))
            ))
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error adding knowledge item: {e}")
            return False
        finally:
            conn.close()

    def get_knowledge_by_ids(self, ids: List[str]) -> List[Dict]:
        """Retrieve specific knowledge items by their IDs."""
        if not ids:
            return []

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Prepare placeholders for IN clause
        placeholders = ','.join('?' for _ in ids)
        query = f"SELECT * FROM knowledge_items WHERE id IN ({placeholders})"

        try:
            cursor.execute(query, tuple(ids))
            rows = cursor.fetchall()

            items = []
            for row in rows:
                item = dict(row)
                item['content'] = json.loads(item['content'])
                # No need to load embedding for details view usually, but let's be consistent
                item['embedding'] = np.frombuffer(item['embedding'], dtype=np.float32)
                item['tags'] = json.loads(item['tags']) if item['tags'] else []
                item['source_trajectories'] = json.loads(item['source_trajectories']) if item['source_trajectories'] else []

                # Load new fields safely
                item['preconditions'] = json.loads(item['preconditions']) if item.get('preconditions') else []
                item['contributed_to_ck_ids'] = json.loads(item['contributed_to_ck_ids']) if item.get('contributed_to_ck_ids') else []
                item['source_experience_ids'] = json.loads(item['source_experience_ids']) if item.get('source_experience_ids') else []

                items.append(item)

            return items
        except Exception as e:
            logger.error(f"Error retrieving knowledge by IDs: {e}")
            return []
        finally:
            conn.close()

    def get_knowledge_items(self, k_type: str = None, agent_scope: str = None, limit: int = None) -> List[Dict]:
        """Retrieve all knowledge items of a certain type."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # NOTE: Removed status filter for debugging to see everything
        query = "SELECT * FROM knowledge_items WHERE 1=1"
        params = []

        if k_type:
            query += " AND type = ?"
            params.append(k_type)

        if agent_scope:
            query += " AND agent_scope = ?"
            params.append(agent_scope)

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()

        items = []
        for row in rows:
            item = dict(row)
            item['content'] = json.loads(item['content'])
            item['embedding'] = np.frombuffer(item['embedding'], dtype=np.float32)

            # Standardize JSON text fields to Python lists
            for json_field in ['tags', 'source_trajectories', 'preconditions', 'contributed_to_ck_ids', 'source_experience_ids']:
                val = item.get(json_field)
                if val and isinstance(val, str):
                    try:
                        item[json_field] = json.loads(val)
                    except (json.JSONDecodeError, TypeError):
                        item[json_field] = []
                elif not val:
                    item[json_field] = []

            items.append(item)

        conn.close()
        return items

    def search_knowledge(self, query_embedding: List[float], k_type: Optional[str] = None, agent_scope: Optional[str] = None, top_k: int = 5) -> List[Dict]:
        """
        Vector search for knowledge items with weighted ranking.
        Score = Similarity * 0.7 + Normalized_Credit * 0.3
        """
        items = self.get_knowledge_items(k_type, agent_scope)
        if not items:
            return []

        query_vec = np.array(query_embedding, dtype=np.float32)
        norm_query = np.linalg.norm(query_vec)

        scored_items = []

        # Calculate max credit for normalization (avoid div by zero)
        max_credit = max((item['credit_score'] for item in items), default=1.0)
        if max_credit < 1.0: max_credit = 1.0

        for item in items:
            item_vec = item['embedding']

            if item_vec.shape != query_vec.shape:
                continue

            norm_item = np.linalg.norm(item_vec)

            if norm_query == 0 or norm_item == 0:
                similarity = 0.0
            else:
                similarity = np.dot(query_vec, item_vec) / (norm_query * norm_item)

            # Weighted Score
            norm_credit = item['credit_score'] / max_credit
            norm_credit = max(0.0, min(1.0, norm_credit))

            final_score = similarity * 0.7 + norm_credit * 0.3

            # Store raw similarity for downstream re-ranking if needed
            item['_similarity'] = similarity

            scored_items.append((final_score, item))

        # Sort by final score descending
        scored_items.sort(key=lambda x: x[0], reverse=True)

        # Diversity Filter: max 1 per name to ensure diversity
        final_items = []
        seen_names = {}

        for score, item in scored_items:
            name = item['name']
            if seen_names.get(name, 0) >= 1:
                continue

            final_items.append(item)
            seen_names[name] = seen_names.get(name, 0) + 1

            if len(final_items) >= top_k:
                break

        return final_items

    def update_knowledge_usage(self, item_id: str, success: bool,
                               success_delta: float = 0.1,
                               failure_delta: float = -0.2,
                               archive_threshold: float = -0.5):
        """
        Update credit score and usage count based on outcome.

        Args:
            item_id: Knowledge item ID.
            success: Whether the task succeeded.
            success_delta: Credit increment on success.
            failure_delta: Credit decrement on failure.
            archive_threshold: Auto-archive when credit drops below this.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            delta = success_delta if success else failure_delta

            # Get current global counter
            cursor.execute("SELECT value FROM system_metadata WHERE key = 'global_access_counter'")
            result = cursor.fetchone()
            current_counter = int(result[0]) if result else 0

            cursor.execute('''
                UPDATE knowledge_items
                SET usage_count = usage_count + 1,
                    credit_score = credit_score + ?,
                    last_access_counter = ?,
                    update_time = ?
                WHERE id = ?
            ''', (delta, current_counter, int(time.time()), item_id))

            # Archive check
            cursor.execute('''
                UPDATE knowledge_items
                SET status = 'archived'
                WHERE id = ? AND credit_score < ?
            ''', (item_id, archive_threshold))

            conn.commit()
        except Exception as e:
            logger.error(f"Error updating knowledge stats: {e}")
        finally:
            conn.close()

    # NOTE: add_trajectory() and get_trajectories() have been removed.
    # Trajectories are no longer persisted after distillation.
    # The extracted instances and experiences in knowledge_items are the durable artifacts.

    def add_skill(self, skill_data: Dict) -> bool:
        """Add a new skill to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            now = int(time.time())
            embedding_bytes = np.array(skill_data['embedding'], dtype=np.float32).tobytes()

            cursor.execute('''
            INSERT INTO skills (name, principle, when_to_apply, source, embedding, create_time, update_time, version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                skill_data['name'],
                skill_data['principle'],
                skill_data['when_to_apply'],
                json.dumps(skill_data['source']),  # Store source as JSON list
                embedding_bytes,
                now,
                now,
                1
            ))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            logger.warning(f"Skill with name {skill_data['name']} already exists.")
            return False
        except Exception as e:
            logger.error(f"Error adding skill: {e}")
            return False
        finally:
            conn.close()

    def update_skill(self, name: str, update_data: Dict) -> bool:
        """Update an existing skill."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            fields = []
            values = []
            for k, v in update_data.items():
                if k == 'embedding':
                    v = np.array(v, dtype=np.float32).tobytes()
                elif k == 'source':
                    v = json.dumps(v)
                fields.append(f"{k} = ?")
                values.append(v)

            fields.append("update_time = ?")
            values.append(int(time.time()))

            # Increment version
            fields.append("version = version + 1")

            query = f"UPDATE skills SET {', '.join(fields)} WHERE name = ?"
            values.append(name)

            cursor.execute(query, tuple(values))
            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error updating skill {name}: {e}")
            return False
        finally:
            conn.close()

    def get_all_skills(self) -> List[Dict]:
        """Retrieve all skills (helper for vector search)."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM skills")
        rows = cursor.fetchall()

        skills = []
        for row in rows:
            skill = dict(row)
            skill['embedding'] = np.frombuffer(skill['embedding'], dtype=np.float32)
            skill['source'] = json.loads(skill['source'])
            skills.append(skill)

        conn.close()
        return skills

    def get_relevant_skills(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """Retrieve top_k relevant skills based on cosine similarity."""
        skills = self.get_all_skills()
        if not skills:
            return []

        query_vec = np.array(query_embedding, dtype=np.float32)
        norm_query = np.linalg.norm(query_vec)

        scored_skills = []
        for skill in skills:
            skill_vec = skill['embedding']

            # Check dimension match
            if skill_vec.shape != query_vec.shape:
                continue

            norm_skill = np.linalg.norm(skill_vec)

            if norm_query == 0 or norm_skill == 0:
                similarity = 0.0
            else:
                similarity = np.dot(query_vec, skill_vec) / (norm_query * norm_skill)

            scored_skills.append((similarity, skill))

        # Sort by similarity descending
        scored_skills.sort(key=lambda x: x[0], reverse=True)

        return [s[1] for s in scored_skills[:top_k]]

    def merge_skill(self, new_skill_data: Dict, similarity_threshold: float = 0.85) -> bool:
        """
        Check if a similar skill exists. If so, merge; otherwise add new.
        Returns True if merged, False if added as new.
        """
        try:
            # Get existing skills by name first (exact match check)
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM skills WHERE name = ?", (new_skill_data['name'],))
            existing_by_name = cursor.fetchone()
            conn.close()

            if existing_by_name:
                  print(f"  [Storage] Skill '{new_skill_data['name']}' already exists by name. Merging sources...")

                  # Update source list
                  current_sources = json.loads(existing_by_name[3]) # source is at index 3
                  new_sources = new_skill_data['source']
                  combined = list(set(current_sources + new_sources))

                  conn = sqlite3.connect(self.db_path)
                  cursor = conn.cursor()
                  cursor.execute("UPDATE skills SET source = ?, update_time = ?, version = version + 1 WHERE name = ?",
                                (json.dumps(combined), int(time.time()), new_skill_data['name']))
                  conn.commit()
                  conn.close()
                  return True

            relevant = self.get_relevant_skills(new_skill_data['embedding'], top_k=1)
        except Exception as e:
            # If get_relevant_skills fails, just add as new
            print(f"  ⚠️ Vector search failed: {e}. Adding as new.")
            self.add_skill(new_skill_data)
            return False

        if relevant:
            best_match = relevant[0]
            query_vec = np.array(new_skill_data['embedding'], dtype=np.float32)
            match_vec = best_match['embedding']

            similarity = np.dot(query_vec, match_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(match_vec))

            if similarity >= similarity_threshold:
                # Merge logic
                logger.info(f"Merging new skill into existing skill: {best_match['name']} (sim={similarity:.2f})")

                # Combine sources
                existing_sources = best_match['source']
                new_sources = new_skill_data['source']
                combined_sources = list(set(existing_sources + new_sources))

                n_existing = best_match.get('version', 1)
                new_embedding = (match_vec * n_existing + query_vec) / (n_existing + 1)

                update_data = {
                    'source': combined_sources,
                    'embedding': new_embedding,
                }
                self.update_skill(best_match['name'], update_data)
                return True

        # If no match or not similar enough, add as new
        self.add_skill(new_skill_data)
        return False
