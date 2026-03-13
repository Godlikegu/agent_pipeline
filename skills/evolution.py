
import logging
import numpy as np
import sqlite3
from sklearn.cluster import DBSCAN
from typing import List, Dict, Tuple, Optional
import json
import uuid
import time

logger = logging.getLogger(__name__)

class EvolutionManager:
    def __init__(self, skill_manager):
        self.skill_manager = skill_manager
        self.storage = skill_manager.storage
        self.client = skill_manager.client
        self.model_name = skill_manager.teacher.model_name # Reuse teacher's model config

    def run_evolution_loop(self):
        """
        Main entry point for the offline evolutionary loop.
        """
        logger.info(">>> Starting Offline Evolution Loop...")

        # 1. Fetch all Experiences (Patterns)
        experiences = self.storage.get_knowledge_items(k_type='experience', agent_scope='General')
        if not experiences:
            logger.info("No experiences found. Skipping evolution.")
            return

        logger.info(f"Loaded {len(experiences)} experiences for clustering.")

        # 2. Clustering
        clusters = self.cluster_experiences(experiences)
        logger.info(f"Identified {len(clusters)} clusters.")

        if not clusters:
             logger.warning("No clusters found (DBSCAN might be too strict or data too sparse).")

        # 3. Induction (Generate Core Knowledge Candidates)
        for cluster_id, cluster_items in clusters.items():
            if len(cluster_items) < 2:
                continue # Need at least 2 experiences to generalize

            logger.info(f"Processing Cluster {cluster_id} ({len(cluster_items)} items)...")
            print(f"  [Cluster {cluster_id}] Processing {len(cluster_items)} items...")

            candidate_ck = self.induce_core_knowledge(cluster_items)
            if not candidate_ck:
                print("    -> Induction Failed (LLM returned None)")
                continue

            print(f"    -> Induced Candidate: {candidate_ck['name']}")

            # 4. Adversarial Verification (Critic)
            existing_ck = self.storage.get_knowledge_items(k_type='core')
            decision = self.verify_candidate(candidate_ck, existing_ck)

            print(f"    -> Critic Decision: {decision['action']} ({decision.get('reason')})")

            if decision['action'] == 'create':
                print(f"    -> Action: Creating New Core Knowledge")
                self.record_new_core_knowledge(candidate_ck, cluster_items)
            elif decision['action'] == 'merge':
                print(f"    -> Action: Merging into {decision['target_id']}")
                self.merge_into_core_knowledge(decision['target_id'], candidate_ck, cluster_items)
            elif decision['action'] == 'discard':
                logger.info(f"Candidate discarded: {decision['reason']}")

    def cluster_experiences(self, experiences: List[Dict]) -> Dict[int, List[Dict]]:
        """
        Cluster experiences based on embedding similarity using DBSCAN.
        """
        if not experiences:
            return {}

        # Ensure all embeddings have same dimension
        valid_exps = []
        vectors = []

        # Detect majority dimension
        dims = {}
        for item in experiences:
            e = item['embedding']
            if hasattr(e, 'shape'): d = e.shape[0]
            else: d = len(e)
            dims[d] = dims.get(d, 0) + 1

        majority_dim = max(dims, key=dims.get)
        logger.info(f"Clustering majority dimension: {majority_dim} (Distribution: {dims})")

        for item in experiences:
            e = item['embedding']
            if hasattr(e, 'shape'): d = e.shape[0]
            else: d = len(e)

            if d == majority_dim:
                vectors.append(e)
                valid_exps.append(item)
            else:
                logger.warning(f"Skipping item {item['name']} with embedding dim {d} != {majority_dim}")

        if not vectors:
            return {}

        embeddings = np.array(vectors)

        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-10)

        clustering = DBSCAN(eps=0.5, min_samples=2, metric='euclidean').fit(embeddings)

        clusters = {}
        for idx, label in enumerate(clustering.labels_):
            if label == -1:
                continue # Noise
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(valid_exps[idx])

        return clusters

    def induce_core_knowledge(self, cluster_items: List[Dict]) -> Optional[Dict]:
        """
        Use LLM (Generator) to induce a Core Knowledge principle from a cluster of experiences.
        """
        prompt = "You are a Senior Knowledge Engineer. Your task is to distill a Core Knowledge Principle from a set of specific Experiences.\n\n"
        prompt += "### Input Experiences:\n"

        for i, item in enumerate(cluster_items):
            content = item['content']
            prompt += f"{i+1}. **{item['name']}**\n"
            prompt += f"   - Condition: {content.get('condition')}\n"
            prompt += f"   - Action: {content.get('action')}\n"
            prompt += f"   - Rationale: {content.get('rationale')}\n\n"

        prompt += "### Instructions:\n"
        prompt += "1. Analyze the common underlying principle behind these experiences.\n"
        prompt += "2. Formulate a generalized 'Core Knowledge' item.\n"
        prompt += "3. The Principle must be universally applicable (abstract away specific variable names).\n"
        prompt += "4. Include a 'Checklist' of 3-5 verification steps.\n"
        prompt += "5. Identify 'Preconditions' (tags/scenarios) where this applies.\n"
        prompt += "6. Output valid JSON only.\n"

        prompt += """
        ### Output Format:
        {
            "name": "Short Title",
            "principle": "Detailed explanation of the core principle...",
            "checklist": ["Step 1", "Step 2"],
            "preconditions": ["tag1", "tag2"],
            "tags": ["category1"]
        }
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert system architect."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            content = response.choices[0].message.content
            # Extract JSON
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                return json.loads(json_match.group(0))
            return None
        except Exception as e:
            logger.error(f"Induction failed: {e}")
            return None

    def verify_candidate(self, candidate: Dict, existing_ck: List[Dict]) -> Dict:
        """
        Use LLM (Critic) to check for conflicts or merge opportunities.
        """
        if not existing_ck:
            return {'action': 'create'}

        candidate_emb = self.skill_manager.get_embedding(f"{candidate['name']} {candidate['principle']}")
        similar_existing = self.storage.search_knowledge(candidate_emb, k_type='core', top_k=3)

        if not similar_existing:
            return {'action': 'create'}

        # Construct Critic Prompt
        prompt = "You are a Knowledge Base Gatekeeper. Evaluate a Candidate Core Knowledge against Existing entries.\n\n"

        prompt += "### Candidate:\n"
        prompt += f"Name: {candidate['name']}\n"
        prompt += f"Principle: {candidate['principle']}\n"
        prompt += f"Preconditions: {candidate['preconditions']}\n\n"

        prompt += "### Existing Top-3 Matches:\n"
        for item in similar_existing:
            c = item['content']
            prompt += f"- ID: {item['id']}\n"
            prompt += f"  Name: {item['name']}\n"
            prompt += f"  Principle: {c.get('principle')}\n"
            prompt += f"  Preconditions: {item.get('preconditions')}\n\n"

        prompt += "### Decision Logic:\n"
        prompt += "1. **Conflict**: Principles contradict each other under same preconditions -> Reject Candidate.\n"
        prompt += "2. **Redundant**: Candidate is semantically identical -> Discard.\n"
        prompt += "3. **Refine/Merge**: Candidate is better/broader/more detailed -> Merge (Target ID).\n"
        prompt += "4. **Novel**: Principles are different or orthogonal -> Create New.\n"

        prompt += """
        ### Output Format:
        {
            "action": "create" | "merge" | "discard",
            "target_id": "id_if_merge_otherwise_null",
            "reason": "explanation"
        }
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a strict logic verifier."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            content = response.choices[0].message.content
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                return json.loads(json_match.group(0))
            return {'action': 'discard', 'reason': 'Parse error'}
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return {'action': 'discard', 'reason': f"Error: {e}"}

    def record_new_core_knowledge(self, candidate: Dict, source_experiences: List[Dict]):
        """
        Save new Core Knowledge and link source experiences.
        """
        ck_id = str(uuid.uuid4())

        emb_text = f"{candidate['name']} {candidate['principle']}"
        embedding = self.skill_manager.get_embedding(emb_text)

        source_ids = [e['id'] for e in source_experiences]

        item_data = {
            "id": ck_id,
            "name": candidate['name'],
            "type": "core",
            "content": {
                "principle": candidate['principle'],
                "checklist": candidate['checklist']
            },
            "embedding": embedding,
            "tags": candidate.get('tags', []),
            "preconditions": candidate.get('preconditions', []),
            "source_experience_ids": source_ids,
            "status": "hypothesis",
            "version": 1,
            "credit_score": 0.0,
            "agent_scope": "General"
        }

        if self.storage.add_knowledge_item(item_data):
            logger.info(f"Created Core Knowledge: {candidate['name']}")
            print(f"      DB Insert Success: {candidate['name']} (ID: {ck_id})")
            self._update_experience_links(ck_id, source_ids)
        else:
            print(f"      DB Insert Failed for {candidate['name']}")

    def merge_into_core_knowledge(self, target_id: str, candidate: Dict, source_experiences: List[Dict]):
        """
        Update existing Core Knowledge (Refine/Version Up).
        """
        items = self.storage.get_knowledge_by_ids([target_id])
        if not items:
            return
        target = items[0]

        new_source_ids = list(set(target.get('source_experience_ids', []) + [e['id'] for e in source_experiences]))

        new_content = {
            "principle": candidate['principle'],
            "checklist": candidate['checklist']
        }

        conn = sqlite3.connect(self.storage.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE knowledge_items
            SET content = ?,
                version = version + 1,
                source_experience_ids = ?,
                update_time = ?,
                status = 'hypothesis'
            WHERE id = ?
        ''', (
            json.dumps(new_content),
            json.dumps(new_source_ids),
            int(time.time()),
            target_id
        ))
        conn.commit()
        conn.close()

        logger.info(f"Merged/Refined Core Knowledge: {target['name']} -> v{target['version']+1}")
        self._update_experience_links(target_id, [e['id'] for e in source_experiences])

    def _update_experience_links(self, ck_id: str, exp_ids: List[str]):
        """
        Update experiences to point to the CK they contributed to.
        """
        conn = sqlite3.connect(self.storage.db_path)
        cursor = conn.cursor()

        for eid in exp_ids:
            cursor.execute("SELECT contributed_to_ck_ids FROM knowledge_items WHERE id = ?", (eid,))
            row = cursor.fetchone()
            if row:
                current_list = json.loads(row[0]) if row[0] else []
                if ck_id not in current_list:
                    current_list.append(ck_id)
                    cursor.execute("UPDATE knowledge_items SET contributed_to_ck_ids = ? WHERE id = ?",
                                  (json.dumps(current_list), eid))
        conn.commit()
        conn.close()
