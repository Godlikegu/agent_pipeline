
import os
import json
import time
from typing import List, Dict, Any

class ExecutionReporter:
    def __init__(self, root_output_dir: str):
        self.root_output_dir = root_output_dir
        self.start_time = time.time()
        self.results = []
        self.total_stats = {
            'knowledge_general': 0,
            'knowledge_task_specific': 0,
            'code': 0,
        }

    def add_result(self, task_name: str, workflow: Any, success: bool, elapsed: float):
        """
        Extracts execution details from a workflow instance and adds to report.
        """
        loops_used = workflow.retry_count

        # Extract generated knowledge stats (if available)
        dist_stats = getattr(workflow, 'distillation_stats', {'instances': 0, 'experiences': 0, 'core': 0})

        for key in self.total_stats:
            self.total_stats[key] += dist_stats.get(key, 0)

        result_entry = {
            "task_name": task_name,
            "outcome": "Success" if success else "Failure",
            "loops_used": loops_used,
            "elapsed_seconds": round(elapsed, 2),
            "generated_knowledge": dist_stats,
            "used_knowledge_count": len(workflow.used_knowledge_ids) if hasattr(workflow, 'used_knowledge_ids') else 0,
            "error_summary": self._get_error_summary(workflow) if not success else None
        }
        self.results.append(result_entry)

    def _get_error_summary(self, workflow: Any) -> str:
        if hasattr(workflow, 'failure_history') and workflow.failure_history:
            last = workflow.failure_history[-1]
            return f"{last.get('ticket_assigned_to', 'Unknown')}: {last.get('analysis', '')[:100]}..."
        return "Unknown Error"

    def generate_report(self):
        """
        Writes the JSON report to file.
        """
        duration = time.time() - self.start_time
        success_count = sum(1 for r in self.results if r['outcome'] == 'Success')
        failure_count = len(self.results) - success_count

        report = {
            "meta": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_duration_seconds": round(duration, 2),
                "total_tasks": len(self.results),
                "success_count": success_count,
                "failure_count": failure_count,
                "success_rate": round(success_count / len(self.results) * 100, 1) if self.results else 0
            },
            "knowledge_generation_summary": self.total_stats,
            "tasks": self.results
        }

        filename = f"execution_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
        path = os.path.join(self.root_output_dir, filename)

        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"\n✅ Report generated: {path}")
            return path
        except Exception as e:
            print(f"\n❌ Failed to generate report: {e}")
            return None
