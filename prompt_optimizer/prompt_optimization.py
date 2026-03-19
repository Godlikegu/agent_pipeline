"""Simplified textgrad prompt optimization for the task generator."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import random
from typing import List

import textgrad as tg
from textgrad import BlackboxLLM, Variable
from textgrad.engine import LiteLLMEngine
from textgrad.optimizer import TextualGradientDescent
from litellm import completion
import yaml

from agents.task_generator import (
    DEFAULT_SYSTEM_PROMPT_PATH,
    DEFAULT_USER_PROMPT,
    TaskGeneratorAgent,
)


LOGGER = logging.getLogger(__name__)

TASK_DESCRIPTION_COMPARISON_PROMPT = """
You are optimizing the system prompt of a task generator.

Task name: {task_name}

Ground truth task_description:
{ground_truth_description}

Generation input:
{generation_input}

Compare the generated task_description against the ground truth.
Focus on:
1. Missing problem formulation, solver logic, or implementation constraints.
2. Missing input/output contract details.
3. Unsupported hallucinated details not grounded in the paper.
4. Structural weaknesses that make the description less useful for downstream agents.

Return concise, concrete feedback for how the SYSTEM PROMPT should change so future task descriptions are closer to the ground truth.
"""


@dataclass
class OptimizationSample:
    """One paper/ground-truth pair used for optimization."""

    task_name: str
    paper_markdown: str
    ground_truth_description: str
    user_prompt: str


class SafeLiteLLMEngine(LiteLLMEngine):
    """LiteLLM engine with explicit api_key/base_url forwarding."""

    def __init__(self, model_string, api_key=None, base_url=None, **kwargs):
        super().__init__(model_string, **kwargs)
        self.api_key = api_key
        self.base_url = base_url

    def lite_llm_generate(self, content, system_prompt=None, **kwargs) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        return completion(
            model=self.model_string,
            messages=messages,
            api_key=self.api_key,
            base_url=self.base_url,
            **kwargs,
        )["choices"][0]["message"]["content"]


def load_yaml(path: str | Path) -> dict:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def get_llm_config(config_path: str | Path, model_key: str) -> dict:
    config = load_yaml(config_path)
    models = config.get("models", {})
    if model_key not in models:
        raise ValueError(
            f"Model '{model_key}' not found in {config_path}. Available: {list(models)}"
        )
    return models[model_key]


def build_engine(config_path: str | Path, model_key: str) -> SafeLiteLLMEngine:
    model_conf = get_llm_config(config_path, model_key)
    model_name = model_conf.get("model_name", model_key)
    return SafeLiteLLMEngine(
        model_string=f"openai/{model_name}",
        api_key=model_conf.get("api_key"),
        base_url=model_conf.get("base_url"),
    )


def normalize_task_name(path: Path) -> str:
    name = path.stem
    for suffix in ("_description", "-description", ".description"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_dataset(
    paper_dir: str | Path,
    ground_truth_dir: str | Path,
    user_prompt: str,
) -> List[OptimizationSample]:
    paper_root = Path(paper_dir).expanduser().resolve()
    ground_truth_root = Path(ground_truth_dir).expanduser().resolve()
    if not paper_root.exists():
        raise FileNotFoundError(f"Paper markdown directory not found: {paper_root}")
    if not ground_truth_root.exists():
        raise FileNotFoundError(
            f"Ground truth task_description directory not found: {ground_truth_root}"
        )

    paper_files = sorted(path for path in paper_root.rglob("*.md") if path.is_file())
    paper_map = {path.stem: path for path in paper_files}

    dataset: List[OptimizationSample] = []
    gt_files = sorted(path for path in ground_truth_root.rglob("*.md") if path.is_file())
    for gt_path in gt_files:
        task_name = normalize_task_name(gt_path)
        paper_path = paper_map.get(task_name)
        if paper_path is None:
            LOGGER.warning(
                "Skipping %s because no matching paper markdown was found.",
                gt_path.name,
            )
            continue

        dataset.append(
            OptimizationSample(
                task_name=task_name,
                paper_markdown=read_text(paper_path),
                ground_truth_description=read_text(gt_path),
                user_prompt=user_prompt,
            )
        )

    LOGGER.info("Loaded %d optimization sample(s).", len(dataset))
    return dataset


def log_artifact(path: Path, name: str, content: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n{'=' * 20} {name} {'=' * 20}\n")
        handle.write(content)
        handle.write("\n")


def process_single_item(
    sample: OptimizationSample,
    model_name: str,
    desc_engines: dict,
    judge_engine: SafeLiteLLMEngine,
    artifacts_log_path: Path,
):
    generation_input = TaskGeneratorAgent.build_model_input(
        paper_markdown=sample.paper_markdown,
        user_prompt=sample.user_prompt,
    )

    generated_description = desc_engines[model_name](
        Variable(
            generation_input,
            requires_grad=False,
            role_description="task_generator_input",
        )
    )
    generated_description.set_role_description(
        f"generated_task_description_{sample.task_name}_{model_name}"
    )

    log_artifact(
        artifacts_log_path,
        f"Generated Description ({sample.task_name} - {model_name})",
        generated_description.value,
    )

    loss_fn = tg.TextLoss(
        eval_system_prompt=TASK_DESCRIPTION_COMPARISON_PROMPT.format(
            task_name=sample.task_name,
            ground_truth_description=sample.ground_truth_description,
            generation_input=generation_input,
        ),
        engine=judge_engine,
    )
    loss = loss_fn(generated_description)

    log_artifact(
        artifacts_log_path,
        f"Comparison Feedback ({sample.task_name} - {model_name})",
        loss.value,
    )
    return loss


def train_step(
    batch_data: List[OptimizationSample],
    optimizer: TextualGradientDescent,
    desc_engines: dict,
    judge_engine: SafeLiteLLMEngine,
    models: List[str],
    artifacts_log_path: Path,
) -> str:
    optimizer.zero_grad()
    total_loss_nodes = []

    LOGGER.info("Processing batch of size %d", len(batch_data))
    for sample in batch_data:
        for model_name in models:
            total_loss_nodes.append(
                process_single_item(
                    sample=sample,
                    model_name=model_name,
                    desc_engines=desc_engines,
                    judge_engine=judge_engine,
                    artifacts_log_path=artifacts_log_path,
                )
            )

    LOGGER.info(
        "Computed %d feedback signal(s). Backpropagating...",
        len(total_loss_nodes),
    )
    for loss_node in total_loss_nodes:
        loss_node.backward()

    optimizer.step()
    updated_prompt = desc_engines[models[0]].system_prompt.value
    log_artifact(artifacts_log_path, "Updated System Prompt", updated_prompt)
    return updated_prompt


def get_batches(dataset: List[OptimizationSample], batch_size: int):
    shuffled = dataset[:]
    random.shuffle(shuffled)
    for start in range(0, len(shuffled), batch_size):
        yield shuffled[start : start + batch_size]


def resolve_user_prompt(args) -> str:
    if args.user_prompt:
        return args.user_prompt
    if args.user_prompt_file:
        return Path(args.user_prompt_file).expanduser().resolve().read_text(
            encoding="utf-8"
        ).strip()
    return DEFAULT_USER_PROMPT


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize the task generator prompt with textgrad"
    )
    parser.add_argument("--paper-dir", required=True, help="Directory containing paper markdown files.")
    parser.add_argument(
        "--ground-truth-dir",
        required=True,
        help="Directory containing ground truth task_description markdown files.",
    )
    parser.add_argument(
        "--llm-config",
        default=str((Path(__file__).parent.parent / "config" / "llm.yaml").resolve()),
        help="Path to llm.yaml.",
    )
    parser.add_argument(
        "--models",
        required=True,
        help="Comma-separated generator model keys.",
    )
    parser.add_argument(
        "--optimizer-model",
        required=True,
        help="Model key used for textgrad backward/evaluation.",
    )
    parser.add_argument("--user-prompt", default="", help="User prompt shared by all samples.")
    parser.add_argument(
        "--user-prompt-file",
        default="",
        help="Optional file containing the shared user prompt.",
    )
    parser.add_argument(
        "--prompt-path",
        default=str(DEFAULT_SYSTEM_PROMPT_PATH.resolve()),
        help="Initial task generator system prompt path.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs.")
    parser.add_argument(
        "--output-dir",
        default=str((Path(__file__).parent / "optimized_prompts").resolve()),
        help="Directory for logs and optimized prompts.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_log_path = output_dir / "optimization_artifacts.log"
    execution_log_path = output_dir / "optimization_execution.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(execution_log_path, mode="a"),
            logging.StreamHandler(),
        ],
    )

    models = [item.strip() for item in args.models.split(",") if item.strip()]
    user_prompt = resolve_user_prompt(args)
    dataset = load_dataset(
        paper_dir=args.paper_dir,
        ground_truth_dir=args.ground_truth_dir,
        user_prompt=user_prompt,
    )
    if not dataset:
        raise RuntimeError("No optimization samples were loaded.")

    initial_prompt = Path(args.prompt_path).expanduser().resolve().read_text(
        encoding="utf-8"
    ).strip()

    optimizer_engine = build_engine(args.llm_config, args.optimizer_model)
    tg.set_backward_engine(optimizer_engine)

    system_prompt_var = Variable(
        initial_prompt,
        requires_grad=True,
        role_description="task_generator_system_prompt",
    )

    desc_engines = {}
    for model_name in models:
        desc_engines[model_name] = BlackboxLLM(
            engine=build_engine(args.llm_config, model_name),
            system_prompt=system_prompt_var,
        )

    optimizer = TextualGradientDescent(
        parameters=[system_prompt_var],
        engine=optimizer_engine,
        constraints=[
            "Keep the system prompt concise and reusable.",
            "Do not hardcode details for a single paper.",
        ],
    )

    LOGGER.info("Initial System Prompt:\n%s", system_prompt_var.value)
    log_artifact(artifacts_log_path, "Initial System Prompt", system_prompt_var.value)

    for epoch in range(args.epochs):
        LOGGER.info("--- Epoch %d ---", epoch + 1)
        for batch_index, batch in enumerate(get_batches(dataset, args.batch_size), start=1):
            LOGGER.info("Processing Batch %d...", batch_index)
            updated_prompt = train_step(
                batch_data=batch,
                optimizer=optimizer,
                desc_engines=desc_engines,
                judge_engine=optimizer_engine,
                models=models,
                artifacts_log_path=artifacts_log_path,
            )
            LOGGER.info("Updated System Prompt (after Batch %d):\n%s", batch_index, updated_prompt)

        epoch_prompt_path = output_dir / f"task_generator_prompt_epoch_{epoch + 1:02d}.txt"
        epoch_prompt_path.write_text(system_prompt_var.value, encoding="utf-8")
        LOGGER.info("Saved epoch prompt to %s", epoch_prompt_path)

    final_prompt_path = output_dir / "task_generator_prompt_final.txt"
    final_prompt_path.write_text(system_prompt_var.value, encoding="utf-8")
    LOGGER.info("Final optimized prompt saved to %s", final_prompt_path)
    log_artifact(
        artifacts_log_path,
        "Final Optimized System Prompt",
        system_prompt_var.value,
    )

    print(
        json.dumps(
            {
                "final_prompt_path": str(final_prompt_path),
                "epochs": args.epochs,
                "dataset_size": len(dataset),
                "models": models,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
