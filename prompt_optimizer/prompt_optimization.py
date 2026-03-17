"""Reusable textgrad-based prompt optimization framework."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from difflib import SequenceMatcher
import json
import logging
from pathlib import Path
import random
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from task_gen.task_generator import (
    DEFAULT_SYSTEM_PROMPT_PATH,
    TaskDescriptionGenerator,
)


LOGGER = logging.getLogger(__name__)

DEFAULT_OPTIMIZER_CONSTRAINTS = [
    "Keep the system prompt generalizable across related tasks.",
    "Do not hardcode details that only apply to one sample.",
    "Prefer clear instructions over verbose repetition.",
]


def _import_yaml():
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required for loading prompt optimization datasets or model configs."
        ) from exc
    return yaml


def _require_textgrad():
    try:
        import textgrad as tg
        from textgrad import BlackboxLLM, Variable
        from textgrad.optimizer import TextualGradientDescent
    except ImportError as exc:
        raise ImportError(
            "textgrad is required for prompt optimization. "
            "Install it before running prompt_optimizer."
        ) from exc
    return tg, BlackboxLLM, Variable, TextualGradientDescent


def _require_litellm_completion():
    try:
        from litellm import completion
    except ImportError as exc:
        raise ImportError(
            "litellm is required for prompt optimization. "
            "Install it before running prompt_optimizer."
        ) from exc
    return completion


def load_yaml(path: str | Path) -> dict:
    yaml = _import_yaml()
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


def _resolve_text_path(path: str | Path, base_dir: Optional[Path] = None) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() and base_dir is not None:
        candidate = base_dir / candidate
    return candidate.resolve()


def _load_text_file(path: str | Path, base_dir: Optional[Path] = None) -> str:
    resolved_path = _resolve_text_path(path, base_dir)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Text file not found: {resolved_path}")
    return resolved_path.read_text(encoding="utf-8")


def _serialize_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _batched(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def compute_text_similarity(reference_text: str, candidate_text: str) -> float:
    return SequenceMatcher(None, reference_text, candidate_text).ratio()


@dataclass
class PromptOptimizationSample:
    """One prompt-optimization training example."""

    name: str
    inputs: Dict[str, Any]
    reference_output: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        mapping: Dict[str, Any],
        base_dir: Optional[Path] = None,
        default_name: Optional[str] = None,
    ) -> "PromptOptimizationSample":
        data = dict(mapping)
        metadata = dict(data.pop("metadata", {}) or {})
        if base_dir is not None:
            metadata.setdefault("dataset_base_dir", str(base_dir.resolve()))

        name = data.pop("name", None) or data.pop("task_name", None) or default_name
        if not name:
            raise ValueError("Each optimization sample requires a name or task_name.")

        reference_output = (
            data.pop("reference_output", None)
            or data.pop("standard_task_description", None)
            or data.pop("ground_truth_output", None)
            or data.pop("reference_text", None)
        )
        reference_path = (
            data.pop("reference_output_path", None)
            or data.pop("standard_task_description_path", None)
            or data.pop("ground_truth_output_path", None)
            or data.pop("reference_text_path", None)
        )
        if reference_output is None and reference_path is not None:
            reference_output = _load_text_file(reference_path, base_dir=base_dir)
        if reference_output is None:
            raise ValueError(
                f"Sample '{name}' is missing reference_output or a supported reference path."
            )

        inputs = dict(data.pop("inputs", {}) or {})
        if not inputs:
            inputs = data

        return cls(
            name=name,
            inputs=inputs,
            reference_output=str(reference_output),
            metadata=metadata,
        )


@dataclass
class PromptOptimizationCheckpoint:
    """A saved prompt snapshot created during optimization."""

    epoch: int
    prompt_text: str
    path: Optional[Path] = None


@dataclass
class PromptOptimizationResult:
    """Final output of an optimization run."""

    target_name: str
    optimized_prompt: str
    checkpoints: List[PromptOptimizationCheckpoint]
    history: List[Dict[str, Any]]
    saved_prompt_path: Optional[Path] = None


@dataclass
class PromptTarget:
    """Definition of a prompt that can be optimized."""

    name: str
    input_renderer: Callable[[PromptOptimizationSample], str]
    initial_prompt: Optional[str] = None
    prompt_path: Optional[Path] = None
    constraints: List[str] = field(default_factory=list)
    role_description: str = "system_prompt_under_optimization"

    def load_initial_prompt(self) -> str:
        if self.initial_prompt is not None:
            return self.initial_prompt.strip()
        if self.prompt_path is not None:
            if not self.prompt_path.exists():
                raise FileNotFoundError(f"Prompt file not found: {self.prompt_path}")
            return self.prompt_path.read_text(encoding="utf-8").strip()
        raise ValueError(f"Prompt target '{self.name}' has no initial prompt configured.")

    def render_input(self, sample: PromptOptimizationSample) -> str:
        return self.input_renderer(sample)

    def save_prompt(
        self,
        prompt_text: str,
        output_path: Optional[str | Path] = None,
    ) -> Path:
        destination = (
            Path(output_path).expanduser().resolve()
            if output_path is not None
            else (
                self.prompt_path.resolve()
                if self.prompt_path is not None
                else None
            )
        )
        if destination is None:
            raise ValueError(
                f"Prompt target '{self.name}' has no save path. Provide output_path explicitly."
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(prompt_text, encoding="utf-8")
        return destination


class SafeLiteLLMEngine:
    """
    Runtime wrapper around textgrad LiteLLMEngine that forwards api_key/base_url.

    The wrapper keeps imports lazy so the module can be imported without textgrad/litellm
    installed, which is useful for static analysis and local tests.
    """

    def __init__(
        self,
        model_string: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ):
        tg, _, _, _ = _require_textgrad()
        completion = _require_litellm_completion()
        base_engine_cls = tg.engine.LiteLLMEngine

        class _RuntimeSafeLiteLLMEngine(base_engine_cls):
            def __init__(
                self,
                runtime_model_string: str,
                runtime_api_key: Optional[str] = None,
                runtime_base_url: Optional[str] = None,
                **runtime_kwargs: Any,
            ):
                super().__init__(runtime_model_string, **runtime_kwargs)
                self.api_key = runtime_api_key
                self.base_url = runtime_base_url

            def lite_llm_generate(
                self,
                content: str,
                system_prompt: Optional[str] = None,
                **runtime_kwargs: Any,
            ) -> str:
                response = completion(
                    model=self.model_string,
                    messages=[
                        {"role": "system", "content": system_prompt or ""},
                        {"role": "user", "content": content},
                    ],
                    api_key=self.api_key,
                    base_url=self.base_url,
                    **runtime_kwargs,
                )
                return response["choices"][0]["message"]["content"]

        self._engine = _RuntimeSafeLiteLLMEngine(
            model_string,
            runtime_api_key=api_key,
            runtime_base_url=base_url,
            **kwargs,
        )

    def __getattr__(self, item: str) -> Any:
        return getattr(self._engine, item)


class ReferenceTextComparisonStrategy:
    """Generic text-to-text comparison for prompt optimization."""

    DEFAULT_TEMPLATE = """
You are optimizing the system prompt for `{target_name}`.

Reference output:
{reference_output}

Model input used to produce the candidate output:
{rendered_input}

Sample metadata:
{metadata}

Your task:
1. Compare the generated output against the reference output.
2. Identify missing information, unsupported additions, structural weaknesses, and instruction-following gaps.
3. Focus on how the SYSTEM PROMPT should change, not how the sample input should change, unless the sample is inherently ambiguous.
4. Return short, concrete feedback that would help improve the system prompt on future samples.
"""

    def __init__(self, evaluation_template: Optional[str] = None):
        self.evaluation_template = evaluation_template or self.DEFAULT_TEMPLATE

    def build_loss(
        self,
        tg: Any,
        evaluator_engine: Any,
        sample: PromptOptimizationSample,
        target: PromptTarget,
        rendered_input: str,
    ) -> Any:
        eval_system_prompt = self.evaluation_template.format(
            target_name=target.name,
            reference_output=sample.reference_output,
            rendered_input=rendered_input,
            metadata=_serialize_json(sample.metadata or {}),
        )
        return tg.TextLoss(eval_system_prompt=eval_system_prompt, engine=evaluator_engine)


class TaskDescriptionComparisonStrategy(ReferenceTextComparisonStrategy):
    """Task-description-specific comparison guidance."""

    DEFAULT_TEMPLATE = """
You are optimizing the system prompt for a task-description generator named `{target_name}`.

Standard task_description:
{reference_output}

Generator input:
{rendered_input}

Sample metadata:
{metadata}

Evaluate the generated task_description against the standard one.
Focus on:
1. Coverage of the problem statement, forward model, and inverse objective.
2. Clarity of the input/output contract for `dataset/input.npy` and `output.npy`.
3. Quality of the reconstruction strategy, ordered implementation steps, and hyperparameters.
4. Whether implementation constraints, assumptions, and risks are captured.
5. Unsupported hallucinations or paper details that should not have been invented.

If the generated output misses critical content, explain what guidance the SYSTEM PROMPT should add.
If the generator added unsupported detail, explain how the SYSTEM PROMPT should become more conservative.
Return concise, prompt-edit-oriented feedback only.
"""

    def __init__(self, evaluation_template: Optional[str] = None):
        super().__init__(evaluation_template or self.DEFAULT_TEMPLATE)


def _render_task_generator_input(sample: PromptOptimizationSample) -> str:
    base_dir_value = sample.metadata.get("dataset_base_dir")
    base_dir = Path(base_dir_value) if base_dir_value else None

    paper_markdown = sample.inputs.get("paper_markdown")
    if paper_markdown is None and sample.inputs.get("paper_markdown_path"):
        paper_markdown = _load_text_file(
            sample.inputs["paper_markdown_path"],
            base_dir=base_dir,
        )
    if paper_markdown is None:
        raise ValueError(
            f"Sample '{sample.name}' is missing paper_markdown or paper_markdown_path."
        )

    user_prompt = sample.inputs.get("user_prompt")
    if not user_prompt:
        raise ValueError(f"Sample '{sample.name}' is missing user_prompt.")

    return TaskDescriptionGenerator.build_model_input(
        paper_markdown=str(paper_markdown),
        user_prompt=str(user_prompt),
        paper_markdown_char_limit=sample.inputs.get("paper_markdown_char_limit"),
    )


def build_task_generator_target(
    initial_prompt: Optional[str] = None,
    prompt_path: Optional[str | Path] = None,
    constraints: Optional[List[str]] = None,
) -> PromptTarget:
    resolved_prompt_path = (
        Path(prompt_path).expanduser().resolve()
        if prompt_path is not None
        else DEFAULT_SYSTEM_PROMPT_PATH.resolve()
    )
    return PromptTarget(
        name="task_generator",
        input_renderer=_render_task_generator_input,
        initial_prompt=initial_prompt,
        prompt_path=resolved_prompt_path,
        constraints=constraints
        or [
            "Preserve a clear Markdown output schema.",
            "Keep the prompt general enough for different inverse-problem papers.",
            "Encourage grounded assumptions instead of hallucinated implementation details.",
        ],
        role_description="task_generator_system_prompt",
    )


def load_optimization_dataset(dataset_path: str | Path) -> List[PromptOptimizationSample]:
    path = Path(dataset_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    if path.suffix.lower() in {".yaml", ".yml"}:
        raw = load_yaml(path)
    elif path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() == ".jsonl":
        raw = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        raise ValueError(
            "Unsupported dataset format. Use .json, .jsonl, .yaml, or .yml."
        )

    if isinstance(raw, dict):
        items = raw.get("samples") or raw.get("dataset") or raw.get("items")
        if items is None:
            raise ValueError(
                f"Dataset at {path} must contain a top-level list or a 'samples'/'dataset' key."
            )
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError(f"Unsupported dataset payload type: {type(raw)}")

    samples = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Dataset item #{index} is not a mapping: {item!r}")
        samples.append(
            PromptOptimizationSample.from_mapping(
                item,
                base_dir=path.parent,
                default_name=f"sample_{index:03d}",
            )
        )
    return samples


class PromptOptimizer:
    """Optimizes prompt text with textgrad and configurable comparison strategies."""

    def __init__(
        self,
        llm_config_path: str | Path,
        optimizer_model_key: str,
        output_dir: Optional[str | Path] = None,
    ):
        self.llm_config_path = Path(llm_config_path).expanduser().resolve()
        self.optimizer_model_key = optimizer_model_key
        self.output_dir = (
            Path(output_dir).expanduser().resolve()
            if output_dir is not None
            else (Path(__file__).parent / "optimized_prompts").resolve()
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_log_path = self.output_dir / "optimization_artifacts.jsonl"

    def _build_engine(self, model_key: str) -> SafeLiteLLMEngine:
        model_conf = get_llm_config(self.llm_config_path, model_key)
        model_name = model_conf.get("model_name", model_key)
        model_string = f"openai/{model_name}"
        return SafeLiteLLMEngine(
            model_string=model_string,
            api_key=model_conf.get("api_key"),
            base_url=model_conf.get("base_url"),
        )

    def _append_artifact(self, payload: Dict[str, Any]) -> None:
        with self.artifact_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")

    def generate_output(
        self,
        target: PromptTarget,
        sample: PromptOptimizationSample,
        generator_model_key: str,
        prompt_text: Optional[str] = None,
    ) -> str:
        tg, BlackboxLLM, Variable, _ = _require_textgrad()
        generator = BlackboxLLM(
            engine=self._build_engine(generator_model_key),
            system_prompt=Variable(
                prompt_text or target.load_initial_prompt(),
                requires_grad=False,
                role_description=target.role_description,
            ),
        )
        rendered_input = target.render_input(sample)
        generated_output = generator(
            Variable(
                rendered_input,
                requires_grad=False,
                role_description=f"{target.name}_input_for_generation",
            )
        )
        return generated_output.value

    def evaluate(
        self,
        target: PromptTarget,
        samples: Sequence[PromptOptimizationSample],
        generator_model_keys: Sequence[str],
        comparator: Optional[ReferenceTextComparisonStrategy] = None,
        prompt_text: Optional[str] = None,
        evaluator_model_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        tg, _, Variable, _ = _require_textgrad()
        comparator = comparator or ReferenceTextComparisonStrategy()
        evaluator_engine = self._build_engine(
            evaluator_model_key or self.optimizer_model_key
        )

        records = []
        for sample in samples:
            rendered_input = target.render_input(sample)
            for model_key in generator_model_keys:
                generated_text = self.generate_output(
                    target=target,
                    sample=sample,
                    generator_model_key=model_key,
                    prompt_text=prompt_text,
                )
                similarity = compute_text_similarity(
                    sample.reference_output, generated_text
                )
                feedback_node = comparator.build_loss(
                    tg=tg,
                    evaluator_engine=evaluator_engine,
                    sample=sample,
                    target=target,
                    rendered_input=rendered_input,
                )(
                    Variable(
                        generated_text,
                        requires_grad=False,
                        role_description=f"{target.name}_generated_output_for_eval",
                    )
                )
                records.append(
                    {
                        "sample_name": sample.name,
                        "generator_model": model_key,
                        "similarity": similarity,
                        "generated_output": generated_text,
                        "feedback": feedback_node.value,
                    }
                )

        average_similarity = (
            sum(record["similarity"] for record in records) / len(records)
            if records
            else 0.0
        )
        return {
            "target_name": target.name,
            "average_similarity": average_similarity,
            "records": records,
        }

    def optimize(
        self,
        target: PromptTarget,
        samples: Sequence[PromptOptimizationSample],
        generator_model_keys: Sequence[str],
        comparator: Optional[ReferenceTextComparisonStrategy] = None,
        epochs: int = 1,
        batch_size: int = 1,
        save_prompt_path: Optional[str | Path] = None,
        seed: int = 0,
    ) -> PromptOptimizationResult:
        if not samples:
            raise ValueError("Prompt optimization requires at least one sample.")
        if not generator_model_keys:
            raise ValueError("Prompt optimization requires at least one generator model.")

        tg, BlackboxLLM, Variable, TextualGradientDescent = _require_textgrad()
        comparator = comparator or ReferenceTextComparisonStrategy()
        evaluator_engine = self._build_engine(self.optimizer_model_key)
        tg.set_backward_engine(evaluator_engine)

        prompt_var = Variable(
            target.load_initial_prompt(),
            requires_grad=True,
            role_description=target.role_description,
        )
        generators = {
            model_key: BlackboxLLM(
                engine=self._build_engine(model_key),
                system_prompt=prompt_var,
            )
            for model_key in generator_model_keys
        }

        optimizer = TextualGradientDescent(
            parameters=[prompt_var],
            engine=evaluator_engine,
            constraints=target.constraints or DEFAULT_OPTIMIZER_CONSTRAINTS,
        )

        randomizer = random.Random(seed)
        working_samples = list(samples)
        history: List[Dict[str, Any]] = []
        checkpoints: List[PromptOptimizationCheckpoint] = []

        LOGGER.info(
            "Starting prompt optimization for %s with %d sample(s), %d epoch(s), models=%s",
            target.name,
            len(samples),
            epochs,
            ",".join(generator_model_keys),
        )

        for epoch in range(1, epochs + 1):
            randomizer.shuffle(working_samples)
            for batch_index, batch in enumerate(_batched(working_samples, batch_size), start=1):
                optimizer.zero_grad()
                loss_nodes = []
                batch_records = []

                for sample in batch:
                    rendered_input = target.render_input(sample)
                    input_var = Variable(
                        rendered_input,
                        requires_grad=False,
                        role_description=f"{target.name}_input_{sample.name}",
                    )

                    for model_key, generator in generators.items():
                        generated_output = generator(input_var)
                        generated_output.set_role_description(
                            f"{target.name}_generated_output_{model_key}"
                        )
                        loss_fn = comparator.build_loss(
                            tg=tg,
                            evaluator_engine=evaluator_engine,
                            sample=sample,
                            target=target,
                            rendered_input=rendered_input,
                        )
                        loss_node = loss_fn(generated_output)
                        loss_nodes.append(loss_node)

                        record = {
                            "epoch": epoch,
                            "batch_index": batch_index,
                            "sample_name": sample.name,
                            "generator_model": model_key,
                            "similarity": compute_text_similarity(
                                sample.reference_output, generated_output.value
                            ),
                            "generated_output": generated_output.value,
                            "feedback": loss_node.value,
                        }
                        batch_records.append(record)
                        self._append_artifact(record)

                for loss_node in loss_nodes:
                    loss_node.backward()
                optimizer.step()

                history.append(
                    {
                        "epoch": epoch,
                        "batch_index": batch_index,
                        "feedback_signals": len(loss_nodes),
                        "prompt_text": prompt_var.value,
                        "records": batch_records,
                    }
                )

            checkpoint_path = self.output_dir / f"{target.name}_epoch_{epoch:02d}.txt"
            checkpoint_path.write_text(prompt_var.value, encoding="utf-8")
            checkpoints.append(
                PromptOptimizationCheckpoint(
                    epoch=epoch,
                    prompt_text=prompt_var.value,
                    path=checkpoint_path,
                )
            )

        final_prompt_path = target.save_prompt(
            prompt_var.value,
            save_prompt_path or (self.output_dir / f"{target.name}_optimized.txt"),
        )

        return PromptOptimizationResult(
            target_name=target.name,
            optimized_prompt=prompt_var.value,
            checkpoints=checkpoints,
            history=history,
            saved_prompt_path=final_prompt_path,
        )


def _configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt optimization with textgrad")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to a YAML/JSON/JSONL dataset.",
    )
    parser.add_argument(
        "--llm-config",
        default=str((Path(__file__).parent.parent / "config" / "llm.yaml").resolve()),
        help="Path to llm.yaml.",
    )
    parser.add_argument(
        "--generator-models",
        required=True,
        help="Comma-separated model keys used to generate outputs.",
    )
    parser.add_argument(
        "--optimizer-model",
        required=True,
        help="Model key used by textgrad for comparison/backpropagation.",
    )
    parser.add_argument(
        "--target",
        default="task_generator",
        choices=["task_generator"],
        help="Which built-in prompt target to optimize.",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size.")
    parser.add_argument(
        "--prompt-path",
        default=None,
        help="Optional path to the prompt file being optimized.",
    )
    parser.add_argument(
        "--save-prompt-path",
        default=None,
        help="Optional final path to write the optimized prompt to.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for checkpoints and optimization artifacts.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    _configure_logging(verbose=args.verbose)

    samples = load_optimization_dataset(args.dataset)
    generator_models = [item.strip() for item in args.generator_models.split(",") if item.strip()]
    optimizer = PromptOptimizer(
        llm_config_path=args.llm_config,
        optimizer_model_key=args.optimizer_model,
        output_dir=args.output_dir,
    )

    if args.target == "task_generator":
        target = build_task_generator_target(prompt_path=args.prompt_path)
        comparator = TaskDescriptionComparisonStrategy()
    else:
        raise ValueError(f"Unsupported target: {args.target}")

    result = optimizer.optimize(
        target=target,
        samples=samples,
        generator_model_keys=generator_models,
        comparator=comparator,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_prompt_path=args.save_prompt_path,
        seed=args.seed,
    )

    summary = {
        "target_name": result.target_name,
        "saved_prompt_path": str(result.saved_prompt_path) if result.saved_prompt_path else None,
        "checkpoint_paths": [str(item.path) for item in result.checkpoints if item.path],
    }
    print(_serialize_json(summary))


if __name__ == "__main__":
    main()
