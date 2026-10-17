"""Model policy: IBM Granite defaults, and no model developed in China anywhere in config.

Rules-as-tests: this is the enforceable form of the project decision recorded
in ``config.py``'s module docstring. It covers every default in ``Settings``
and every configuration surface a deployment copies from (``src/``,
``.env.example``, ``mise.toml``, CI workflows).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from docling.datamodel.pipeline_options import TesseractCliOcrOptions, VlmPipelineOptions

from docling_llamaindex_pipeline.config import Settings

ROOT = Path(__file__).resolve().parents[2]

# Model families / vendors developed in China (lower-case substrings).
FORBIDDEN = re.compile(
    r"qwen|deepseek|\bbge[-_]|baai|chatglm|\bglm-?[0-9]|internlm|internvl|baichuan|"
    r"minimax|moonshot|\bkimi\b|\byi-[0-9]|paddle|rapidocr|hunyuan|ernie|doubao|step-[0-9]",
    re.IGNORECASE,
)

CONFIG_SURFACES = [
    ROOT / ".env.example",
    ROOT / "mise.toml",
    *sorted((ROOT / ".github").rglob("*.yml")),
]


def _code_strings(path: Path) -> list[tuple[int, str]]:
    """String literals in ``path`` that are values, not docstrings (prose may explain the policy)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                docstrings.add(id(body[0].value))
    return [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


def _defaults() -> dict[str, str]:
    settings = Settings(_env_file=None)  # pyright: ignore[reportCallIssue]
    return {name: str(getattr(settings, name)) for name in Settings.model_fields}


def test_generation_and_embedding_defaults_are_granite() -> None:
    defaults = _defaults()

    assert defaults["ollama_llm_model"] == "granite4.2:latest"
    assert defaults["ollama_embed_model"] == "granite-embedding:278m"
    assert defaults["chunk_tokenizer"] == "ibm-granite/granite-embedding-278m-multilingual"
    assert defaults["watsonx_embed_model"].startswith("ibm/")


def test_no_default_names_a_forbidden_model() -> None:
    offenders = {k: v for k, v in _defaults().items() if FORBIDDEN.search(v)}

    assert offenders == {}


def test_source_code_names_no_forbidden_model() -> None:
    offenders = [
        f"{path.relative_to(ROOT)}:{lineno}: {value!r}"
        for path in [
            *sorted((ROOT / "src").rglob("*.py")),
            *sorted((ROOT / "examples").glob("*.py")),
        ]
        for lineno, value in _code_strings(path)
        if FORBIDDEN.search(value)
    ]

    assert offenders == []


def test_config_surfaces_name_no_forbidden_model() -> None:
    offenders = [
        f"{path.relative_to(ROOT)}:{lineno}: {line.strip()}"
        for path in CONFIG_SURFACES
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if FORBIDDEN.search(line) and not line.lstrip().startswith("#")
    ]

    assert offenders == []


def test_policy_pattern_catches_known_names() -> None:
    for name in ("Qwen3-8B", "deepseek-r1", "bge-m3", "BAAI/bge-large", "glm-4", "rapidocr"):
        assert FORBIDDEN.search(name), name
    for name in ("granite4.2:latest", "granite-embedding:278m", "text-embedding-3-small"):
        assert not FORBIDDEN.search(name), name


def test_docling_defaults_we_rely_on_are_ibm_models() -> None:
    """Upstream defaults this policy depends on; a Docling bump that changes them must fail here."""
    vlm = VlmPipelineOptions().vlm_options
    spec = getattr(vlm, "model_spec", None)

    assert spec is not None
    assert spec.default_repo_id.startswith("ibm-granite/")
    assert TesseractCliOcrOptions.kind == "tesseract"
