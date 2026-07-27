# Copyright(C) 2026 Specific AI Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""CLI: split a classification model into head ``.npy`` + GGUF encoder.

Writes head ``.npy`` files into ``MODEL_DIR``, exports a temporary encoder-only
directory, converts it to GGUF via llama.cpp's ``convert_hf_to_gguf.py``, then
deletes the temporary encoder directory.

Requires: pip install "specific-ai-tools[split]"  (adds torch)

Example:

    .venv/bin/python scripts/split_model.py /path/to/model --outtype f32
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from specific_ai_tools.embedding_heads.split import (
    DEFAULT_ENCODER_DIRNAME,
    DEFAULT_GGUF_FILENAME,
    split_classification_model,
)

DEFAULT_LLAMACPP_REPO = "https://github.com/ggml-org/llama.cpp.git"
DEFAULT_LLAMACPP_DIR = Path.home() / ".cache" / "specific-ai-tools" / "llama.cpp"


def ensure_llamacpp_repo(repo_url: str, llamacpp_dir: Path) -> Path:
    """Clone ``repo_url`` into ``llamacpp_dir`` when missing; return the path."""
    convert_script = llamacpp_dir / "convert_hf_to_gguf.py"
    if convert_script.is_file():
        return llamacpp_dir

    if llamacpp_dir.exists() and any(llamacpp_dir.iterdir()):
        raise FileNotFoundError(
            f"{llamacpp_dir} exists but convert_hf_to_gguf.py was not found. "
            "Pass --llamacpp-dir pointing at a llama.cpp checkout, or remove the directory."
        )

    llamacpp_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Cloning llama.cpp from {repo_url} into {llamacpp_dir} …")
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(llamacpp_dir)],
        check=True,
    )
    if not convert_script.is_file():
        raise FileNotFoundError(f"convert_hf_to_gguf.py missing after clone: {convert_script}")
    return llamacpp_dir


def convert_encoder_to_gguf(
    *,
    encoder_dir: Path,
    outfile: Path,
    outtype: str,
    llamacpp_dir: Path,
) -> Path:
    """Run llama.cpp ``convert_hf_to_gguf.py`` and return ``outfile``."""
    convert_script = llamacpp_dir / "convert_hf_to_gguf.py"
    cmd = [
        sys.executable,
        str(convert_script),
        str(encoder_dir),
        "--outfile",
        str(outfile),
        "--outtype",
        outtype,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    if not outfile.is_file():
        raise FileNotFoundError(f"GGUF conversion finished but outfile missing: {outfile}")
    return outfile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_dir",
        type=Path,
        help="Path to a Hugging Face sequence-classification model directory",
    )
    parser.add_argument(
        "--outtype",
        default=None,
        help="GGUF outtype for convert_hf_to_gguf.py (e.g. f32, f16, bf16). "
        "When omitted, derived from the model config torch_dtype/dtype.",
    )
    parser.add_argument(
        "--llamacpp-repo",
        default=DEFAULT_LLAMACPP_REPO,
        help=f"llama.cpp git URL (default: {DEFAULT_LLAMACPP_REPO})",
    )
    parser.add_argument(
        "--llamacpp-dir",
        type=Path,
        default=DEFAULT_LLAMACPP_DIR,
        help=f"Local llama.cpp checkout (default: {DEFAULT_LLAMACPP_DIR})",
    )
    parser.add_argument(
        "--gguf-filename",
        default=DEFAULT_GGUF_FILENAME,
        help=f"Output GGUF filename inside model_dir (default: {DEFAULT_GGUF_FILENAME})",
    )
    parser.add_argument(
        "--encoder-dirname",
        default=DEFAULT_ENCODER_DIRNAME,
        help=f"Temporary encoder-only subdirectory name (default: {DEFAULT_ENCODER_DIRNAME})",
    )
    args = parser.parse_args(argv)

    model_dir = args.model_dir.expanduser().resolve()
    gguf_path = model_dir / Path(args.gguf_filename).name
    encoder_dir: Path | None = None

    try:
        result = split_classification_model(
            model_dir,
            encoder_dirname=args.encoder_dirname,
            outtype=args.outtype,
        )
        encoder_dir = result.encoder_dir
        print(f"Wrote head weights: {[str(p) for p in result.npy_paths]}")
        print(f"Exported encoder to {encoder_dir} (outtype={result.outtype})")

        llamacpp_dir = ensure_llamacpp_repo(args.llamacpp_repo, args.llamacpp_dir.expanduser().resolve())
        convert_encoder_to_gguf(
            encoder_dir=encoder_dir,
            outfile=gguf_path,
            outtype=result.outtype,
            llamacpp_dir=llamacpp_dir,
        )
        print(f"Wrote GGUF: {gguf_path}")
    finally:
        if encoder_dir is not None and encoder_dir.is_dir():
            shutil.rmtree(encoder_dir)
            print(f"Removed temporary encoder dir: {encoder_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
