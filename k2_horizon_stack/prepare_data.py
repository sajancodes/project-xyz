from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
from tokenizers import Tokenizer


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonl_docs(path: Path, text_field: str) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj.get(text_field)
            if isinstance(text, str) and text.strip():
                yield text


def hf_docs(dataset: str, dataset_config: str | None, split: str, text_field: str) -> Iterable[str]:
    from datasets import load_dataset

    kwargs = {"split": split, "streaming": True}
    ds = load_dataset(dataset, dataset_config, **kwargs) if dataset_config else load_dataset(dataset, **kwargs)
    for row in ds:
        text = row.get(text_field)
        if isinstance(text, str) and text.strip():
            yield text


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare an auditable packed token stream for Project XYZ")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", help="Hugging Face dataset id")
    src.add_argument("--jsonl", help="Local JSONL file")
    ap.add_argument("--dataset-config", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--eos-token-id", type=int, default=3)
    ap.add_argument("--max-docs", type=int, default=0, help="0 = unlimited")
    args = ap.parse_args()

    tokenizer_path = Path(args.tokenizer)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(str(tokenizer_path))

    vocab_size = tok.get_vocab_size()
    np_dtype = np.uint16 if vocab_size <= np.iinfo(np.uint16).max else np.uint32

    docs = (
        jsonl_docs(Path(args.jsonl), args.text_field)
        if args.jsonl
        else hf_docs(args.dataset, args.dataset_config, args.split, args.text_field)
    )

    doc_count = 0
    token_count = 0
    with output.open("wb") as f:
        for text in docs:
            if args.max_docs and doc_count >= args.max_docs:
                break
            ids = tok.encode(text).ids
            if args.eos_token_id >= 0:
                ids.append(args.eos_token_id)
            if not ids:
                continue
            arr = np.asarray(ids, dtype=np_dtype)
            arr.tofile(f)
            doc_count += 1
            token_count += int(arr.size)
            if doc_count % 1000 == 0:
                print(f"documents={doc_count:,} tokens={token_count:,}")

    if token_count < 2:
        raise RuntimeError("Prepared dataset contains fewer than 2 tokens")

    source = {
        "type": "jsonl" if args.jsonl else "huggingface",
        "value": os.path.abspath(args.jsonl) if args.jsonl else args.dataset,
        "configuration": args.dataset_config,
        "split": args.split,
        "text_field": args.text_field,
    }
    meta = {
        "format": "project-xyz-packed-token-stream-v1",
        "source": source,
        "tokenizer": str(tokenizer_path),
        "tokenizer_sha256": sha256(tokenizer_path),
        "vocab_size": vocab_size,
        "dtype": np.dtype(np_dtype).name,
        "eos_token_id": args.eos_token_id,
        "documents": doc_count,
        "tokens": token_count,
        "max_docs": args.max_docs,
    }
    meta_path = Path(str(output) + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"wrote: {output}")
    print(f"manifest: {meta_path}")


if __name__ == "__main__":
    main()
