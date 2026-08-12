#!/usr/bin/env python3
"""
EXPERIMENT TRACKING SYSTEM FOR PROJECT XYZ
==========================================
Records all experiments with full metadata for reproducibility.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from paths import EXPERIMENTS_DIR as PROJECT_EXPERIMENTS_DIR


EXPERIMENTS_DIR = Path(PROJECT_EXPERIMENTS_DIR)
RESULTS_FILE = EXPERIMENTS_DIR / "results.jsonl"
SUMMARY_FILE = EXPERIMENTS_DIR / "summary.json"


def load_results():
    """Load all experiment results."""
    if not RESULTS_FILE.exists():
        return []
    results = []
    with open(RESULTS_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def save_result(result):
    """Append a single experiment result."""
    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(result) + "\n")


def generate_experiment_id():
    """Generate unique experiment ID."""
    existing = load_results()
    return f"exp_{len(existing) + 1:04d}"


def record_experiment(
    name,
    parent_checkpoint,
    dataset,
    dataset_size,
    training_steps,
    tokens_processed,
    learning_rate,
    batch_size,
    sequence_length,
    optimizer,
    scheduler,
    training_loss,
    validation_loss,
    evaluation_scores,
    observations="",
    checkpoint_path="",
):
    """Record a complete experiment."""
    exp_id = generate_experiment_id()
    
    record = {
        "experiment_id": exp_id,
        "name": name,
        "timestamp": datetime.now().isoformat(),
        "parent_checkpoint": parent_checkpoint,
        "dataset": dataset,
        "dataset_size": dataset_size,
        "training_steps": training_steps,
        "tokens_processed": tokens_processed,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "training_loss": training_loss,
        "validation_loss": validation_loss,
        "evaluation_scores": evaluation_scores,
        "observations": observations,
        "checkpoint_path": checkpoint_path,
    }
    
    save_result(record)
    update_summary()
    return exp_id


def update_summary():
    """Generate human-readable summary."""
    results = load_results()
    
    summary = {
        "last_updated": datetime.now().isoformat(),
        "total_experiments": len(results),
        "experiments": []
    }
    
    for r in results:
        summary["experiments"].append({
            "id": r["experiment_id"],
            "name": r["name"],
            "timestamp": r["timestamp"],
            "parent": r["parent_checkpoint"],
            "steps": r["training_steps"],
            "tokens": r["tokens_processed"],
            "lr": r["learning_rate"],
            "batch": r["batch_size"],
            "train_loss": r["training_loss"],
            "val_loss": r["validation_loss"],
            "checkpoint": r["checkpoint_path"],
            "scores": r["evaluation_scores"],
        })
    
    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Also generate markdown table
    generate_markdown_summary(results)


def generate_markdown_summary(results):
    """Generate markdown summary table."""
    lines = [
        "# Experiment Summary",
        "",
        f"Last updated: {datetime.now().isoformat()}",
        f"Total experiments: {len(results)}",
        "",
        "| Exp ID | Name | Parent | Steps | Tokens | LR | Batch | Train Loss | Val Loss | Grammar | Factual QA | Entity Rel | Paraphrase | Instruction | Reasoning | Checkpoint |",
        "|--------|------|--------|-------|--------|----|-------|------------|----------|---------|------------|------------|------------|-------------|-----------|------------|",
    ]
    
    for r in results:
        scores = r.get("evaluation_scores", {})
        val_loss_str = f"{r['validation_loss']:.4f}" if r['validation_loss'] is not None else "N/A"
        lines.append(
            f"| {r['experiment_id']} | {r['name'][:20]} | "
            f"{r['parent_checkpoint'][:20]} | {r['training_steps']} | "
            f"{r['tokens_processed']:,} | {r['learning_rate']} | {r['batch_size']} | "
            f"{r['training_loss']:.4f} | {val_loss_str} | "
            f"{scores.get('grammar', 'N/A')} | {scores.get('factual_qa', 'N/A')} | "
            f"{scores.get('entity_relations', 'N/A')} | {scores.get('paraphrase', 'N/A')} | "
            f"{scores.get('instruction_following', 'N/A')} | {scores.get('reasoning', 'N/A')} | "
            f"{r['checkpoint_path']} |"
        )
    
    with open(EXPERIMENTS_DIR / "SUMMARY.md", "w") as f:
        f.write("\n".join(lines))


def get_checkpoint_lineage():
    """Extract checkpoint lineage from experiments."""
    results = load_results()
    lineage = {}
    for r in results:
        parent = r["parent_checkpoint"]
        child = r["checkpoint_path"]
        if parent not in lineage:
            lineage[parent] = []
        lineage[parent].append({
            "child": child,
            "experiment": r["experiment_id"],
            "name": r["name"],
            "steps": r["training_steps"],
        })
    return lineage


def print_lineage():
    """Print checkpoint lineage tree."""
    lineage = get_checkpoint_lineage()
    print("CHECKPOINT LINEAGE")
    print("="*60)
    
    def print_node(node, indent=0):
        for child_info in lineage.get(node, []):
            prefix = "  " * indent + "|-- "
            print(f"{prefix}{child_info['child']} ({child_info['name']}, {child_info['steps']} steps)")
            print_node(child_info['child'], indent + 1)
    
    # Find roots (checkpoints that are never children)
    all_children = set()
    for children in lineage.values():
        for c in children:
            all_children.add(c['child'])
    
    roots = [p for p in lineage.keys() if p not in all_children]
    if not roots:
        roots = list(lineage.keys())[:1]
    
    for root in roots:
        print(f"{root} (root)")
        print_node(root)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "lineage":
        print_lineage()
    else:
        update_summary()
        print("Summary updated.")


# Example usage:
if False:
    record_experiment(
        name="FineWeb Pretraining 0-106k",
        parent_checkpoint="none (random init)",
        dataset="FineWeb CC-MAIN-2025-26",
        dataset_size="streaming",
        training_steps=106000,
        tokens_processed=106000 * 16 * 512,  # 870M tokens
        learning_rate=3e-4,
        batch_size=16,
        sequence_length=512,
        optimizer="AdamW",
        scheduler="constant",
        training_loss=3.557,
        validation_loss=None,
        evaluation_scores={"grammar": 55.6, "factual_qa": 0.0, "entity_relations": 100.0},
        observations="Initial pretraining on FineWeb. Good grammar, no factual knowledge.",
        checkpoint_path="checkpoint-106k.pt"
    )
    
    record_experiment(
        name="Semantic Fine-tuning 5k steps",
        parent_checkpoint="checkpoint-106k.pt",
        dataset="Synthetic semantic/instruction data (41 examples x repeats)",
        dataset_size="~2000 unique tokens repeated",
        training_steps=5000,
        tokens_processed=5000 * 16 * 512,  # 41M tokens
        learning_rate=5e-5,
        batch_size=16,
        sequence_length=512,
        optimizer="AdamW",
        scheduler="constant",
        training_loss=0.005,
        validation_loss=None,
        evaluation_scores={"grammar": 33.3, "factual_qa": 100.0, "entity_relations": 100.0, "reasoning": 100.0},
        observations="Semantic fine-tuning dramatically improved factual QA and reasoning but hurt grammar. Catastrophic forgetting of general language.",
        checkpoint_path="checkpoint-semantic-5000.pt"
    )