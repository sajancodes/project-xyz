# Experiment Summary

Last updated: 2026-08-12T21:11:51.964146
Total experiments: 14

| Exp ID | Name | Parent | Steps | Tokens | LR | Batch | Train Loss | Val Loss | Grammar | Factual QA | Entity Rel | Paraphrase | Instruction | Reasoning | Checkpoint |
|--------|------|--------|-------|--------|----|-------|------------|----------|---------|------------|------------|------------|-------------|-----------|------------|
| exp_0001 | FineWeb Pretraining  | none (random init) | 106000 | 868,352,000 | 0.0003 | 16 | 3.5570 | N/A | 55.6 | 0.0 | 100.0 | 33.3 | 16.7 | 16.7 | checkpoint-106k.pt |
| exp_0003 | Semantic Fine-tuning | checkpoint-106k.pt | 5000 | 40,960,000 | 5e-05 | 16 | 0.0050 | N/A | 33.3 | 100.0 | 100.0 | 66.7 | 16.7 | 100.0 | checkpoint-semantic-5000.pt |
| exp_0004 | FineWeb Continued 10 | checkpoint-106k.pt | 4000 | 32,768,000 | 0.0003 | 16 | 0.0020 | N/A | 44.4 | 28.6 | 100.0 | 33.3 | 16.7 | 16.7 | checkpoint_fineweb.pt |
| exp_0004 | Mixed FineWeb+Semant | checkpoint-106k.pt | 1000 | 8,192,000 | 0.0001 | 16 | 3.1860 | N/A | 66.7 | 85.7 | 100.0 | 100.0 | 16.7 | 100.0 | checkpoint-mixed-1k.pt |
| exp_0005 | Mixed FineWeb+Semant | checkpoint-106k.pt | 2000 | 16,384,000 | 0.0001 | 16 | 3.5470 | N/A | 55.6 | 100.0 | 100.0 | 100.0 | 16.7 | 100.0 | checkpoint-mixed-2k.pt |
| exp_0006 | Mixed FineWeb+Semant | checkpoint-106k.pt | 3000 | 24,576,000 | 0.0001 | 16 | 2.4160 | N/A | 55.6 | 57.1 | 100.0 | 100.0 | 16.7 | 66.7 | checkpoint-mixed-3k.pt |
| exp_0007 | Mixed FineWeb+Semant | checkpoint-106k.pt | 4000 | 32,768,000 | 0.0001 | 16 | 2.6530 | N/A | 55.6 | 71.4 | 100.0 | 100.0 | 16.7 | 83.3 | checkpoint-mixed-4k.pt |
| exp_0008 | Mixed FineWeb+Semant | checkpoint-106k.pt | 5000 | 40,960,000 | 0.0001 | 16 | 3.5330 | N/A | 44.4 | 42.9 | 100.0 | 100.0 | 16.7 | 66.7 | checkpoint-mixed-5k.pt |
| exp_0009 | Mixed FineWeb+Semant | checkpoint-106k.pt | 6000 | 49,152,000 | 0.0001 | 16 | 3.2470 | N/A | 44.4 | 42.9 | 100.0 | 100.0 | 16.7 | 66.7 | checkpoint-mixed-6k.pt |
| exp_0010 | Mixed FineWeb+Semant | checkpoint-106k.pt | 7000 | 57,344,000 | 0.0001 | 16 | 3.7130 | N/A | 44.4 | 71.4 | 100.0 | 100.0 | 16.7 | 50.0 | checkpoint-mixed-7k.pt |
| exp_0011 | Mixed FineWeb+Semant | checkpoint-106k.pt | 8000 | 65,536,000 | 0.0001 | 16 | 3.3480 | N/A | 44.4 | 42.9 | 100.0 | 100.0 | 16.7 | 66.7 | checkpoint-mixed-8k.pt |
| exp_0012 | Mixed FineWeb+Semant | checkpoint-106k.pt | 10000 | 81,920,000 | 0.0001 | 16 | 3.3610 | N/A | 44.4 | 28.6 | 100.0 | 100.0 | 16.7 | 33.3 | checkpoint-mixed-10k.pt |
| exp_0013 | Semantic V2 Mixed (5 | checkpoint-106k.pt ( | 5000 | 40,960,000 | 0.0001 | 16 | 2.8435 | 3.7246 | N/A | N/A | N/A | N/A | N/A | N/A | checkpoints/semantic_v2/checkpoint_exp1_latest.pt |
| exp_0014 | Semantic V2 EXP-2 (3 | checkpoint_exp1_late | 3000 | 24,576,000 | 5e-05 | 16 | 3.4830 | 3.5742 | N/A | N/A | N/A | N/A | N/A | N/A | checkpoints/semantic_v2/checkpoint_exp2_latest.pt |