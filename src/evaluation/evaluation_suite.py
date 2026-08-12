#!/usr/bin/env python3
"""
COMPREHENSIVE EVALUATION SUITE FOR PROJECT XYZ
===============================================
Tests: Grammar, Factual QA, Reading Comprehension, Entity Relations,
Pronoun/Reference Understanding, Paraphrase Understanding,
Instruction Following, Context Retention, Simple Reasoning,
Natural Language Relevance, Repetition, Hallucination, OOD Wording

All tests use deterministic evaluation where possible.
"""

import torch
import json
import os
from datetime import datetime
from tokenizers import Tokenizer
from model import SmallEnglishLLM, ModelConfig


class EvaluationSuite:
    def __init__(self, checkpoint_path, tokenizer_path="tokenizer.json", device="cuda"):
        self.checkpoint_path = checkpoint_path
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.config = ModelConfig()
        self.device = torch.device(device)
        self.model = SmallEnglishLLM(self.config).to(self.device)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        
        self.checkpoint_info = {
            "step": checkpoint.get("step", "unknown"),
            "loss": checkpoint.get("loss", "unknown"),
            "base_checkpoint": checkpoint.get("base_checkpoint", "none"),
            "training_type": checkpoint.get("training_type", "unknown"),
        }
        
    def generate(self, prompt, max_new_tokens=80, temperature=0.7, top_k=40, top_p=None):
        """Generate text with various sampling strategies."""
        encoded = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                input_context = input_ids[:, -self.config.max_seq_len:]
                logits, _ = self.model(input_context)
                next_logits = logits[:, -1, :] / temperature
                
                # Top-k filtering
                if top_k is not None:
                    values, indices = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    filtered = torch.full_like(next_logits, float("-inf"))
                    filtered.scatter_(1, indices, values)
                    next_logits = filtered
                
                # Top-p (nucleus) filtering
                if top_p is not None:
                    probs = torch.softmax(next_logits, dim=-1)
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumsum_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_logits[indices_to_remove] = float("-inf")
                
                probabilities = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                if next_token.item() == 3:  # EOS
                    break
                    
        generated_ids = input_ids[0].tolist()
        return self.tokenizer.decode(generated_ids)
    
    def get_continuation(self, prompt, **gen_kwargs):
        """Get only the generated continuation, not the prompt."""
        full = self.generate(prompt, **gen_kwargs)
        if full.startswith(prompt):
            return full[len(prompt):].strip()
        return full.strip()
    
    def get_next_token_probs(self, prompt):
        """Get probability distribution over next token."""
        encoded = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits, _ = self.model(input_ids)
        next_token_logits = logits[0, -1]
        return torch.softmax(next_token_logits, dim=-1)
    
    # ============================================================
    # 1. GRAMMAR TESTS
    # ============================================================
    def test_grammar(self):
        """Test subject-verb agreement."""
        tests = [
            ("I ", ["am", "is", "are"], "am"),
            ("He ", ["am", "is", "are"], "is"),
            ("She ", ["am", "is", "are"], "is"),
            ("They ", ["am", "is", "are"], "are"),
            ("We ", ["am", "is", "are"], "are"),
            ("You ", ["am", "is", "are"], "are"),
            ("It ", ["am", "is", "are"], "is"),
            ("The dog ", ["is", "are"], "is"),
            ("The dogs ", ["is", "are"], "are"),
        ]
        
        correct = 0
        results = []
        for prefix, candidates, expected in tests:
            probs = self.get_next_token_probs(prefix)
            best_candidate = None
            best_prob = -1
            for c in candidates:
                c_ids = self.tokenizer.encode(c).ids
                if len(c_ids) == 1:
                    p = probs[c_ids[0]].item()
                    if p > best_prob:
                        best_prob = p
                        best_candidate = c
            
            is_correct = best_candidate == expected
            if is_correct:
                correct += 1
            results.append({
                "prefix": prefix.strip(),
                "candidates": candidates,
                "expected": expected,
                "predicted": best_candidate,
                "prob": best_prob,
                "correct": is_correct
            })
        
        return {
            "category": "grammar",
            "accuracy": correct / len(tests),
            "correct": correct,
            "total": len(tests),
            "details": results
        }
    
    # ============================================================
    # 2. FACTUAL QA TESTS
    # ============================================================
    def test_factual_qa(self):
        """Test basic factual knowledge."""
        tests = [
            ("User: Where does the sun rise?\nAssistant:", "east"),
            ("User: What direction does the sun rise from?\nAssistant:", "east"),
            ("User: What color is grass usually?\nAssistant:", "green"),
            ("User: What color is the sky on a clear day?\nAssistant:", "blue"),
            ("User: What do humans breathe?\nAssistant:", "air"),
            ("User: What animal says meow?\nAssistant:", "cat"),
            ("User: What animal says bark?\nAssistant:", "dog"),
        ]
        
        correct = 0
        results = []
        for prompt, expected_keyword in tests:
            response = self.get_continuation(prompt, max_new_tokens=30, temperature=0.3)
            # Check if expected keyword appears in response
            found = expected_keyword.lower() in response.lower()
            if found:
                correct += 1
            results.append({
                "prompt": prompt,
                "expected_keyword": expected_keyword,
                "response": response[:100],
                "correct": found
            })
        
        return {
            "category": "factual_qa",
            "accuracy": correct / len(tests),
            "correct": correct,
            "total": len(tests),
            "details": results
        }
    
    # ============================================================
    # 3. ENTITY RELATION TESTS
    # ============================================================
    def test_entity_relations(self):
        """Test understanding of who did what to whom."""
        tests = [
            ("User: Ravi gave Sajan a book. Who received the book?\nAssistant:", "Sajan"),
            ("User: Ravi gave Sajan a book. Who gave the book?\nAssistant:", "Ravi"),
            ("User: The dog chased the cat. Who chased the cat?\nAssistant:", "dog"),
            ("User: The cat chased the dog. What did the cat chase?\nAssistant:", "dog"),
            ("User: Sajan gave Ravi an apple. Who received the apple?\nAssistant:", "Ravi"),
        ]
        
        correct = 0
        results = []
        for prompt, expected in tests:
            response = self.get_continuation(prompt, max_new_tokens=30, temperature=0.3)
            found = expected.lower() in response.lower()
            if found:
                correct += 1
            results.append({
                "prompt": prompt,
                "expected": expected,
                "response": response[:100],
                "correct": found
            })
        
        return {
            "category": "entity_relations",
            "accuracy": correct / len(tests),
            "correct": correct,
            "total": len(tests),
            "details": results
        }
    
    # ============================================================
    # 4. PARAPHRASE UNDERSTANDING
    # ============================================================
    def test_paraphrase(self):
        """Test generalization to different wording of same concept."""
        tests = [
            ("User: The child is asleep. What is the child doing?\nAssistant:", "sleeping"),
            ("User: The dog is asleep. What is the dog doing?\nAssistant:", "sleeping"),
            ("User: The boy is running. What action is the boy doing?\nAssistant:", "running"),
        ]
        
        correct = 0
        results = []
        for prompt, expected in tests:
            response = self.get_continuation(prompt, max_new_tokens=30, temperature=0.3)
            found = expected.lower() in response.lower()
            if found:
                correct += 1
            results.append({
                "prompt": prompt,
                "expected": expected,
                "response": response[:100],
                "correct": found
            })
        
        return {
            "category": "paraphrase",
            "accuracy": correct / len(tests),
            "correct": correct,
            "total": len(tests),
            "details": results
        }
    
    # ============================================================
    # 5. INSTRUCTION FOLLOWING
    # ============================================================
    def test_instruction_following(self):
        """Test ability to follow explicit instructions."""
        tests = [
            ("User: Complete this sentence: I ___ Sajan.\nAssistant:", "am"),
            ("User: Complete this sentence: He ___ happy.\nAssistant:", "is"),
            ("User: Complete this sentence: They ___ happy.\nAssistant:", "are"),
            ("User: Complete this sentence: We ___ students.\nAssistant:", "are"),
            ("User: Complete this sentence: She ___ here.\nAssistant:", "is"),
            ("User: Give me three animals.\nAssistant:", None),  # Open-ended
        ]
        
        correct = 0
        results = []
        for prompt, expected in tests:
            response = self.get_continuation(prompt, max_new_tokens=30, temperature=0.3)
            if expected is not None:
                found = expected.lower() in response.lower().split()[0] if response else False
                if found:
                    correct += 1
                results.append({
                    "prompt": prompt,
                    "expected": expected,
                    "response": response[:100],
                    "correct": found
                })
            else:
                # Open-ended: just check it generates something reasonable
                results.append({
                    "prompt": prompt,
                    "expected": "any three animals",
                    "response": response[:100],
                    "correct": len(response.split()) > 2
                })
                if len(response.split()) > 2:
                    correct += 1
        
        return {
            "category": "instruction_following",
            "accuracy": correct / len(tests),
            "correct": correct,
            "total": len(tests),
            "details": results
        }
    
    # ============================================================
    # 6. CONTEXT RETENTION / CONVERSATIONAL MEMORY
    # ============================================================
    def test_context_retention(self):
        """Test memory across conversation turns."""
        # Multi-turn: establish context, then query
        prompt1 = "User: My name is Sajan.\nAssistant:"
        response1 = self.get_continuation(prompt1, max_new_tokens=30, temperature=0.3)
        
        # Now test if model remembers
        prompt2 = "User: My name is Sajan.\nAssistant: Your name is Sajan.\nUser: What is my name?\nAssistant:"
        response2 = self.get_continuation(prompt2, max_new_tokens=30, temperature=0.3)
        
        found = "Sajan" in response2
        
        return {
            "category": "context_retention",
            "accuracy": 1.0 if found else 0.0,
            "correct": 1 if found else 0,
            "total": 1,
            "details": [{
                "setup": "User: My name is Sajan.",
                "query": "User: What is my name?",
                "response": response2[:100],
                "correct": found
            }]
        }
    
    # ============================================================
    # 7. SIMPLE REASONING
    # ============================================================
    def test_reasoning(self):
        """Test basic logical reasoning."""
        tests = [
            ("User: All birds have wings. A robin is a bird. Does a robin have wings?\nAssistant:", "yes"),
            ("User: All cats are animals. Milo is a cat. Is Milo an animal?\nAssistant:", "yes"),
            ("User: If something is bigger than a box, and the box is bigger than a cup, what is bigger: the first thing or the cup?\nAssistant:", "first thing"),
            ("User: What is 2 + 2?\nAssistant:", "4"),
            ("User: What is 5 + 3?\nAssistant:", "8"),
            ("User: What is 10 - 4?\nAssistant:", "6"),
        ]
        
        correct = 0
        results = []
        for prompt, expected in tests:
            response = self.get_continuation(prompt, max_new_tokens=30, temperature=0.3)
            found = expected.lower() in response.lower()
            if found:
                correct += 1
            results.append({
                "prompt": prompt,
                "expected": expected,
                "response": response[:100],
                "correct": found
            })
        
        return {
            "category": "reasoning",
            "accuracy": correct / len(tests),
            "correct": correct,
            "total": len(tests),
            "details": results
        }
    
    # ============================================================
    # 8. HALLUCINATION / REPETITION DETECTION
    # ============================================================
    def test_repetition_hallucination(self):
        """Test for degenerate repetition and hallucination."""
        prompt = "User: The child is asleep. What is the child doing?\nAssistant:"
        response = self.get_continuation(prompt, max_new_tokens=60, temperature=0.7)
        
        # Check for excessive repetition
        words = response.split()
        unique_words = set(words)
        repetition_ratio = 1 - (len(unique_words) / len(words)) if words else 0
        
        # Check for "running" when should be "sleeping" (hallucination)
        hallucinated = "running" in response.lower() and "sleep" not in response.lower()
        
        return {
            "category": "repetition_hallucination",
            "repetition_ratio": repetition_ratio,
            "hallucinated": hallucinated,
            "response": response[:200],
            "details": [{
                "prompt": prompt,
                "response": response[:200],
                "repetition_ratio": repetition_ratio,
                "hallucinated_wrong_action": hallucinated
            }]
        }
    
    # ============================================================
    # 9. OUT-OF-DISTRIBUTION WORDING
    # ============================================================
    def test_ood_generalization(self):
        """Test generalization to unseen phrasings."""
        tests = [
            # Paraphrases of sun rising fact
            ("User: Which direction does the sun come up from?\nAssistant:", "east"),
            ("User: In which direction does sunrise occur?\nAssistant:", "east"),
            ("User: The sun comes up in the ___\nAssistant:", "east"),
            # Unseen entity relation
            ("User: Maria gave Ahmed a pen. Who got the pen?\nAssistant:", "Ahmed"),
            # Unseen grammar
            ("User: The bird ___ flying.\nAssistant:", "is"),
        ]
        
        correct = 0
        results = []
        for prompt, expected in tests:
            response = self.get_continuation(prompt, max_new_tokens=30, temperature=0.3)
            found = expected.lower() in response.lower()
            if found:
                correct += 1
            results.append({
                "prompt": prompt,
                "expected": expected,
                "response": response[:100],
                "correct": found
            })
        
        return {
            "category": "ood_generalization",
            "accuracy": correct / len(tests),
            "correct": correct,
            "total": len(tests),
            "details": results
        }
    
    def run_all(self):
        """Run complete evaluation suite."""
        print(f"="*70)
        print(f"EVALUATION SUITE: {self.checkpoint_path}")
        print(f"Step: {self.checkpoint_info['step']} | Loss: {self.checkpoint_info['loss']}")
        print(f"Base: {self.checkpoint_info['base_checkpoint']} | Type: {self.checkpoint_info['training_type']}")
        print(f"="*70)
        
        all_results = {
            "checkpoint": self.checkpoint_path,
            "checkpoint_info": self.checkpoint_info,
            "timestamp": datetime.now().isoformat(),
            "tests": {}
        }
        
        test_methods = [
            ("grammar", self.test_grammar),
            ("factual_qa", self.test_factual_qa),
            ("entity_relations", self.test_entity_relations),
            ("paraphrase", self.test_paraphrase),
            ("instruction_following", self.test_instruction_following),
            ("context_retention", self.test_context_retention),
            ("reasoning", self.test_reasoning),
            ("repetition_hallucination", self.test_repetition_hallucination),
            ("ood_generalization", self.test_ood_generalization),
        ]
        
        for name, method in test_methods:
            print(f"\nRunning {name}...")
            try:
                result = method()
                all_results["tests"][name] = result
                
                if "accuracy" in result:
                    print(f"  Accuracy: {result['accuracy']*100:.1f}% ({result['correct']}/{result['total']})")
                elif "repetition_ratio" in result:
                    print(f"  Repetition: {result['repetition_ratio']*100:.1f}% | Hallucinated: {result['hallucinated']}")
            except Exception as e:
                print(f"  ERROR: {e}")
                all_results["tests"][name] = {"error": str(e)}
        
        # Summary
        print(f"\n{'='*70}")
        print(f"EVALUATION SUMMARY")
        print(f"{'='*70}")
        for name, result in all_results["tests"].items():
            if "accuracy" in result:
                print(f"  {name:30s}: {result['accuracy']*100:5.1f}%")
        
        return all_results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to checkpoint .pt file")
    parser.add_argument("--tokenizer", default="tokenizer.json", help="Tokenizer path")
    parser.add_argument("--output", help="Output JSON file for results")
    args = parser.parse_args()
    
    suite = EvaluationSuite(args.checkpoint, args.tokenizer)
    results = suite.run_all()
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    return results


if __name__ == "__main__":
    main()