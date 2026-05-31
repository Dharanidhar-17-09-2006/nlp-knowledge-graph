"""
train_rebel_mac.py
==================
Fine-tunes Babelscape/rebel-large using your local dataset.
Optimized for Apple Silicon (MPS) with exact Triplet F1 evaluation.
"""

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)

# ══════════════════════════════════════════════════════════════════
# 1. REBEL FORMAT CODEC
# ══════════════════════════════════════════════════════════════════

def encode_triplets(triplets: List[Tuple[str, str, str]]) -> str:
    """(subj, rel, obj) → REBEL decoder target string."""
    return " ".join(f"<triplet> {s} <subj> {o} <obj> {r}" for s, r, o in triplets)

def decode_triplets(text: str) -> List[Tuple[str, str, str]]:
    """REBEL decoder output → (subj, rel, obj) list."""
    results = []
    text = re.sub(r"<s>|</s>|<pad>", "", text).strip()
    for part in text.split("<triplet>"):
        part = part.strip()
        if not part or "<subj>" not in part or "<obj>" not in part:
            continue
        try:
            s_raw, rest  = part.split("<subj>", 1)
            o_raw, r_raw = rest.split("<obj>",  1)
            clean = lambda x: re.sub(r"<[^>]+>", "", x).strip()
            s, o, r = clean(s_raw), clean(o_raw), clean(r_raw)
            if s and o and r and len(s) > 1 and len(o) > 1:
                results.append((s, r, o))
        except (ValueError, IndexError):
            continue
    return results

# ══════════════════════════════════════════════════════════════════
# 2. DATA STRUCTURES & LOADERS
# ══════════════════════════════════════════════════════════════════

@dataclass
class Example:
    sentence: str
    triplets: List[Tuple[str, str, str]]

def load_local_rebel(path: str = "./rebel_dataset/en_train.jsonl", max_samples: int = 5000) -> List[Example]:
    if not os.path.exists(path):
        print(f"[Error] Local dataset not found at {path}")
        return []

    print(f"[DataLoader] Streaming local data from: {path}")
    examples = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if len(examples) >= max_samples: break
            try:
                item = json.loads(line)
                sentence = (item.get("text") or "").strip()
                raw_triples = item.get("triples", [])
                triplets = []
                for t in raw_triples:
                    s = t.get("subject", {}).get("surfaceform", "")
                    p = t.get("predicate", {}).get("surfaceform", "")
                    o = t.get("object", {}).get("surfaceform", "")
                    if s and p and o and not re.match(r"^Q\d+$", s):
                        triplets.append((s, p.lower(), o))
                if sentence and triplets:
                    examples.append(Example(sentence=sentence, triplets=triplets))
                else: skipped += 1
            except:
                skipped += 1
                continue
    print(f"[DataLoader] Loaded {len(examples)} examples (Skipped {skipped})")
    return examples

# Domain Data tailored for IITK Context - Relations now tightly match RELATION_MAP keys
DOMAIN_DATA = [
    Example("Srishwan is a student at IIT Kharagpur.", 
            [("Srishwan", "educated at", "IIT Kharagpur"), ("Srishwan", "student at", "IIT Kharagpur")]),
    Example("Aaron Jason Baptist studies in the AI Department at IIT Kharagpur.", 
            [("Aaron Jason Baptist", "member of", "AI Department"), ("AI Department", "part of", "IIT Kharagpur")]),
    Example("The student resides in Lal Bahadur Shastri Hall.", 
            [("student", "resides in", "Lal Bahadur Shastri Hall")]),
    Example("Chandan is enrolled in Computer Science at IIT Kharagpur.", 
            [("Chandan", "studies", "Computer Science"), ("Chandan", "educated at", "IIT Kharagpur")]),
    Example("He stays in Nehru Hall in West Bengal.", 
            [("He", "stays in", "Nehru Hall"), ("Nehru Hall", "located in", "West Bengal")]),
]

# ══════════════════════════════════════════════════════════════════
# 3. PYTORCH DATASET
# ══════════════════════════════════════════════════════════════════

class REBELDataset(Dataset):
    def __init__(self, examples: List[Example], tokenizer, max_src: int = 256, max_tgt: int = 256):
        self.tok = tokenizer
        self.records = []
        for ex in examples:
            try:
                target = encode_triplets(ex.triplets)
                if not target.strip(): continue
                enc = self.tok(ex.sentence, max_length=max_src, padding="max_length", truncation=True)
                tgt = self.tok(text_target=target, max_length=max_tgt, padding="max_length", truncation=True)
                labels = [(lid if lid != self.tok.pad_token_id else -100) for lid in tgt["input_ids"]]
                self.records.append({
                    "input_ids": enc["input_ids"], 
                    "attention_mask": enc["attention_mask"], 
                    "labels": labels
                })
            except: continue

    def __len__(self): return len(self.records)
    def __getitem__(self, i): return {k: torch.tensor(v) for k, v in self.records[i].items()}

# ══════════════════════════════════════════════════════════════════
# 4. F1 METRIC CALLBACK
# ══════════════════════════════════════════════════════════════════

class TripletF1Callback(TrainerCallback):
    def __init__(self, tokenizer, val_examples: List[Example], device: str):
        self.tok = tokenizer
        self.samples = val_examples[:100]
        self.device = device

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None: return
        model.eval()
        tp = fp = fn = 0
        with torch.no_grad():
            for ex in self.samples:
                inp = self.tok(ex.sentence, return_tensors="pt", truncation=True, max_length=256).to(self.device)
                try:
                    out = model.generate(inp["input_ids"], max_length=256, num_beams=4)
                    decoded = self.tok.decode(out[0], skip_special_tokens=False)
                    pred = set((s.lower(), r.lower(), o.lower()) for s, r, o in decode_triplets(decoded))
                except: pred = set()
                
                gold = set((s.lower(), r.lower(), o.lower()) for s, r, o in ex.triplets)
                tp += len(pred & gold)
                fp += len(pred - gold)
                fn += len(gold - pred)
        
        P = tp/(tp+fp) if (tp+fp)>0 else 0
        R = tp/(tp+fn) if (tp+fn)>0 else 0
        F1 = (2*P*R/(P+R)) if (P+R)>0 else 0
        print(f"\n[Epoch {state.epoch}] Precision: {P:.4f} | Recall: {R:.4f} | F1: {F1:.4f}\n")
        model.train()

# ══════════════════════════════════════════════════════════════════
# 5. TRAINER
# ══════════════════════════════════════════════════════════════════

class REBELTrainer:
    def __init__(self, model_name: str = "Babelscape/rebel-large", output_dir: str = "./rebel-finetuned"):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"[Trainer] Using device: {self.device}")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.output_dir = output_dir

    def train(self, examples: List[Example], epochs: int = 6, batch: int = 2,
              grad_acc: int = 4, early_stopping_patience: int = 2):
        import random
        random.shuffle(examples)
        n = max(10, int(len(examples) * 0.1))
        train_raw, val_raw = examples[:-n], examples[-n:]

        print(f"[Trainer] Train: {len(train_raw)}  Val: {len(val_raw)}")

        total_steps = (len(train_raw) // (batch * grad_acc)) * epochs
        warmup_steps = max(10, total_steps // 20)

        args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch,
            gradient_accumulation_steps=grad_acc,
            learning_rate=3e-5,
            lr_scheduler_type="cosine",
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=False,
            predict_with_generate=True,
            generation_max_length=256,
            report_to="none",
            logging_steps=50,
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=args,
            train_dataset=REBELDataset(train_raw, self.tok),
            eval_dataset=REBELDataset(val_raw, self.tok),
            processing_class=self.tok,
            data_collator=DataCollatorForSeq2Seq(
                self.tok, model=self.model, pad_to_multiple_of=8
            ),
            callbacks=[
                TripletF1Callback(self.tok, val_raw, self.device),
                EarlyStoppingCallback(early_stopping_patience=early_stopping_patience),
            ],
        )

        print(f"[Trainer] Starting training — {total_steps} steps, "
              f"warmup {warmup_steps}, early_stop patience {early_stopping_patience}")
        trainer.train()

        self.model.save_pretrained(self.output_dir)
        self.tok.save_pretrained(self.output_dir)
        print(f"[Trainer] Model saved to {self.output_dir}")

    def inference_test(self, sentences: List[str]):
        self.model.eval()
        print("\n--- Final Inference Test ---")
        for sent in sentences:
            inp = self.tok(sent, return_tensors="pt").to(self.device)
            out = self.model.generate(inp["input_ids"], max_length=128)
            raw_out = self.tok.decode(out[0], skip_special_tokens=False)
            print(f"INPUT : {sent}")
            print(f"OUTPUT: {decode_triplets(raw_out)}\n")

# ══════════════════════════════════════════════════════════════════
# 6. EXECUTION
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples",  type=int, default=1000)
    parser.add_argument("--epochs",   type=int, default=6)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--domain-weight", type=int, default=20)
    args = parser.parse_args()

    dataset = load_local_rebel(max_samples=args.samples)

    max_safe_repeat = max(1, len(dataset) // max(len(DOMAIN_DATA), 1)) if dataset else args.domain_weight
    domain_repeat   = min(args.domain_weight, max_safe_repeat)
    domain_extras   = DOMAIN_DATA * domain_repeat
    dataset.extend(domain_extras)
    
    print(f"[Data] Total samples: {len(dataset)} "
          f"(includes {len(domain_extras)} domain rows, repeat×{domain_repeat})")

    if dataset:
        trainer = REBELTrainer()
        trainer.train(dataset, epochs=args.epochs,
                      early_stopping_patience=args.patience)

        trainer.inference_test([
            "Srishwan is a student at IIT Kharagpur.",
            "Aaron Jason Baptist studies AI at IITK.",
            "He stays in Lal Bahadur Shastri Hall.",
            "Apple was founded by Steve Jobs in California."
        ])
    else:
        print("No data found. Check your JSONL file path.")