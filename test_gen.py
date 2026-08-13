import sys
sys.path.insert(0, r'C:\Users\Sajan\Desktop\Project XYZ')
sys.path.insert(0, r'C:\Users\Sajan\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none')

from model import ModelConfig, SmallEnglishLLM
from tokenizers import Tokenizer
import torch

checkpoint = torch.load(r'checkpoints\best\checkpoint_semv2c-3000.pt', map_location='cpu')
config = ModelConfig()
model = SmallEnglishLLM(config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
tokenizer = Tokenizer.from_file(r'tokenizer.json')

def generate(text, max_new=30, temp=0.3, topk=40):
    encoded = tokenizer.encode(text)
    input_ids = encoded.ids
    input_ids = input_ids[-config.max_seq_len:]
    x = torch.tensor([input_ids], dtype=torch.long)
    
    for _ in range(max_new):
        x_cond = x[:, -config.max_seq_len:]
        logits, _ = model(x_cond)
        next_logits = logits[:, -1, :] / temp
        values, indices = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
        filtered = torch.full_like(next_logits, float('-inf'))
        filtered.scatter_(1, indices, values)
        probs = torch.softmax(filtered, dim=-1)
        next_token = torch.multinomial(probs, 1)
        x = torch.cat([x, next_token], dim=1)
        if next_token.item() == 3:  # eos
            break
    
    generated = tokenizer.decode(x[0, len(input_ids):].tolist())
    return generated.strip()

# Test chat format
tests = [
    'User: What direction does sunrise occur?\nAssistant:',
    'User: Complete this sentence: You ___ kind.\nAssistant:',
    'User: The dog is not inside the house. Is the dog inside?\nAssistant:',
    'User: Alice gave Bob a pen. Who received the pen?\nAssistant:',
    'User: What is 7 + 2?\nAssistant:',
]

for t in tests:
    out = generate(t, max_new=30, temp=0.3, topk=40)
    print(f'Q: {t}')
    print(f'A: {out}')
    print()