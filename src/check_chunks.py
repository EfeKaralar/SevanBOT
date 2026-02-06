import json
from src.chunk_documents import load_tokenizer, count_tokens
tokenizer = load_tokenizer()
over = max_t = 0
with open('chunks.jsonl') as f:
    for line in f:
        t = count_tokens(json.loads(line)['text_for_embedding'], tokenizer)
        if t > 512: over += 1
        if t > max_t: max_t = t
print(f'Over 512: {over}, Max: {max_t}')
