import pickle
import json
import os
import sys

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

results = {}
for fn in os.listdir('./outputs/benchmark/'):
    if fn.endswith('.pkl'):
        try:
            with open(f'./outputs/benchmark/{fn}', 'rb') as f:
                data = pickle.load(f)
                filtered = {k: v for k, v in data.items() if k != 'epoch_metrics'}
                results[fn] = filtered
        except Exception as e:
            pass

with open('debug_out.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("done")
