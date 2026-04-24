import pickle
import json

out = {}
try:
    with open('outputs/benchmark/forest_results.pkl', 'rb') as f:
        rf = pickle.load(f)
    out['RF'] = {"acc": rf['accuracy'], "time": rf['total_time']}
except: pass

try:
    with open('outputs/benchmark/flower_simulation_results.pkl', 'rb') as f:
        fl_c = pickle.load(f)
    out['FL_C_Flower'] = {"acc": fl_c['accuracy'], "time": fl_c['total_time']}
except: pass

try:
    with open('outputs/benchmark/fl_decentralized_results.pkl', 'rb') as f:
        fl_d = pickle.load(f)
    out['FL_D_Ring'] = {"acc": fl_d['rounds'][-1]['accuracy'], "time": fl_d['total_time'], "hops": sum([r['communication_hops'] for r in fl_d['rounds']])}
except: pass

with open('outputs/benchmark/final_summary.json', 'w') as f:
    json.dump(out, f, indent=2)
