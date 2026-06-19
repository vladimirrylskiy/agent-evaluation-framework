import json
with open('raw_check/MAD_human_labelled_dataset.json') as f:
    human = json.load(f)

# HyperAgent trace_id=1
e = [x for x in human if x.get('mas_name')=='HyperAgent' and x.get('trace_id')==1][0]
print('Trace:', e['mas_name'], e['trace_id'], e['benchmark_name'])
print()
print('=== RAW annotations (целиком) ===')
print(json.dumps(e['annotations'], indent=2)[:3000])
