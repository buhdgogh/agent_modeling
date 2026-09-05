"""Run text extraction for paper results."""
import os, sys, json, time
sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv
load_dotenv('api_key.env', override=True)
from text_info_agent import TextInfoAgent

with open('temp/古元古界+青白口系-30k.txt', encoding='utf-8') as f:
    text = f.read()

print(f'Text length: {len(text)} chars')
agent = TextInfoAgent(api_key=os.getenv('DASHSCOPE_API_KEY'), model_name='qwen-max')
t0 = time.time()
result = agent.run(text)
elapsed = time.time() - t0

extraction = result.get('extraction')
data = extraction.model_dump() if hasattr(extraction, 'model_dump') else (
    extraction.dict() if hasattr(extraction, 'dict') else extraction)

strata = data.get('strata', []) if isinstance(data, dict) else []
profiles = data.get('profiles', []) if isinstance(data, dict) else []

print(f'Time: {elapsed:.1f}s')
print(f'Strata: {len(strata)}, Profiles: {len(profiles)}')

with open('temp/text_extraction_results.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

for i, s in enumerate(strata):
    print(f'\n--- Stratum {i+1} ---')
    for k, v in s.items():
        if v:
            print(f'  {k}: {str(v)[:150]}')

for i, p in enumerate(profiles[:5]):
    print(f'\n--- Profile {i+1} ---')
    for k, v in p.items():
        if v:
            print(f'  {k}: {str(v)[:150]}')
