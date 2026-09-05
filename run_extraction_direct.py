"""Direct API text extraction for paper results (no langchain dependency)."""
import os, sys, json, time, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
load_dotenv('api_key.env', override=True)

import dashscope
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')

def extract_chunk(chunk_text, idx):
    """Call Qwen-Max for structured extraction on a single text chunk."""
    prompt = (
        "你是一个专业的文本信息抽取专家。请逐字逐句阅读文本，准确提取【岩石地层】和【剖面】实体。\n"
        "如果没有提及某个字段，必须返回 null。岩石特征描述限30字内。\n"
        "请严格输出纯JSON，键名必须为英文:\n"
        '{"strata":[{"formation":"地层名称","formation_code":"地层代号",'
        '"formation_age_1":"主年代","formation_age_code_1":"主年代代号",'
        '"rock_features":"岩石特征概括","confidence":0.9}],'
        '"profiles":[{"name":"剖面名称","formation":"对应地层","thickness":"厚度",'
        '"overlying_stratum":"上覆层","underlying_stratum":"下伏层","confidence":0.9}]}\n\n'
        f"文本片段:\n{chunk_text}"
    )

    for attempt in range(2):
        try:
            resp = dashscope.Generation.call(
                model='qwen-max',
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.01,
                max_tokens=3000,
                result_format='message'
            )
            if resp.status_code == 200:
                content = resp.output.choices[0].message.content
                content = re.sub(r'```json|```', '', content, flags=re.IGNORECASE).strip()
                match = re.search(r'\{[\s\S]*\}', content)
                if match:
                    return json.loads(match.group())
            else:
                print(f"  Chunk {idx} API error: {resp.code} {resp.message}")
        except Exception as e:
            print(f"  Chunk {idx} attempt {attempt+1} failed: {str(e)[:80]}")
            time.sleep(2)
    return {"strata": [], "profiles": []}


def main():
    with open('temp/古元古界+青白口系-30k.txt', encoding='utf-8') as f:
        text = f.read()

    # Simple chunking: ~6000 chars per chunk
    chunk_size = 6000
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    print(f'Text: {len(text)} chars, {len(chunks)} chunks')

    all_strata, all_profiles = [], []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(extract_chunk, c, i): i for i, c in enumerate(chunks)}
        for f in as_completed(futures):
            i = futures[f]
            try:
                data = f.result()
                if data.get('strata'):
                    all_strata.extend(data['strata'])
                if data.get('profiles'):
                    all_profiles.extend(data['profiles'])
                print(f'  Chunk {i+1}/{len(chunks)}: {len(data.get("strata",[]))} strata, {len(data.get("profiles",[]))} profiles')
            except Exception as e:
                print(f'  Chunk {i+1} error: {e}')

    elapsed = time.time() - t0
    print(f'\nDone in {elapsed:.1f}s: {len(all_strata)} strata, {len(all_profiles)} profiles')

    # Save
    result = {"strata": all_strata, "profiles": all_profiles,
              "stats": {"text_chars": len(text), "chunks": len(chunks),
                        "elapsed_s": round(elapsed,1), "model": "qwen-max"}}
    with open('temp/text_extraction_results.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Print sample
    for i, s in enumerate(all_strata[:8]):
        print(f'\n--- Stratum {i+1} ---')
        for k, v in s.items():
            if v: print(f'  {k}: {str(v)[:120]}')

    for i, p in enumerate(all_profiles[:3]):
        print(f'\n--- Profile {i+1} ---')
        for k, v in p.items():
            if v: print(f'  {k}: {str(v)[:120]}')


if __name__ == '__main__':
    main()
