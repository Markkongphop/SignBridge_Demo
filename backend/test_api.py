# -*- coding: utf-8 -*-
import time
import urllib.request
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

tests = [
    {"text": "\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35", "label": "exact match"},
    {"text": "\u0e2d\u0e32\u0e08\u0e32\u0e23\u0e22\u0e4c\u0e2b\u0e25\u0e48\u0e2d\u0e21\u0e32\u0e01", "label": "greedy tokenize"},
    {"text": "\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35\u0e04\u0e23\u0e31\u0e1a\u0e2d\u0e32\u0e08\u0e32\u0e23\u0e22\u0e4c", "label": "gemini path"},
]

for t in tests:
    data = json.dumps({"text": t["text"]}).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:8000/api/text-to-sign",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=15)
        result = json.loads(resp.read())
        elapsed = time.time() - t0
        print(f"[{t['label']}] text={t['text']} | time={elapsed:.3f}s | glosses={result.get('glosses','')} | videos={len(result.get('video_sequence',[]))}")
    except Exception as e:
        elapsed = time.time() - t0
        print(f"[{t['label']}] text={t['text']} | time={elapsed:.3f}s | ERROR: {e}")
