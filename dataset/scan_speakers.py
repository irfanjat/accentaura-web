#!/usr/bin/env python3
"""
Fast Parallel Scanner to identify speaker metadata from Speech Accent Archive (GMU)
Targeting 7 accent categories:
1. British English
2. American English
3. Australian English
4. Pakistani English
5. Indian English
6. Irish English
7. Scottish English
"""
import urllib.request
import re
import json
from concurrent.futures import ThreadPoolExecutor

def fetch_speaker_meta(speaker_id):
    url = f"https://accent.gmu.edu/samples/{speaker_id}/"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            html = r.read().decode('utf-8', errors='ignore')
            bp_match = re.search(r'<td>birth place</td>\s*<td>(.*?)</td>', html, re.I)
            lang_match = re.search(r'<td>native language</td>\s*<td>(.*?)</td>', html, re.I)
            bp = bp_match.group(1).strip().lower() if bp_match else ""
            lang = lang_match.group(1).strip().lower() if lang_match else ""
            return {"id": speaker_id, "birthplace": bp, "native_lang": lang}
    except Exception:
        return None

def main():
    print("Scanning Speech Accent Archive in parallel...")
    speakers_to_check = [f"english{i}" for i in range(1, 150)] + \
                        [f"urdu{i}" for i in range(1, 15)] + \
                        [f"punjabi{i}" for i in range(1, 10)] + \
                        [f"hindi{i}" for i in range(1, 15)]

    results = {
        "American": [],
        "British": [],
        "Australian": [],
        "Irish": [],
        "Scottish": [],
        "Pakistani": [],
        "Indian": []
    }

    with ThreadPoolExecutor(max_workers=12) as executor:
        meta_list = list(executor.map(fetch_speaker_meta, speakers_to_check))

    for meta in meta_list:
        if not meta:
            continue
        sid = meta["id"]
        bp = meta["birthplace"]
        lang = meta["native_lang"]

        if "urdu" in sid or "punjabi" in sid:
            if ("pakistan" in bp or bp == "" or "lahore" in bp or "karachi" in bp or "islamabad" in bp) and len(results["Pakistani"]) < 8:
                results["Pakistani"].append(sid)
        elif "hindi" in sid:
            if ("india" in bp or bp == "" or "delhi" in bp or "mumbai" in bp) and len(results["Indian"]) < 8:
                results["Indian"].append(sid)
        elif "english" in sid:
            if "scotland" in bp or "glasgow" in bp or "edinburgh" in bp:
                if len(results["Scottish"]) < 8:
                    results["Scottish"].append(sid)
            elif "ireland" in bp or "dublin" in bp or "belfast" in bp:
                if len(results["Irish"]) < 8:
                    results["Irish"].append(sid)
            elif "australia" in bp or "brisbane" in bp or "sydney" in bp or "perth" in bp or "melbourne" in bp:
                if len(results["Australian"]) < 8:
                    results["Australian"].append(sid)
            elif ("uk" in bp or "england" in bp or "london" in bp or "birmingham" in bp) and "scotland" not in bp and "ireland" not in bp:
                if len(results["British"]) < 8:
                    results["British"].append(sid)
            elif "usa" in bp or "america" in bp:
                if len(results["American"]) < 8:
                    results["American"].append(sid)

    print("\n--- IDENTIFIED SPEAKERS PER ACCENT ---")
    for cat, spks in results.items():
        print(f"{cat} ({len(spks)}): {spks}")

    out_path = "/home/irfanali/Desktop/english-accent-identifier/dataset/speakers_catalog.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved catalog to {out_path}")

if __name__ == "__main__":
    main()
