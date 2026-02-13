import os
import json
import argparse
from tqdm import tqdm

def main():
    p = argparse.ArgumentParser(description="Remap label level_ids from filtered to solved paths using meta info.")
    p.add_argument("--meta_dir", default="sokoban_core/levels/letslogic_solved/meta", help="Directory with mapping JSONs")
    p.add_argument("--labels_in", default="data/packs_offpolicy_labels_festival", help="Original labels directory")
    p.add_argument("--labels_out", default="data/packs_offpolicy_labels_festival_solved", help="Output directory for remapped labels")
    args = p.parse_args()

    os.makedirs(args.labels_out, exist_ok=True)

    meta_files = sorted([f for f in os.listdir(args.meta_dir) if f.endswith(".json")])
    if not meta_files:
        print(f"No meta files found in {args.meta_dir}")
        return

    for meta_fn in tqdm(meta_files, desc="Remapping packs"):
        with open(os.path.join(args.meta_dir, meta_fn), "r", encoding="utf-8") as f:
            meta = json.load(f)
        
        pack = meta["pack"]
        old_txt = meta["source_pack_txt"]
        new_txt = os.path.join(os.path.dirname(args.meta_dir), f"{pack}.txt")

        mapping = {}
        for new_idx, old_idx in enumerate(meta["kept_original_indices"]):
            old_lid = f"{old_txt}#{old_idx}"
            new_lid = f"{new_txt}#{new_idx}"
            mapping[old_lid] = new_lid
        
        in_jsonl = os.path.join(args.labels_in, f"{pack}.jsonl")
        if not os.path.exists(in_jsonl):
            continue
            
        out_jsonl = os.path.join(args.labels_out, f"{pack}.jsonl")
        
        written = 0
        skipped = 0
        with open(in_jsonl, "r", encoding="utf-8") as fin, \
             open(out_jsonl, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line: continue
                rec = json.loads(line)
                
                lid = rec.get("level_id")
                if lid in mapping:
                    rec["level_id"] = mapping[lid]
                    if "_source_level_id" in rec:
                        rec["_source_level_id"] = mapping[lid]
                    fout.write(json.dumps(rec) + "\n")
                    written += 1
                else:
                    skipped += 1

if __name__ == "__main__":
    main()
