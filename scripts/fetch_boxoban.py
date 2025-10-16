"""
Script to download Boxoban levels from the public repository.
Downloads zip, unpacks to sokoban_core/levels/boxoban/.
"""
from __future__ import annotations
import io, os, zipfile, argparse, sys
try:
    import requests  # type: ignore
except Exception:
    requests = None

BOXOBAN_ZIP = "https://github.com/deepmind/boxoban-levels/archive/refs/heads/master.zip"
TARGET_DIR   = "sokoban_core/levels/boxoban"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", type=str, default=BOXOBAN_ZIP)
    p.add_argument("--out", type=str, default=TARGET_DIR)
    args = p.parse_args()

    if requests is None:
        print("requests is not installed. Install: pip install requests")
        sys.exit(2)

    os.makedirs(args.out, exist_ok=True)
    print("Downloading:", args.url)
    r = requests.get(args.url)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))

    # Find subfolders medium/easy/test
    root = None
    for name in z.namelist():
        if name.endswith('/') and name.count('/') == 1 and 'boxoban-levels' in name:
            root = name
            break
    if root is None:
        print("Root not found in the zip archive")
        sys.exit(1)

    for name in z.namelist():
        if not name.startswith(root):
            continue
        rel = name[len(root):]
        if not rel:
            continue
        if rel.startswith(('medium/', 'easy/', 'test/')):
            dest = os.path.join(args.out, rel)
            if name.endswith('/'):
                os.makedirs(dest, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with z.open(name) as src, open(dest, 'wb') as dst:
                    dst.write(src.read())
    print("Unpacked to:", args.out)

if __name__ == "__main__":
    main()