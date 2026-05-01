"""
Download selected Aria Gen2 Pilot data products from the manifest JSON.

Usage:
  python aria_download.py --sequence eat_0 --products video_main_rgb diarization
  python aria_download.py --sequence eat_0 --minimum     # == video + diarization
  python aria_download.py --sequence eat_0 --gt_3d       # += depth + slam_points

URLs in the manifest are signed CDN links that expire; if you get a 403,
re-download a fresh manifest from the Aria portal.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from urllib.request import urlretrieve


MANIFEST = Path("/home/myid/zs89458/Documents/executorch/research_dev/AriaGen2PilotDataset_download_urls.json")
OUT_ROOT = Path("/home/myid/zs89458/Documents/executorch/research_dev/aria_data")

MINIMUM = ("video_main_rgb", "diarization")
GT_3D = ("depth", "mps_slam_points", "mps_slam_trajectories", "mps_slam_calibration")


def human(n: int) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def reporthook(block_num: int, block_size: int, total: int) -> None:
    got = block_num * block_size
    pct = min(100, 100 * got / max(1, total))
    sys.stdout.write(f"\r    {pct:5.1f}%  {human(got)}/{human(total)}")
    sys.stdout.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", default="eat_0", help="Sequence name, or 'all'.")
    ap.add_argument("--products", nargs="+", default=None, help="Data product names, or 'all'.")
    ap.add_argument("--minimum", action="store_true",
                    help=f"Shortcut for --products {' '.join(MINIMUM)}")
    ap.add_argument("--gt_3d", action="store_true",
                    help=f"Add {' + '.join(GT_3D)}")
    ap.add_argument("--all", action="store_true",
                    help="Download all sequences and all products (~159 GB).")
    ap.add_argument("--list", action="store_true", help="List available sequences/products and exit.")
    ap.add_argument("--dry_run", action="store_true", help="Print what would be downloaded.")
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    if args.list:
        for name, seq in manifest["sequences"].items():
            total = sum(p.get("file_size_bytes", 0) for p in seq.values())
            print(f"{name:12s}  total {human(total)}  products: {', '.join(seq)}")
        return

    # Resolve sequence list
    if args.all or args.sequence == "all":
        seq_names = list(manifest["sequences"])
    else:
        if args.sequence not in manifest["sequences"]:
            sys.exit(f"Unknown sequence '{args.sequence}'. Use --list to see names.")
        seq_names = [args.sequence]

    # Resolve product list (shared across all sequences)
    products: list[str] = list(args.products or [])
    if args.minimum: products.extend(MINIMUM)
    if args.gt_3d: products.extend(GT_3D)
    if args.all: products = []  # empty => "all products in each sequence"
    products = list(dict.fromkeys(products))
    if not products and not args.all:
        products = list(MINIMUM)

    # Compute grand total first for the plan summary
    def per_seq_products(name: str) -> list[str]:
        return products if products else list(manifest["sequences"][name].keys())

    grand_total = 0
    for name in seq_names:
        seq = manifest["sequences"][name]
        for p in per_seq_products(name):
            grand_total += seq.get(p, {}).get("file_size_bytes", 0)
    print(f"plan: {len(seq_names)} sequence(s), total {human(grand_total)}")
    if args.dry_run:
        return

    done_bytes = 0
    for name in seq_names:
        seq = manifest["sequences"][name]
        seq_dir = OUT_ROOT / name
        seq_dir.mkdir(parents=True, exist_ok=True)
        for p in per_seq_products(name):
            if p not in seq:
                print(f"  [{name}] SKIP: {p}"); continue
            meta = seq[p]
            fname = meta["filename"]
            url = meta["download_url"]
            size = meta.get("file_size_bytes", 0)
            out = seq_dir / fname
            if out.exists() and out.stat().st_size == size:
                print(f"  [{name}] OK (cached): {fname}")
                done_bytes += size
                continue
            print(f"  [{name}] {p} -> {fname} ({human(size)})")
            try:
                urlretrieve(url, out, reporthook)
                print()
                done_bytes += size
            except Exception as e:
                print(f"\n  FAILED: {e}")
                print(f"  (Signed CDN URLs expire after a few hours. Re-fetch the manifest JSON from Aria portal to continue.)")
                raise
            # Running total
            print(f"  progress: {human(done_bytes)}/{human(grand_total)} "
                  f"({100*done_bytes/max(1,grand_total):.1f}%)")

    print(f"\ndone -> {OUT_ROOT}")


if __name__ == "__main__":
    main()
