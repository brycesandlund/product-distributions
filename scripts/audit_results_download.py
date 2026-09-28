"""Verify downloaded Modal result file sizes and save a local inventory."""
import argparse
import json
from pathlib import Path

import modal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("destination")
    args = parser.parse_args()
    root = Path(args.destination)
    volume = modal.Volume.from_name("product-distributions-results")
    inventory, errors = [], []
    for entry in volume.iterdir("/", recursive=True):
        if entry.type.name != "FILE":
            continue
        path = root / entry.path.lstrip("/")
        size = path.stat().st_size if path.is_file() else None
        inventory.append({"path": entry.path, "remote_bytes": entry.size,
                          "local_bytes": size, "remote_mtime": entry.mtime})
        if size != entry.size:
            errors.append(entry.path)
    result = {"volume": "product-distributions-results", "file_count": len(inventory),
              "total_bytes": sum(r["remote_bytes"] for r in inventory),
              "verification": "file presence and byte size (not content hashes)",
              "errors": errors, "files": inventory}
    (root / "download_inventory.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
