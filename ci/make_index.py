"""把一个 Power 各平台的发布包信息（package.py 写的 .json）合进 Friday 读的索引：
    python ci/make_index.py --power llm --dist dist --tag llm-v0.1.0 --repo integemjack/Friday.Powers --previous previous --out out

两份索引都放在 index Release 里：
  - index-v2.json：所有 Power（Friday 2.7 起读它）；
  - index.json：只有清单 schema 1 的 Power（voice），Friday 2.6 及以前读它，看不到它们用不了的 llm / diffusion。
--previous 是上次的两份索引所在的目录（没有 index-v2.json 时从 index.json 起步）；只替换这个 Power 的那一项，别的保留。
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(path: Path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def replace(index, entry):
    index["powers"] = [p for p in index.get("powers", []) if p.get("id") != entry["id"]] + [entry]
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--power", required=True)
    parser.add_argument("--dist", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--previous", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    manifest = json.loads((ROOT / args.power / "power.json").read_text(encoding="utf-8"))
    builds = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(args.dist).rglob(f"{args.power}-*.json"))]
    builds = [b for b in builds if "target" in b]
    if not builds:
        raise SystemExit("dist 里没有发布包信息")
    targets = {}
    for b in builds:
        t = dict(manifest["targets"][b["target"]])
        t.update({
            "url": f"https://github.com/{args.repo}/releases/download/{args.tag}/{b['file']}",
            "sha256": b["sha256"],
            "size": b["size"],
        })
        targets[b["target"]] = t
    entry = {k: v for k, v in manifest.items() if k != "targets"}
    entry["version"] = builds[0]["version"]
    entry["release"] = args.tag
    entry["targets"] = targets

    previous = Path(args.previous)
    old = load(previous / "index.json") or {"schema": 1, "powers": []}
    new = load(previous / "index-v2.json") or {"schema": 2, "powers": list(old.get("powers", []))}
    new["schema"] = 2
    replace(new, entry)
    if manifest.get("schema", 1) == 1:
        replace(old, entry)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "index-v2.json").write_text(json.dumps(new, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "index.json").write_text(json.dumps(old, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"index-v2.json：{args.power} {entry['version']}，{len(targets)} 个平台；index.json {'也更新了' if manifest.get('schema', 1) == 1 else '不变'}")


if __name__ == "__main__":
    main()
