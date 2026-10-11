"""把一个 Power（或共用运行库）各平台的发布包信息（package.py / package_runtime.py 写的 .json）合进 Friday 读的索引：
    python ci/make_index.py --power llm --dist dist --tag llm-v0.1.0 --repo integemjack/Friday.Powers --previous previous --out out
    python ci/make_index.py --runtime cuda13 --dist dist --tag cuda13-v13.0.0 --repo integemjack/Friday.Powers --previous previous --out out

两份索引都放在 index Release 里：
  - index-v2.json：所有 Power（Friday 2.7 起读它），还有 runtimes（共用运行库，Friday 2.7.1 起认：平台包的 target 写着 "runtime" 时先装它）；
  - index.json：只有清单 schema 1 的 Power（voice），Friday 2.6 及以前读它，看不到它们用不了的 llm / diffusion。
--previous 是上次的两份索引所在的目录（没有 index-v2.json 时从 index.json 起步）；只替换这一项，别的保留。
发布任务里用 ci/publish_index.py（下载、合并、上传、核对）。
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


def builds_in(dist: Path, prefix: str, key: str):
    builds = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(dist).rglob(f"{prefix}-*.json"))]
    builds = [b for b in builds if key in b]
    if not builds:
        raise SystemExit("dist 里没有发布包信息")
    return builds


def build(previous: Path, out: Path, dist: Path, tag: str, repo: str, power: str = None, runtime: str = None):
    """合并、写到 out；返回 {kind, id, version}（publish_index.py 拿它核对）"""
    old = load(previous / "index.json") or {"schema": 1, "powers": []}
    new = load(previous / "index-v2.json") or {"schema": 2, "powers": list(old.get("powers", []))}
    new["schema"] = 2
    url = lambda name: f"https://github.com/{repo}/releases/download/{tag}/{name}"

    if runtime:
        spec = json.loads((ROOT / "runtimes" / f"{runtime}.json").read_text(encoding="utf-8"))
        builds = builds_in(dist, runtime, "platform")
        entry = dict(spec)
        entry["version"] = builds[0]["version"]
        entry["release"] = tag
        entry["packages"] = {b["platform"]: {"url": url(b["file"]), "sha256": b["sha256"], "size": b["size"]} for b in builds}
        new.setdefault("runtimes", {})[runtime] = entry
        expected = {"kind": "runtime", "id": runtime, "version": entry["version"]}
        note = f"index-v2.json：运行库 {runtime} {entry['version']}，{len(builds)} 个平台；index.json 不变"
    else:
        manifest = json.loads((ROOT / power / "power.json").read_text(encoding="utf-8"))
        targets = {}
        builds = builds_in(dist, power, "target")
        for b in builds:
            t = dict(manifest["targets"][b["target"]])
            t.update({"url": url(b["file"]), "sha256": b["sha256"], "size": b["size"]})
            targets[b["target"]] = t
        entry = {k: v for k, v in manifest.items() if k != "targets"}
        entry["version"] = builds[0]["version"]
        entry["release"] = tag
        entry["targets"] = targets
        replace(new, entry)
        if manifest.get("schema", 1) == 1:
            replace(old, entry)
        expected = {"kind": "power", "id": power, "version": entry["version"]}
        note = (f"index-v2.json：{power} {entry['version']}，{len(targets)} 个平台；"
                f"index.json {'也更新了' if manifest.get('schema', 1) == 1 else '不变'}")

    out.mkdir(parents=True, exist_ok=True)
    (out / "index-v2.json").write_text(json.dumps(new, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "index.json").write_text(json.dumps(old, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(note)
    return expected


def contains(index_path: Path, expected) -> bool:
    """索引里有这一项（版本也对）"""
    index = load(index_path)
    if not index:
        return False
    if expected["kind"] == "runtime":
        return index.get("runtimes", {}).get(expected["id"], {}).get("version") == expected["version"]
    return any(p.get("id") == expected["id"] and p.get("version") == expected["version"] for p in index.get("powers", []))


def main():
    parser = argparse.ArgumentParser()
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--power")
    which.add_argument("--runtime")
    parser.add_argument("--dist", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--previous", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    build(Path(args.previous), Path(args.out), Path(args.dist), args.tag, args.repo, power=args.power, runtime=args.runtime)


if __name__ == "__main__":
    main()
