"""把各平台的发布包信息（package.py 写的 .json）合成 Friday 读的 index.json：
    python voice/ci/make_index.py --dist dist --tag voice-v0.1.0 --repo integemjack/Friday.Powers --out dist/index.json

index.json 里每个 Power 一项：清单（标题、能力、启动参数、模型）+ 各平台的下载地址、sha256、大小。
已有 index.json（--previous）时只替换 voice 这一项，别的 Power 保留。
"""
import argparse
import json
from pathlib import Path

VOICE = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--previous")
    args = parser.parse_args()

    manifest = json.loads((VOICE / "power.json").read_text(encoding="utf-8"))
    builds = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(args.dist).rglob("voice-*.json"))]
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

    index = {"schema": 1, "powers": []}
    if args.previous and Path(args.previous).exists():
        index = json.loads(Path(args.previous).read_text(encoding="utf-8"))
    index["powers"] = [p for p in index.get("powers", []) if p.get("id") != "voice"] + [entry]
    Path(args.out).write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"index.json：voice {entry['version']}，{len(targets)} 个平台")


if __name__ == "__main__":
    main()
