"""把一个共用运行库（如 cuda13：Windows 上本地能力的 CUDA 版共用的 cuBLAS）打成发布包：
    python ci/package_runtime.py --runtime cuda13 --platform windows-x64 --version 13.0.0 --files a.dll b.dll --license EULA.txt --out dist

包里：这些文件（平铺在根目录）、runtime.json（填好版本和平台）、LICENSES/。旁边写一个同名 .json 记录 sha256 和大小，
给 ci/make_index.py --runtime 用。Friday 把它解到 Powers/runtimes/<运行库>/<版本>/，拉起用它的 Power 时加进 PATH。
"""
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sha256(path: Path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    spec = json.loads((ROOT / "runtimes" / f"{args.runtime}.json").read_text(encoding="utf-8"))
    out = Path(args.out)
    stage = out / f"{args.runtime}-stage"
    shutil.rmtree(stage, ignore_errors=True)
    (stage / "LICENSES").mkdir(parents=True)
    names = []
    for file in args.files:
        source = Path(file)
        shutil.copy2(source, stage / source.name)
        names.append(source.name)
    shutil.copy2(args.license, stage / "LICENSES" / "NVIDIA-CUDA-EULA.txt")
    manifest = dict(spec, version=args.version, platform=args.platform, files=sorted(names))
    (stage / "runtime.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    name = f"{args.runtime}-{args.version}-{args.platform}.zip"
    archive = out / name
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(stage).as_posix())
    shutil.rmtree(stage)
    info = {"runtime": args.runtime, "platform": args.platform, "version": args.version, "file": name,
            "sha256": sha256(archive), "size": archive.stat().st_size}
    (out / f"{args.runtime}-{args.version}-{args.platform}.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(info))


if __name__ == "__main__":
    main()
