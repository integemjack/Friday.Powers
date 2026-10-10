"""把一个 Power（llm、diffusion；voice 用 voice/ci/package.py）在一个平台的构建结果打成发布包：
    python ci/package.py --power llm --target windows-x64-vulkan --build llm/build --out dist [--extra 某.dll ...]

包里：可执行文件、power.json（填好版本和平台）、LICENSES/、附带的库（如 cuBLAS）。
Windows 出 .zip，其余出 .tar.gz（保留可执行权限）；旁边写一个同名 .json 记录 sha256 和大小，给 ci/make_index.py 用。
"""
import argparse
import hashlib
import json
import re
import shutil
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 各 Power 要随包带的许可证：包里的文件名 → 构建目录 _deps 下的候选路径
LICENSES = {
    "llm": {
        "llama.cpp.txt": ["llama-src/LICENSE"],
        "civetweb.txt": ["civetweb-src/LICENSE.md"],
    },
    "diffusion": {
        "stable-diffusion.cpp.txt": ["sdcpp-src/LICENSE"],
        "ggml.txt": ["sdcpp-src/ggml/LICENSE"],
        "civetweb.txt": ["civetweb-src/LICENSE.md"],
    },
}


def version(power: Path, executable: str):
    text = (power / "CMakeLists.txt").read_text(encoding="utf-8")
    return re.search(r"project\(" + re.escape(executable) + r" VERSION ([0-9.]+)", text).group(1)


def sha256(path: Path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--power", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--build", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--extra", nargs="*", default=[])
    args = parser.parse_args()

    power = ROOT / args.power
    manifest = json.loads((power / "power.json").read_text(encoding="utf-8"))
    executable = manifest["executable"]
    build = Path(args.build)
    out = Path(args.out)
    windows = args.target.startswith("windows")
    ver = version(power, executable)
    stage = out / "stage" / executable
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    exe = build / "bin" / (executable + ".exe" if windows else executable)
    shutil.copy2(exe, stage / exe.name)
    for extra in args.extra:
        shutil.copy2(extra, stage / Path(extra).name)

    target = manifest["targets"][args.target]
    manifest["version"] = ver
    manifest["target"] = args.target
    manifest["capabilities"] = target.get("capabilities", manifest["capabilities"])
    manifest["models"] = [m for m in manifest.get("models", []) if m["id"] in target.get("models", [])]
    del manifest["targets"]
    (stage / "power.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    licenses = stage / "LICENSES"
    licenses.mkdir()
    for name, candidates in LICENSES.get(args.power, {}).items():
        for candidate in candidates:
            path = build / "_deps" / candidate
            if path.exists():
                shutil.copyfile(path, licenses / name)
                break
    notices = power / "THIRD_PARTY_NOTICES.md"
    if notices.exists():
        shutil.copyfile(notices, licenses / "THIRD_PARTY_NOTICES.md")

    name = f"{args.power}-{ver}-{args.target}"
    if windows:
        archive = out / f"{name}.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for f in sorted(stage.rglob("*")):
                if f.is_file():
                    z.write(f, f.relative_to(stage.parent))
    else:
        archive = out / f"{name}.tar.gz"
        (stage / exe.name).chmod(0o755)
        with tarfile.open(archive, "w:gz") as t:
            t.add(stage, arcname=executable)
    info = {"power": args.power, "target": args.target, "version": ver, "file": archive.name, "sha256": sha256(archive),
            "size": archive.stat().st_size}
    (out / f"{name}.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    shutil.rmtree(out / "stage")
    print(json.dumps(info))


if __name__ == "__main__":
    main()
