"""把一个平台的构建结果打成发布包：
    python voice/ci/package.py --target windows-x64-vulkan --build build --out dist [--extra 某.dll ...]

包里：friday-voice[.exe]、voices/（有合成时）、power.json（填好版本和平台）、LICENSES/、附带的库（如 cuBLAS）。
Windows 出 .zip，Linux 出 .tar.gz（保留可执行权限）；旁边写一个同名 .json 记录 sha256 和大小，给 make_index.py 用。
"""
import argparse
import hashlib
import json
import re
import shutil
import tarfile
import zipfile
from pathlib import Path

VOICE = Path(__file__).resolve().parent.parent


def version():
    text = (VOICE / "CMakeLists.txt").read_text(encoding="utf-8")
    return re.search(r"project\(friday-voice VERSION ([0-9.]+)", text).group(1)


def licenses(build: Path, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    deps = build / "_deps"
    found = {
        "ggml.txt": [deps / "ggml-src" / "LICENSE"],
        "cosyvoice.cpp.txt": [deps / "cosyvoice-src" / "LICENSE"],
        "civetweb.txt": [deps / "civetweb-src" / "LICENSE.md"],
        "pcre2.txt": [deps / "pcre2-src" / "LICENCE.md", deps / "pcre2-src" / "LICENCE"],
        "FunASR.txt": [VOICE / "third_party" / "funasr" / "LICENSE"],
    }
    for name, candidates in found.items():
        for c in candidates:
            if c.exists():
                shutil.copyfile(c, dest / name)
                break
    shutil.copyfile(VOICE / "THIRD_PARTY_NOTICES.md", dest / "THIRD_PARTY_NOTICES.md")


def sha256(path: Path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--build", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--extra", nargs="*", default=[])
    args = parser.parse_args()

    build = Path(args.build)
    out = Path(args.out)
    windows = args.target.startswith("windows")
    ver = version()
    stage = out / "stage" / "friday-voice"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    exe = build / "bin" / ("friday-voice.exe" if windows else "friday-voice")
    shutil.copy2(exe, stage / exe.name)
    if (build / "bin" / "voices").exists():
        shutil.copytree(build / "bin" / "voices", stage / "voices")
    for extra in args.extra:
        shutil.copy2(extra, stage / Path(extra).name)

    manifest = json.loads((VOICE / "power.json").read_text(encoding="utf-8"))
    target = manifest["targets"][args.target]
    manifest["version"] = ver
    manifest["target"] = args.target
    manifest["capabilities"] = target.get("capabilities", manifest["capabilities"])
    manifest["models"] = [m for m in manifest["models"] if m["id"] in target["models"]]
    del manifest["targets"]
    (stage / "power.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    licenses(build, stage / "LICENSES")

    name = f"voice-{ver}-{args.target}"
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
            t.add(stage, arcname="friday-voice")
    info = {"target": args.target, "version": ver, "file": archive.name, "sha256": sha256(archive), "size": archive.stat().st_size}
    (out / f"{name}.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    shutil.rmtree(out / "stage")
    print(json.dumps(info))


if __name__ == "__main__":
    main()
