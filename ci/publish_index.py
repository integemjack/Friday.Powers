"""发布任务的最后一步：把这一项合进 index Release 的两份索引（index-v2.json、index.json）并上传：
    python3 ci/publish_index.py --power llm --dist dist --tag llm-v0.1.1 --repo integemjack/Friday.Powers
    python3 ci/publish_index.py --runtime cuda13 --dist dist --tag cuda13-v13.0.0 --repo integemjack/Friday.Powers

几个 Power / 运行库的发布任务可能同时在改这两份索引（下载、合并、上传之间没有锁，后传的会盖掉先传的那一项）：
上传后隔一会儿再下载核对，这一项被别的任务盖掉了就从头再来（最多 5 次）。要 gh（GH_TOKEN）。
"""
import argparse
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_index  # noqa: E402


def gh(*args, check=True):
    return subprocess.run(["gh", *args], check=check)


def download(repo: str, dest: Path):
    """两份都要下到：别的任务正在换（--clobber 是先删再传）时会短暂没有，稍等再下；一直没有就停（不能拿空的起步、把别的项盖掉）"""
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("index.json", "index-v2.json"):
        for attempt in range(10):
            gh("release", "download", "index", "-R", repo, "-p", name, "-D", str(dest), "--clobber", check=False)
            if (dest / name).exists():
                break
            time.sleep(15)
        else:
            raise SystemExit(f"下不到 index Release 里的 {name}")


def main():
    parser = argparse.ArgumentParser()
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--power")
    which.add_argument("--runtime")
    parser.add_argument("--dist", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--repo", required=True)
    args = parser.parse_args()

    # index Release 和两份索引是第一次发 voice 时建的，之后一直在（没有就停，别从空的起步）
    if gh("release", "view", "index", "-R", args.repo, check=False).returncode != 0:
        raise SystemExit("没有 index Release")
    for attempt in range(1, 6):
        with tempfile.TemporaryDirectory() as tmp:
            previous, out, check = Path(tmp) / "previous", Path(tmp) / "out", Path(tmp) / "check"
            download(args.repo, previous)
            expected = make_index.build(previous, out, Path(args.dist), args.tag, args.repo, power=args.power, runtime=args.runtime)
            gh("release", "upload", "index", "-R", args.repo, str(out / "index.json"), str(out / "index-v2.json"), "--clobber")
            time.sleep(random.uniform(40, 80))
            download(args.repo, check)
            if make_index.contains(check / "index-v2.json", expected):
                print(f"索引里有 {expected['id']} {expected['version']} 了（第 {attempt} 次）")
                return
        print(f"第 {attempt} 次传上去的被别的发布任务盖掉了，重来")
    raise SystemExit("合不进索引：一直被别的发布任务盖掉")


if __name__ == "__main__":
    main()
