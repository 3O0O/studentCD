#!/usr/bin/env python3
"""Download only safe inference assets from Qwen's official ModelScope repo."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import urllib.parse
import urllib.request


REPO = "Qwen/Qwen2.5-Coder-7B-Instruct"
API = "https://modelscope.cn/api/v1/models/" + REPO + "/repo/files?Revision=master&Recursive=true"
SMALL_FILES = {
    "config.json", "generation_config.json", "tokenizer.json",
    "tokenizer_config.json", "vocab.json", "merges.txt",
    "model.safetensors.index.json", "LICENSE", "README.md",
}
REQUIRED_FILES = {"config.json", "tokenizer.json", "model.safetensors.index.json"}
MAX_DOWNLOAD_BYTES = 17_000_000_000
SHARD_PATTERN = r"model-\d{5}-of-\d{5}\.safetensors"


def allowed_asset(name):
    return isinstance(name, str) and (name in SMALL_FILES or re.fullmatch(SHARD_PATTERN, name) is not None)


def validate_manifest(manifest):
    """Apply the same checks to newly fetched and previously saved metadata."""
    if (not isinstance(manifest, dict) or manifest.get("repository") != REPO
            or manifest.get("api") != API):
        raise ValueError("Unexpected model manifest identity")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Model manifest must contain assets")
    names, total = set(), 0
    for entry in files:
        if not isinstance(entry, dict) or not allowed_asset(entry.get("path")):
            raise ValueError("Unsafe model manifest path")
        name = entry["path"]
        if name in names:
            raise ValueError("Duplicate model manifest path: " + name)
        names.add(name)
        for field, pattern in (("sha256", r"[a-f0-9]{64}"), ("revision", r"[a-f0-9]{40}")):
            value = entry.get(field)
            if not isinstance(value, str) or not re.fullmatch(pattern, value):
                raise ValueError("Invalid immutable asset " + field + ": " + name)
        size = entry.get("size")
        if type(size) is not int or size < 0:
            raise ValueError("Invalid asset size: " + name)
        if size == 0 and (name in REQUIRED_FILES or name.endswith(".safetensors")):
            raise ValueError("Required model asset is empty: " + name)
        total += size
    if not REQUIRED_FILES <= names:
        raise ValueError("Model manifest is missing required assets")
    if not any(name.endswith(".safetensors") for name in names):
        raise ValueError("Model manifest contains no weights")
    if total > MAX_DOWNLOAD_BYTES:
        raise ValueError("Unexpected model download size")


def validate_weight_map(index, verified_assets):
    """Only nonempty mappings to checksum-verified safe weight files qualify."""
    mapping = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Model index must contain a nonempty weight_map")
    for tensor, name in mapping.items():
        if (not isinstance(tensor, str) or not tensor or not isinstance(name, str)
                or not re.fullmatch(SHARD_PATTERN, name) or name not in verified_assets):
            raise ValueError("Index references unverified weights")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def partial_snapshot(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("Partial asset must be a regular non-symlink file: " + str(path))
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def check_partial_unchanged(path, snapshot, stream=None):
    if partial_snapshot(path) != snapshot:
        raise ValueError("Partial asset changed during verification: " + str(path))
    if stream is not None:
        info = os.fstat(stream.fileno())
        current = (info.st_dev, info.st_ino, info.st_mode, info.st_size,
                   info.st_mtime_ns, info.st_ctime_ns)
        if current != snapshot:
            raise ValueError("Partial asset changed during verification: " + str(path))


def rehash_partial(path, expected_size):
    snapshot = partial_snapshot(path)
    if snapshot[3] > expected_size:
        raise ValueError("Partial asset larger than authoritative size: " + str(path))
    digest, count = hashlib.sha256(), 0
    # O_NONBLOCK avoids blocking if a raced replacement is a FIFO; O_NOFOLLOW
    # rejects a raced symlink. The descriptor and pathname must remain identical.
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        check_partial_unchanged(path, snapshot, stream)
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            count += len(chunk)
            if count > snapshot[3]:
                raise ValueError("Partial asset grew during verification: " + str(path))
            digest.update(chunk)
        check_partial_unchanged(path, snapshot, stream)
    if count != snapshot[3]:
        raise ValueError("Partial asset changed during verification: " + str(path))
    return digest, count, snapshot


def download(destination, resume_partial=False):
    destination = Path(destination).absolute()
    if destination.is_symlink() or destination.resolve() != destination:
        raise ValueError("Model destination must not contain symlinks")
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "download_manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("Model manifest is a symlink")
    existing_manifest = manifest_path.exists()
    if existing_manifest:
        manifest = json.loads(manifest_path.read_text())
    else:
        with urllib.request.urlopen(API, timeout=30) as response:
            index = json.load(response)
        if index.get("Code") != 200:
            raise ValueError("Official model index request failed")
        assets = []
        for item in index["Data"]["Files"]:
            name = item["Path"]
            if not allowed_asset(name):
                continue
            assets.append({"path": name, "sha256": item.get("Sha256"),
                           "size": item.get("Size"), "revision": item.get("Revision")})
        manifest = {"repository": REPO, "api": API, "files": assets}
    validate_manifest(manifest)
    if not existing_manifest:
        with manifest_path.open("x") as stream:
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
    verified_assets = set()
    for entry in manifest["files"]:
        name = entry["path"]
        target = destination / name
        if target.is_symlink():
            raise ValueError("Model asset is a symlink")
        if target.exists():
            if target.stat().st_size != entry["size"] or sha256(target) != entry["sha256"]:
                raise ValueError("Existing asset differs: " + name)
            verified_assets.add(name)
            print("Verified existing " + name, flush=True)
            continue
        partial = destination / (name + ".partial")
        has_partial = partial.exists() or partial.is_symlink()
        if has_partial and not resume_partial:
            raise ValueError("Partial asset exists; inspect before retry: " + str(partial))
        digest, count, snapshot = hashlib.sha256(), 0, None
        if has_partial:
            digest, count, snapshot = rehash_partial(partial, entry["size"])
            if count == entry["size"]:
                if digest.hexdigest() != entry["sha256"]:
                    raise ValueError("Asset checksum failed: " + name)
                check_partial_unchanged(partial, snapshot)
                if target.exists() or target.is_symlink():
                    raise ValueError("Target appeared during partial verification: " + name)
                partial.rename(target)
                verified_assets.add(name)
                print("Verified complete partial " + name, flush=True)
                continue
        url = ("https://modelscope.cn/models/" + REPO + "/resolve/"
               + entry["revision"] + "/" + urllib.parse.quote(name))
        request = url
        if has_partial:
            request = urllib.request.Request(url, headers={"Range": "bytes={}-".format(count),
                                                           "Accept-Encoding": "identity"})
        print("{} {} ({} of {} bytes present)".format(
            "Resuming" if has_partial else "Downloading", name, count, entry["size"]), flush=True)
        with urllib.request.urlopen(request, timeout=60) as response:
            if has_partial:
                expected_range = "bytes {}-{}/{}".format(count, entry["size"] - 1, entry["size"])
                if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                    raise ValueError("Resume requires exact 206 Content-Range: " + name)
                descriptor = os.open(partial, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW | os.O_NONBLOCK)
                stream = os.fdopen(descriptor, "ab")
            else:
                stream = partial.open("xb")
            with stream:
                if has_partial:
                    check_partial_unchanged(partial, snapshot, stream)
                while True:
                    chunk = response.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    count += len(chunk)
                    if count > entry["size"]:
                        raise ValueError("Asset larger than authoritative size")
                    digest.update(chunk)
                    stream.write(chunk)
        if (count != entry["size"] or partial_snapshot(partial)[3] != entry["size"]
                or digest.hexdigest() != entry["sha256"]):
            raise ValueError("Asset checksum failed: " + name)
        if target.exists() or target.is_symlink():
            raise ValueError("Target appeared during download: " + name)
        partial.rename(target)
        verified_assets.add(name)
        print("Verified " + name, flush=True)
    with (destination / "model.safetensors.index.json").open() as stream:
        validate_weight_map(json.load(stream), verified_assets)
    print("All assets verified; no model code has been executed.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--resume-partial", action="store_true",
                        help="Verify existing partial bytes and resume only an exact HTTP range; never retry")
    args = parser.parse_args()
    download(args.destination, resume_partial=args.resume_partial)
