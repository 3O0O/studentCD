#!/usr/bin/env python3
"""Validate and unpack an official ProgFeed archive as inert research data."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile


def extract(archive, destination):
    archive, destination = Path(archive), Path(destination)
    if destination.exists() or destination.is_symlink():
        raise ValueError("Destination already exists; do not overwrite raw data")
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        total = sum(member.size for member in members)
        if len(members) > 150000 or total > 2_000_000_000:
            raise ValueError("Archive exceeds the explicit research-data limits")
        roots = set()
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or "\\" in member.name:
                raise ValueError("Unsafe archive path")
            if not (member.isfile() or member.isdir()):
                raise ValueError("Archive contains links or special files")
            roots.add(path.parts[0])
        if len(roots) != 1 or not next(iter(roots)).startswith("progFeed-dataset-public-"):
            raise ValueError("Unexpected upstream archive root")
        destination.mkdir(parents=True)
        for member in members:
            relative = PurePosixPath(member.name).parts[1:]
            if not relative:
                continue
            output = destination.joinpath(*relative)
            if member.isdir():
                output.mkdir(parents=True, exist_ok=True)
            else:
                output.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(member) as source, output.open("xb") as target:
                    shutil.copyfileobj(source, target)
                output.chmod(0o644)  # Downloaded programs are data, never executable.
    manifest = {
        "source": "https://github.com/umass-ml4ed/progFeed-dataset-public",
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "archive_root": next(iter(roots)),
        "files": sum(member.isfile() for member in members),
        "uncompressed_bytes": total,
        "executed_upstream_code": False,
    }
    with (destination / "SOURCE_ARCHIVE.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    print(json.dumps(extract(args.archive, args.destination), indent=2))
