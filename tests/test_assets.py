"""Synthetic asset checks only: no network, model loading or student execution."""

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import stat
import tarfile
import tempfile
import unittest
from unittest import mock


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / (name + ".py")
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


model = load_script("download_model")
progfeed_archive = load_script("extract_progfeed")
SHARD = "model-00001-of-00001.safetensors"


class ModelAssetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="model-assets-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.assets = {
            "config.json": b"{}",
            "tokenizer.json": b"{}",
            "model.safetensors.index.json": json.dumps({"weight_map": {"layer.weight": SHARD}}).encode(),
            SHARD: b"inert synthetic weight bytes; never loaded",
            "README.md": "# Official-style model description\n中文说明\n".encode(),
            "LICENSE": b"Synthetic test fixture license text",
        }
        blocked_network = mock.patch.object(model.urllib.request, "urlopen",
                                            side_effect=AssertionError("Network is forbidden in asset tests"))
        self.network = blocked_network.start()
        self.addCleanup(blocked_network.stop)
        self.output = io.StringIO()
        redirected = contextlib.redirect_stdout(self.output)
        redirected.__enter__()
        self.addCleanup(redirected.__exit__, None, None, None)

    def manifest(self):
        return {"repository": model.REPO, "api": model.API, "files": [
            {"path": name, "size": len(content), "sha256": hashlib.sha256(content).hexdigest(),
             "revision": "a" * 40} for name, content in self.assets.items()
        ]}

    def cached(self, name="cached", manifest=None):
        destination = self.root / name
        destination.mkdir()
        for path, content in self.assets.items():
            (destination / path).write_bytes(content)
        (destination / "download_manifest.json").write_text(json.dumps(manifest or self.manifest()))
        return destination

    def partial(self, name="partial", content=None):
        destination = self.cached(name)
        partial = destination / (SHARD + ".partial")
        (destination / SHARD).rename(partial)
        partial.write_bytes(self.assets[SHARD][:9] if content is None else content)
        return destination, partial

    def range_response(self, content, content_range, status=206):
        response = io.BytesIO(content)
        response.status = status
        response.headers = {} if content_range is None else {"Content-Range": content_range}
        return response

    def test_valid_manifest_preserves_readme_and_optional_empty_text(self):
        manifest = self.manifest()
        manifest["files"][-1]["size"] = 0  # Optional LICENSE text is not a weight/config requirement.
        manifest["files"][-1]["sha256"] = hashlib.sha256(b"").hexdigest()
        model.validate_manifest(manifest)
        self.assertTrue(model.allowed_asset("README.md"))

    def test_manifest_requires_identity_assets_and_weights(self):
        valid = self.manifest()
        invalid = [[], {}, dict(valid, repository="other/repo"), dict(valid, files=[]),
                   dict(valid, files={}), dict(valid, files=[None])]
        for missing in ("config.json", "tokenizer.json", "model.safetensors.index.json", SHARD):
            invalid.append(dict(valid, files=[entry for entry in valid["files"] if entry["path"] != missing]))
        duplicate = copy.deepcopy(valid)
        duplicate["files"].append(duplicate["files"][0])
        invalid.append(duplicate)
        for manifest in invalid:
            with self.subTest(manifest=manifest), self.assertRaises(ValueError):
                model.validate_manifest(manifest)

    def test_manifest_rejects_paths_checksums_revisions_and_invalid_sizes(self):
        changes = [("path", "../escaped.json"), ("path", "/tmp/config.json"),
                   ("path", "model.py"), ("path", None), ("sha256", "bad"),
                   ("sha256", None), ("revision", "master"), ("revision", "../main"),
                   ("size", -1), ("size", 0), ("size", True), ("size", 1.5),
                   ("size", "12"), ("size", model.MAX_DOWNLOAD_BYTES + 1)]
        for field, value in changes:
            manifest = self.manifest()
            manifest["files"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                model.validate_manifest(manifest)

    def test_cached_empty_manifest_cannot_report_success(self):
        destination = self.cached(manifest={"repository": model.REPO, "api": model.API, "files": []})
        (destination / "model.safetensors.index.json").write_text('{"weight_map": {}}')
        with self.assertRaisesRegex(ValueError, "must contain assets"):
            model.download(destination)
        self.network.assert_not_called()
        self.assertNotIn("All assets verified", self.output.getvalue())

    def test_cached_valid_assets_are_verified_without_network(self):
        destination = self.cached()
        model.download(destination)
        self.network.assert_not_called()
        self.assertIn("All assets verified", self.output.getvalue())
        self.assertEqual((destination / "README.md").read_bytes(), self.assets["README.md"])

    def test_cached_corruption_and_manifest_symlink_are_rejected(self):
        destination = self.cached()
        (destination / SHARD).write_bytes(b"changed bytes")
        with self.assertRaisesRegex(ValueError, "Existing asset differs"):
            model.download(destination)
        saved = destination / "saved_manifest.json"
        (destination / "download_manifest.json").rename(saved)
        (destination / "download_manifest.json").symlink_to(saved)
        with self.assertRaisesRegex(ValueError, "manifest is a symlink"):
            model.download(destination)
        self.network.assert_not_called()

    def test_weight_map_requires_verified_safetensors_and_nonempty_mapping(self):
        invalid = [None, {}, {"weight_map": {}}, {"weight_map": []},
                   {"weight_map": {"weight": "model-00002-of-00002.safetensors"}},
                   {"weight_map": {"weight": "config.json"}},
                   {"weight_map": {"weight": "../outside.safetensors"}},
                   {"weight_map": {"weight": None}}, {"weight_map": {"": SHARD}}]
        for index in invalid:
            with self.subTest(index=index), self.assertRaises(ValueError):
                model.validate_weight_map(index, set(self.assets))
        model.validate_weight_map({"weight_map": {"weight": SHARD}}, {SHARD})

    def test_cached_checksum_valid_but_empty_index_is_rejected(self):
        self.assets["model.safetensors.index.json"] = b'{"weight_map":{}}'
        destination = self.cached()
        with self.assertRaisesRegex(ValueError, "nonempty weight_map"):
            model.download(destination)
        self.network.assert_not_called()
        self.assertNotIn("All assets verified", self.output.getvalue())

    def test_bad_download_checksum_keeps_partial_without_publishing_asset(self):
        destination = self.cached()
        (destination / SHARD).unlink()
        wrong_bytes = b"x" * len(self.assets[SHARD])
        with mock.patch.object(model.urllib.request, "urlopen", return_value=io.BytesIO(wrong_bytes)):
            with self.assertRaisesRegex(ValueError, "Asset checksum failed"):
                model.download(destination)
        self.assertFalse((destination / SHARD).exists())
        self.assertTrue((destination / (SHARD + ".partial")).is_file())
        self.assertNotIn("All assets verified", self.output.getvalue())

    def test_partial_requires_explicit_resume_even_when_already_complete(self):
        for index, content in enumerate((self.assets[SHARD][:9], self.assets[SHARD])):
            destination, partial = self.partial("default-%d" % index, content)
            with self.assertRaisesRegex(ValueError, "Partial asset exists"):
                model.download(destination)
            self.assertEqual(partial.read_bytes(), content)
            self.assertFalse((destination / SHARD).exists())
        self.network.assert_not_called()

    def test_resume_hashes_prefix_and_requests_exact_remaining_immutable_range(self):
        content = self.assets[SHARD]
        for prefix_size in (0, 9):
            destination, partial = self.partial("resume-%d" % prefix_size, content[:prefix_size])
            response = self.range_response(content[prefix_size:], "bytes {}-{}/{}".format(
                prefix_size, len(content) - 1, len(content)))
            with mock.patch.object(model.urllib.request, "urlopen", return_value=response) as network:
                model.download(destination, resume_partial=True)
            network.assert_called_once()
            request = network.call_args.args[0]
            self.assertEqual(request.full_url, "https://modelscope.cn/models/" + model.REPO
                             + "/resolve/" + "a" * 40 + "/" + SHARD)
            self.assertEqual(request.get_header("Range"), "bytes=%d-" % prefix_size)
            self.assertEqual(request.get_header("Accept-encoding"), "identity")
            self.assertFalse(partial.exists())
            self.assertEqual((destination / SHARD).read_bytes(), content)

    def test_complete_partial_is_verified_and_published_without_network(self):
        destination, partial = self.partial(content=self.assets[SHARD])
        model.download(destination, resume_partial=True)
        self.network.assert_not_called()
        self.assertFalse(partial.exists())
        self.assertEqual((destination / SHARD).read_bytes(), self.assets[SHARD])

    def test_complete_corrupt_partial_is_preserved_without_network(self):
        content = b"x" * len(self.assets[SHARD])
        destination, partial = self.partial(content=content)
        with self.assertRaisesRegex(ValueError, "Asset checksum failed"):
            model.download(destination, resume_partial=True)
        self.network.assert_not_called()
        self.assertEqual(partial.read_bytes(), content)
        self.assertFalse((destination / SHARD).exists())

    def test_resume_rejects_symlinks_nonregular_and_oversized_partials(self):
        for kind in ("symlink", "directory", "oversized"):
            destination, partial = self.partial(kind)
            if kind == "symlink":
                saved = destination / "saved-partial"
                partial.rename(saved)
                partial.symlink_to(saved)
            elif kind == "directory":
                partial.rename(destination / "saved-partial")
                partial.mkdir()
            else:
                partial.write_bytes(self.assets[SHARD] + b"extra")
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "regular|larger"):
                model.download(destination, resume_partial=True)
            self.assertTrue(partial.exists())
            self.assertFalse((destination / SHARD).exists())
        self.network.assert_not_called()

    def test_resume_rejects_full_response_and_inconsistent_ranges_before_append(self):
        size, start = len(self.assets[SHARD]), 9
        cases = [(200, "bytes {}-{}/{}".format(start, size - 1, size)),
                 (206, None), (206, "bytes {}-{}/{}".format(start + 1, size - 1, size)),
                 (206, "bytes {}-{}/{}".format(start, size - 2, size)),
                 (206, "bytes {}-{}/{}".format(start, size - 1, size + 1)),
                 (206, "bytes {}-{}/*".format(start, size - 1)),
                 (206, "bytes {}-{}/{} extra".format(start, size - 1, size))]
        for index, (status, content_range) in enumerate(cases):
            destination, partial = self.partial("range-%d" % index)
            response = self.range_response(self.assets[SHARD][start:], content_range, status)
            with mock.patch.object(model.urllib.request, "urlopen", return_value=response) as network:
                with self.subTest(status=status, content_range=content_range), self.assertRaisesRegex(
                        ValueError, "exact 206 Content-Range"):
                    model.download(destination, resume_partial=True)
            network.assert_called_once()
            self.assertEqual(partial.read_bytes(), self.assets[SHARD][:start])
            self.assertFalse((destination / SHARD).exists())

    def test_resume_rejects_changed_partial_before_append(self):
        destination, partial = self.partial()
        prefix, size = self.assets[SHARD][:9], len(self.assets[SHARD])

        def changed_response(request, timeout):
            partial.rename(destination / "original-partial")
            partial.write_bytes(prefix)  # Equal contents and size, different inode.
            return self.range_response(self.assets[SHARD][9:], "bytes 9-{}/{}".format(size - 1, size))

        with mock.patch.object(model.urllib.request, "urlopen", side_effect=changed_response) as network:
            with self.assertRaisesRegex(ValueError, "changed during verification"):
                model.download(destination, resume_partial=True)
        network.assert_called_once()
        self.assertEqual(partial.read_bytes(), prefix)
        self.assertFalse((destination / SHARD).exists())

    def test_resume_checks_final_size_and_full_hash_without_retry_or_deletion(self):
        original, start = self.assets[SHARD], 9
        cases = [(original[:start], original[start:-1], "checksum"),
                 (original[:start], original[start:] + b"extra", "larger"),
                 (original[:start], b"x" * (len(original) - start), "checksum"),
                 (b"x" * start, original[start:], "checksum")]
        for index, (prefix, suffix, message) in enumerate(cases):
            destination, partial = self.partial("payload-%d" % index, prefix)
            response = self.range_response(suffix, "bytes 9-{}/{}".format(len(original) - 1, len(original)))
            with mock.patch.object(model.urllib.request, "urlopen", return_value=response) as network:
                with self.subTest(index=index), self.assertRaisesRegex(ValueError, message):
                    model.download(destination, resume_partial=True)
            network.assert_called_once()
            self.assertTrue(partial.is_file())
            self.assertFalse((destination / SHARD).exists())
            self.assertEqual(partial.read_bytes(), prefix if message == "larger" else prefix + suffix)

    def test_resume_read_failure_preserves_received_bytes_without_retry(self):
        destination, partial = self.partial()
        original = self.assets[SHARD]
        response = self.range_response(b"", "bytes 9-{}/{}".format(len(original) - 1, len(original)))
        response.read = mock.Mock(side_effect=[original[9:14], OSError("synthetic interrupted transfer")])
        with mock.patch.object(model.urllib.request, "urlopen", return_value=response) as network:
            with self.assertRaisesRegex(OSError, "interrupted transfer"):
                model.download(destination, resume_partial=True)
        network.assert_called_once()
        self.assertEqual(partial.read_bytes(), original[:14])
        self.assertFalse((destination / SHARD).exists())

    def test_fresh_download_uses_same_validation_and_checks_each_payload(self):
        manifest = self.manifest()
        index = {"Code": 200, "Data": {"Files": [
            {"Path": entry["path"], "Size": entry["size"], "Sha256": entry["sha256"],
             "Revision": entry["revision"]} for entry in manifest["files"]
        ]}}
        urls = []

        def response(url, timeout):
            urls.append(url)
            if url == model.API:
                return io.BytesIO(json.dumps(index).encode())
            prefix = "https://modelscope.cn/models/" + model.REPO + "/resolve/" + "a" * 40 + "/"
            self.assertTrue(url.startswith(prefix))
            return io.BytesIO(self.assets[url[len(prefix):]])

        destination = self.root / "fresh"
        with mock.patch.object(model.urllib.request, "urlopen", side_effect=response):
            model.download(destination)
        self.assertEqual(len(urls), len(self.assets) + 1)
        for name, content in self.assets.items():
            self.assertEqual((destination / name).read_bytes(), content)
        self.assertEqual(json.loads((destination / "download_manifest.json").read_text()), manifest)

        index["Data"]["Files"][0]["Revision"] = "master"
        with mock.patch.object(model.urllib.request, "urlopen", side_effect=response):
            with self.assertRaisesRegex(ValueError, "revision"):
                model.download(self.root / "invalid-fresh")
        self.assertFalse((self.root / "invalid-fresh" / "download_manifest.json").exists())


class ArchiveAssetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="archive-assets-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.prefix = "progFeed-dataset-public-testrevision"

    def archive(self, name, items):
        archive = self.root / (name + ".tar.gz")
        with tarfile.open(archive, "w:gz") as stream:
            root = tarfile.TarInfo(self.prefix)
            root.type = tarfile.DIRTYPE
            stream.addfile(root)
            for path, kind, content in items:
                member = tarfile.TarInfo(path)
                member.type, member.mode = kind, 0o755
                if kind == tarfile.REGTYPE:
                    member.size = len(content)
                    stream.addfile(member, io.BytesIO(content))
                else:
                    member.linkname = "../../outside"
                    stream.addfile(member)
        return archive

    def test_normal_archive_is_copied_as_inert_files_with_source_hash(self):
        content = b"raise RuntimeError('must never execute this synthetic fixture')\n"
        archive = self.archive("normal", [(self.prefix + "/all_labs/student.py", tarfile.REGTYPE, content)])
        destination = self.root / "extracted"
        manifest = progfeed_archive.extract(archive, destination)
        code = destination / "all_labs/student.py"
        self.assertEqual(code.read_bytes(), content)
        self.assertEqual(stat.S_IMODE(code.stat().st_mode), 0o644)
        self.assertEqual(manifest["archive_sha256"], hashlib.sha256(archive.read_bytes()).hexdigest())
        self.assertEqual(manifest["files"], 1)
        self.assertFalse(manifest["executed_upstream_code"])
        self.assertEqual(json.loads((destination / "SOURCE_ARCHIVE.json").read_text()), manifest)
        with self.assertRaisesRegex(ValueError, "Destination already exists"):
            progfeed_archive.extract(archive, destination)

    def test_archive_traversal_absolute_and_backslash_paths_are_rejected(self):
        paths = [self.prefix + "/../outside.py", "/tmp/outside.py", self.prefix + "/dir\\outside.py"]
        for index, path in enumerate(paths):
            archive = self.archive("bad-path-%d" % index, [(path, tarfile.REGTYPE, b"untrusted data")])
            destination = self.root / ("rejected-%d" % index)
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "Unsafe archive path"):
                progfeed_archive.extract(archive, destination)
            self.assertFalse(destination.exists())

    def test_archive_symlinks_hardlinks_and_special_files_are_rejected(self):
        for index, kind in enumerate((tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE)):
            archive = self.archive("bad-kind-%d" % index, [(self.prefix + "/link", kind, b"")])
            destination = self.root / ("rejected-kind-%d" % index)
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "links or special files"):
                progfeed_archive.extract(archive, destination)
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
