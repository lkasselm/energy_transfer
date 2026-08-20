"""Fetches, checksums, extracts, and converts the DD0024 gold-standard Enzo snapshot
(github.com/pgrete/energy-transfer-analysis's own testing/README.md fixture) into an
ADIOS2/bp5 file energy_transfer can read. Everything is cached under cache_dir so this
only actually happens once; subsequent calls just return the cached .bp path.
"""

import hashlib
import os
import subprocess
import sys
import tarfile
import urllib.request

DOWNLOAD_URL = "https://pgrete.de/dl/EnTrans/EnTransTesting-DD0024.tgz"
MD5 = "9b2dfb5412770ae63277726de39b8f63"
SHA1 = "d3df730b08943a2067f377140943f70ff8b9fd20"

TARBALL_NAME = "EnTransTesting-DD0024.tgz"
ENZO_DATA_SUBPATH = os.path.join("DD0024", "data0024")
CONVERTED_BP_NAME = "enzo_data_dd0024.bp"

_ENZO_TO_BP5 = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "scripts", "enzo_to_bp5.py"
)


class DatasetUnavailable(Exception):
    """Raised when the gold-standard dataset couldn't be fetched/converted -- the
    caller should treat this as "skip the test", not "fail the test"."""


def _checksum(path, algo):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_checksums(path):
    for algo, expected in (("md5", MD5), ("sha1", SHA1)):
        actual = _checksum(path, algo)
        if actual != expected:
            raise DatasetUnavailable(
                f"{algo} mismatch for {path}: expected {expected}, got {actual}"
            )


def _download(cache_dir):
    tarball_path = os.path.join(cache_dir, TARBALL_NAME)
    part_path = tarball_path + ".part"
    try:
        print(f"gold_standard_dd0024: downloading {DOWNLOAD_URL} ...")
        urllib.request.urlretrieve(DOWNLOAD_URL, part_path)
    except OSError as e:
        raise DatasetUnavailable(f"could not download {DOWNLOAD_URL}: {e}") from e
    os.replace(part_path, tarball_path)
    _verify_checksums(tarball_path)
    return tarball_path


def _extract(cache_dir, tarball_path):
    print(f"gold_standard_dd0024: extracting {tarball_path} ...")
    try:
        with tarfile.open(tarball_path) as tf:
            tf.extractall(cache_dir)
    except tarfile.TarError as e:
        raise DatasetUnavailable(f"could not extract {tarball_path}: {e}") from e


def _convert(cache_dir):
    data_path = os.path.join(cache_dir, ENZO_DATA_SUBPATH)
    output_path = os.path.join(cache_dir, CONVERTED_BP_NAME)
    if not os.path.exists(data_path):
        raise DatasetUnavailable(f"expected Enzo data at {data_path}, not found after extraction")

    cmd = [
        sys.executable,
        os.path.abspath(_ENZO_TO_BP5),
        data_path,
        "--output",
        output_path,
        "--res",
        "128",
        "--quantity-type",
        "primitive",
    ]
    print("gold_standard_dd0024: converting via " + " ".join(cmd))
    try:
        proc = subprocess.run(cmd, cwd=cache_dir, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True)
    except OSError as e:
        raise DatasetUnavailable(f"could not run enzo_to_bp5.py: {e}") from e
    print(proc.stdout)
    if proc.returncode != 0:
        raise DatasetUnavailable(
            f"enzo_to_bp5.py exited with code {proc.returncode} (see output above -- "
            "likely a missing yt/adios2 module or a field-name mismatch against this "
            "specific Enzo dataset)"
        )
    if not os.path.exists(output_path):
        # ADIOS2's BP5 engine writes output_path as a directory (data.0, md.0,
        # md.idx, ...), not a single file -- os.path.isfile() would always be
        # False here even on success.
        raise DatasetUnavailable(f"enzo_to_bp5.py reported success but {output_path} is missing")
    return output_path


def ensure_dataset(cache_dir):
    """Returns the absolute path to the converted .bp file, fetching/converting it
    first if necessary. Raises DatasetUnavailable (caught by the TestCase, which
    turns it into a clean ctest skip) on any failure."""
    os.makedirs(cache_dir, exist_ok=True)

    bp_path = os.path.join(cache_dir, CONVERTED_BP_NAME)
    if os.path.exists(bp_path):
        # Same directory-not-file caveat as in _convert().
        return os.path.abspath(bp_path)

    data_path = os.path.join(cache_dir, ENZO_DATA_SUBPATH)
    if not os.path.exists(data_path):
        tarball_path = os.path.join(cache_dir, TARBALL_NAME)
        if not (os.path.isfile(tarball_path) and _checksums_ok(tarball_path)):
            tarball_path = _download(cache_dir)
        _extract(cache_dir, tarball_path)

    return os.path.abspath(_convert(cache_dir))


def _checksums_ok(path):
    try:
        _verify_checksums(path)
        return True
    except DatasetUnavailable:
        return False
