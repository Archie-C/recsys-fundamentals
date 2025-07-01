import hashlib, pathlib, urllib.request, zipfile, argparse, sys

URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
MD5 = "0e33842e24a9c977be4e0107933c0723"
ZIP_NAME = "ml-100k.zip"
RAW_DIR = pathlib.Path("data/raw")

def md5sum(fp: pathlib.Path) -> str:
    return hashlib.md5(fp.read_bytes()).hexdigest()

def download(dest: pathlib.Path):
    print(f"Downloading {URL} → {dest} ...")
    urllib.request.urlretrieve(URL, dest)

def unpack(zip_fp: pathlib.Path, out_dir: pathlib.Path):
    print(f"Unpacking to {out_dir} ...")
    with zipfile.ZipFile(zip_fp, 'r') as zf:
        zf.extractall(out_dir)

def main(force: bool):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_fp = RAW_DIR / ZIP_NAME

    if zip_fp.exists() and not force:
        if md5sum(zip_fp) == MD5:
            print("✓ dataset already present and checksum OK – nothing to do.")
            return
        else:
            print("⚠ existing file failed checksum, re-downloading…")
    if force and zip_fp.exists():
        zip_fp.unlink()

    download(zip_fp)

    if md5sum(zip_fp) != MD5:
        sys.exit("✗ MD5 mismatch – download corrupted. Aborting.")

    unpack(zip_fp, RAW_DIR)
    print("✓ Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="redownload even if file seems OK")
    main(parser.parse_args().force)