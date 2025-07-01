import hashlib, pathlib

def test_ml100k_checksum():
    fn = pathlib.Path("data/raw/ml-100k.zip")
    md5 = hashlib.md5 (fn.read_bytes()).hexdigest()
    assert md5 == "0e33842e24a9c977be4e0107933c0723"