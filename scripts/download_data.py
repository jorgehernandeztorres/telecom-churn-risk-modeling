#!/usr/bin/env python3
"""Download the CC BY 4.0 Iranian Churn CSV directly from UCI (ID 563).

Expected upstream archive: https://archive.ics.uci.edu/static/public/563/iranian+churn+dataset.zip
Data source/rights: https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset
"""
from __future__ import annotations
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError
import hashlib
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'data' / 'iranian_churn.csv'
URL = 'https://archive.ics.uci.edu/static/public/563/iranian+churn+dataset.zip'


def main():
    if DEST.exists():
        print(f'Already downloaded: {DEST} (remove this file to download again)')
        return
    request = Request(URL, headers={'User-Agent': 'Mozilla/5.0 (educational reproducibility study)'})
    try:
        with urlopen(request, timeout=35) as response:
            archive = response.read()
    except (URLError, TimeoutError) as exc:
        raise SystemExit(
            f'Unable to download original UCI archive: {exc}\n'
            'Download the dataset manually from https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset\n'
            'and place the original 14-column CSV at data/iranian_churn.csv.'
        ) from exc
    with zipfile.ZipFile(BytesIO(archive)) as z:
        candidates = [p for p in z.namelist()
                      if Path(p).name.lower() == 'customer churn.csv'
                      and not p.startswith('__MACOSX/')]
        if len(candidates) != 1:
            raise RuntimeError(f'Expected exactly one Customer Churn.csv in UCI archive; got {candidates}')
        payload = z.read(candidates[0])
    # Validate an on-disk temporary copy *before* committing it. A corrupt or
    # changed upstream schema must never leave a sticky invalid local CSV.
    DEST.parent.mkdir(parents=True, exist_ok=True)
    temporary = DEST.with_suffix('.csv.part')
    try:
        temporary.write_bytes(payload)
        from sys import path as sys_path
        sys_path.insert(0, str(ROOT))
        from src.data import load_uci_csv, data_audit
        audit = data_audit(load_uci_csv(temporary))
        temporary.replace(DEST)
    finally:
        temporary.unlink(missing_ok=True)
    print('Downloaded from:', URL)
    print('Local CSV:', DEST)
    print('Bytes:', len(payload))
    print('SHA-256:', hashlib.sha256(payload).hexdigest())
    print('Dataset audit:', audit)


if __name__ == '__main__':
    main()
