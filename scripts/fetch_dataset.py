# -*- coding: utf-8 -*-
"""D2 step 1: fetch NEU-DET (raw VOC: IMAGES + ANNOTATIONS) from GitHub to E:\yolo\datasets."""
import os
import ssl
import sys
import time
import urllib.request
import zipfile

CTX = ssl.create_default_context()
UA = {'User-Agent': 'Mozilla/5.0'}
ROOT = r'E:\yolo\datasets'
ZIP = os.path.join(ROOT, 'neudet_github.zip')
URL = ('https://codeload.github.com/siddhartamukherjee/'
       'NEU-DET-Steel-Surface-Defect-Detection/zip/refs/heads/master')


def log(msg):
    line = time.strftime('%H:%M:%S') + '  ' + msg
    print(line, flush=True)
    with open(os.path.join(ROOT, 'fetch_dataset.log'), 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def download(url, dst, retries=5):
    for attempt in range(1, retries + 1):
        start = os.path.getsize(dst) if os.path.exists(dst) else 0
        headers = dict(UA)
        if start:
            headers['Range'] = 'bytes=%d-' % start
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60, context=CTX) as r:
                total = r.headers.get('Content-Length')
                total = int(total) + start if total else None
                mode = 'ab' if start and r.status == 206 else 'wb'
                if mode == 'wb':
                    start = 0
                got = start
                last = time.time()
                with open(dst, mode) as f:
                    while True:
                        chunk = r.read(1 << 16)
                        if not chunk:
                            break
                        f.write(chunk)
                        got += len(chunk)
                        if time.time() - last > 3:
                            last = time.time()
                            pct = ('%.1f%%' % (100.0 * got / total)) if total else '?'
                            log('  download %d/%s bytes (%s)' % (got, total, pct))
            log('  download finished, size=%d' % os.path.getsize(dst))
            return dst
        except Exception as e:
            log('  attempt %d FAILED: %s %s' % (attempt, type(e).__name__, e))
            time.sleep(3)
    raise RuntimeError('download failed after %d attempts' % retries)


def main():
    os.makedirs(ROOT, exist_ok=True)
    log('=== fetch NEU-DET ===')
    download(URL, ZIP)

    out = os.path.join(ROOT, 'NEU-DET')
    os.makedirs(out, exist_ok=True)
    with zipfile.ZipFile(ZIP) as z:
        names = z.namelist()
        log('zip entries=%d' % len(names))
        wanted = [n for n in names if '/IMAGES/' in n or '/ANNOTATIONS/' in n]
        log('entries to extract=%d' % len(wanted))
        for i, n in enumerate(wanted):
            z.extract(n, out)
            if (i + 1) % 500 == 0:
                log('  extracted %d/%d' % (i + 1, len(wanted)))

    # flatten: zip has one top folder, so pull its children up one level
    subs = [d for d in os.listdir(out) if os.path.isdir(os.path.join(out, d))]
    if len(subs) == 1:
        inner = os.path.join(out, subs[0])
        for child in os.listdir(inner):
            src = os.path.join(inner, child)
            dst = os.path.join(out, child)
            if not os.path.exists(dst):
                os.rename(src, dst)
        try:
            os.rmdir(inner)
            log('  flattened top folder "%s"' % subs[0])
        except OSError as e:
            log('  flatten left over: %s' % e)

    for d in ('IMAGES', 'ANNOTATIONS'):
        p = os.path.join(out, d)
        if os.path.isdir(p):
            n = len(os.listdir(p))
            log('  %s: %d files' % (d, n))
        else:
            log('  MISSING %s' % d)

    log('=== fetch DONE ===')


if __name__ == '__main__':
    main()
