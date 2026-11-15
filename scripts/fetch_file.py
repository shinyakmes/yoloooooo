# -*- coding: utf-8 -*-
"""Resumable downloader that uses Python's TLS stack.

Why not curl/Invoke-WebRequest on this machine: both go through Windows Schannel,
which fails here with SEC_E_NO_CREDENTIALS under the sandbox, while Python's
OpenSSL path works fine.

Usage:
  python fetch_file.py <url> <dest>
"""
import os
import ssl
import sys
import time
import urllib.request

CTX = ssl.create_default_context()
UA = {'User-Agent': 'Mozilla/5.0'}


def human(n):
    return '%.1f MB' % (n / 1048576.0) if n > 1048576 else '%.0f KB' % (n / 1024.0)


def download(url, dst, retries=8):
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
                mode = 'ab' if (start and r.status == 206) else 'wb'
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
                        if time.time() - last > 5:
                            last = time.time()
                            print('  %s / %s (%s)' % (human(got),
                                                      human(total) if total else '?',
                                                      ('%.0f%%' % (100.0 * got / total)) if total else '?'),
                                  flush=True)
                size = os.path.getsize(dst)
                if total and size < total:
                    print('  short read: %d < %d, retrying' % (size, total), flush=True)
                    continue
                print('  done: %s (%s)' % (dst, human(size)), flush=True)
                return True
        except Exception as e:
            print('  attempt %d failed: %s %s' % (attempt, type(e).__name__, e), flush=True)
            time.sleep(2)
    return False


if __name__ == '__main__':
    url, dst = sys.argv[1], sys.argv[2]
    os.makedirs(os.path.dirname(dst) or '.', exist_ok=True)
    ok = download(url, dst)
    # sanity: torch .pt files are zip archives
    if ok:
        with open(dst, 'rb') as f:
            magic = f.read(4)
        print('  magic=%r  %s' % (magic, 'OK (zip/pt)' if magic[:2] == b'PK' else 'UNEXPECTED'))
    sys.exit(0 if ok else 1)
