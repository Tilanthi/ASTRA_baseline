# Copyright 2026 Glenn J. White
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Execute every ```python block in the project documentation.

Run from the repository root:

    python astra_core/tests/check_doc_examples.py

Exits non-zero if any documented example fails. This exists because an audit
found that 16 of 24 documented examples did not run at all: the User Manual
described 18 methods and 2 imports that have never existed in the codebase.
Blocks are run cumulatively per document, so a snippet may build on the ones
before it in the same file.
"""

import re, sys, subprocess, pathlib


def main() -> int:
    DOCS = ["README.md", "BASELINE_README.md", "CLAUDE.md", "User_Manual/User_Manual.md"]
    real = []
    for doc in DOCS:
        text = pathlib.Path(doc).read_text(encoding="utf-8")
        blocks = re.findall(r"```python\n(.*?)```", text, re.S)
        prefix = ""
        for i, code in enumerate(blocks, 1):
            script = prefix + "\n" + code
            proc = subprocess.run([sys.executable, "-c", script],
                                  capture_output=True, text=True, timeout=300)
            if proc.returncode == 0:
                prefix = script          # this block worked; later ones may build on it
                continue
            err = proc.stderr.strip().split("\n")[-1]
            # a block that only fails on names never defined anywhere is illustrative
            real.append((doc, i, err, code.strip().split("\n")[0][:64]))
    print(f"blocks still failing with cumulative context: {len(real)}\n")
    for doc, i, err, first in real:
        print(f"{doc} block {i}: {first}")
        print(f"    -> {err[:130]}")
    return 1 if real else 0


if __name__ == "__main__":
    sys.exit(main())
