#!/usr/bin/env bash
# Fetch third-party code at pinned commits and build src/modules/{rots,owmd,wmdo}.py.
# The upstream code is not redistributed in this repository.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="${REPO_ROOT}/third_party"

fetch() {  # name url commit
    if [ ! -d "${THIRD_PARTY}/$1" ]; then
        git clone "$2" "${THIRD_PARTY}/$1"
    fi
    git -C "${THIRD_PARTY}/$1" checkout --quiet "$3"
}

mkdir -p "${THIRD_PARTY}"
fetch rots https://github.com/zihao-wang/rots.git 27f8214bd28147286da5481924c3bb031d2d55a9
fetch SentenceMatching https://github.com/BangLiu/SentenceMatching.git 4b8c194206d647fa6b211fc426d1e6fbc0726cf2
fetch WMDO https://github.com/julianchow/WMDO.git f1716cc551dd5c0aff1c6694f8e1a64e93bb0dfb

python "${REPO_ROOT}/setup/build_third_party_modules.py" "${REPO_ROOT}"
git -C "${REPO_ROOT}" apply --unidiff-zero patches/third_party_modules.patch
echo "src/modules/{rots,owmd,wmdo}.py are ready"
