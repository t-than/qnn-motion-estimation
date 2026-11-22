#!/usr/bin/env bash
# Week 1 - clone and build the HEVC reference encoder (HM).
#
# Usage:  ./build_hm.sh [HM_REF]
#   HM_REF  git tag/branch/commit to build (default: HM-18.0, falls back to master)
#
# Result: ./HM/bin/TAppEncoderStatic and ./HM/bin/TAppDecoderStatic
#         ./HM/hm_build_info.txt records the exact commit (cite it in your report).
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HM_DIR="${PROJECT_DIR}/HM"
HM_REF="${1:-HM-18.0}"
HM_URL="https://vcgit.hhi.fraunhofer.de/jvet/HM.git"
JOBS="$(nproc)"

if [[ ! -d "${HM_DIR}/.git" ]]; then
    echo "==> Cloning HM from ${HM_URL}"
    git clone "${HM_URL}" "${HM_DIR}"
fi

cd "${HM_DIR}"
git fetch --tags --quiet || true
if git rev-parse --verify --quiet "${HM_REF}^{commit}" >/dev/null; then
    echo "==> Checking out ${HM_REF}"
    git checkout --quiet "${HM_REF}"
else
    echo "WARNING: ref '${HM_REF}' not found; building current master instead."
    echo "         Available HM tags (latest 10):"
    git tag --list 'HM-*' | sort -V | tail -10 | sed 's/^/           /'
    git checkout --quiet master
fi

# Every step ends in "|| return 1": build() is called as "if ! build", and bash
# ignores `set -e` inside a function called from an if-condition. Without these,
# a failed make was followed by a successful "cd ..", so build() returned 0 and the
# -Werror retry below never ran.
build() {
    rm -rf build && mkdir build && cd build || return 1
    cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_SEARCH_OPENSSL=off || return 1
    make -j"${JOBS}" || return 1
    cd ..
}

echo "==> Building HM (Release, ${JOBS} jobs)"
if ! build; then
    # HM compiles GCC builds with -Werror. Newer GCC releases (Arch ships the
    # latest) sometimes add warnings that HM's code triggers, which breaks the
    # build without anything being actually wrong. Retry with warnings allowed.
    # (GCC 16 on HM-18.0: -Warray-bounds false positive in TComLoopFilter.cpp and
    #  -Wunused-but-set-variable in SEIFilmGrainSynthesizer.cpp.)
    echo
    echo "==> Build failed; retrying with -Werror disabled (common with very new GCC)"
    cd "${HM_DIR}"
    sed -i 's/bb_enable_warnings( gcc warnings-as-errors/bb_enable_warnings( gcc/' CMakeLists.txt
    build
    echo "NOTE: built with warnings-as-errors disabled (local edit to CMakeLists.txt)."
fi

ENC="${HM_DIR}/bin/TAppEncoderStatic"
DEC="${HM_DIR}/bin/TAppDecoderStatic"
[[ -x "${ENC}" && -x "${DEC}" ]] || { echo "ERROR: binaries not found in ${HM_DIR}/bin"; exit 1; }

{
    echo "HM ref requested : ${HM_REF}"
    echo "HM commit        : $(git rev-parse HEAD)"
    echo "HM describe      : $(git describe --tags --always 2>/dev/null)"
    echo "Local changes    : $(git status --porcelain --untracked-files=no | wc -l) file(s)"
    echo "Compiler         : $(g++ --version | head -1)"
    echo "Built            : $(date -Iseconds)"
} | tee "${HM_DIR}/hm_build_info.txt"

echo
echo "==> Smoke test: encoding 5 synthetic frames"
TMP="$(mktemp -d)"
ffmpeg -loglevel error -y -f lavfi -i "testsrc2=size=416x240:rate=30,format=yuv420p" \
       -frames:v 5 "${TMP}/t.yuv"
"${ENC}" -c "${HM_DIR}/cfg/encoder_lowdelay_P_main.cfg" -i "${TMP}/t.yuv" \
         -wdt 416 -hgt 240 -fr 30 -f 5 -q 32 -b "${TMP}/t.bin" -o "${TMP}/rec.yuv" \
         > "${TMP}/enc.log" 2>&1
"${DEC}" -b "${TMP}/t.bin" -o "${TMP}/dec.yuv" > "${TMP}/dec.log" 2>&1
if cmp -s "${TMP}/rec.yuv" "${TMP}/dec.yuv"; then
    echo "OK: encoder reconstruction matches decoder output."
    grep -A2 "^SUMMARY" "${TMP}/enc.log" | tail -1
else
    echo "ERROR: encoder/decoder mismatch - something is wrong with the build."
    exit 1
fi
rm -rf "${TMP}"
