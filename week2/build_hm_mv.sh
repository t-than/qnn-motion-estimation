#!/usr/bin/env bash
# Week 2 - build HM-mv: HM-18.0 plus the changes in hm-mv.patch.
#
# HM-mv is a git worktree of ../week1/HM (same repository, separate checkout), so the
# week-1 anchor binary stays untouched. hm-mv.patch, tracked in this repo, is the
# source of truth for the changes.
#
# Usage:
#   ./build_hm_mv.sh               create HM-mv if needed (HM-18.0 + hm-mv.patch), build, smoke test
#   ./build_hm_mv.sh --save-patch  write the current HM-mv changes to hm-mv.patch (no build)
#   ./build_hm_mv.sh --fresh       discard the HM-mv source tree, re-create it from HM-18.0 +
#                                  hm-mv.patch, build. Unsaved edits in HM-mv are lost.
#
# Workflow: edit the sources in HM-mv/, build with ./build_hm_mv.sh, and save with
# --save-patch before committing. The build warns if HM-mv has changes that are not in
# hm-mv.patch yet.
#
# Result: HM-mv/bin/TAppEncoderStatic and HM-mv/bin/TAppDecoderStatic,
#         HM-mv/hm_mv_build_info.txt (HM commit, patch checksum, compiler).
set -euo pipefail

WEEK2="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HM_SRC="${WEEK2}/../week1/HM"
HM_MV="${WEEK2}/HM-mv"
PATCH="${WEEK2}/hm-mv.patch"
HM_REF="HM-18.0"
JOBS="$(nproc)"
MODE="${1:-build}"

# Never part of the patch: HM's CMakeLists.txt (edited by the -Werror fallback below, a
# build workaround, not a motion-vector change) and this script's build-info file.
# Build output (build/, bin/, lib/) is already ignored by HM's own .gitignore.
PATCH_EXCLUDE=(':(exclude)CMakeLists.txt' ':(exclude)hm_mv_build_info.txt')

case "${MODE}" in
    build|--fresh|--save-patch) ;;
    *) echo "unknown option: ${MODE}  (use --save-patch or --fresh)"; exit 1 ;;
esac

[[ -d "${HM_SRC}/.git" ]] || { echo "ERROR: ${HM_SRC} not found; run week1/build_hm.sh first"; exit 1; }

if [[ "${MODE}" == "--save-patch" ]]; then
    [[ -d "${HM_MV}" ]] || { echo "ERROR: ${HM_MV} does not exist yet"; exit 1; }
    git -C "${HM_MV}" add --intent-to-add --all -- . "${PATCH_EXCLUDE[@]}"   # include new files
    git -C "${HM_MV}" diff --binary "${HM_REF}" -- . "${PATCH_EXCLUDE[@]}" > "${PATCH}"
    echo "Wrote ${PATCH}: $(grep -c '^diff --git' "${PATCH}" || true) file(s), $(wc -l < "${PATCH}") lines"
    exit 0
fi

if [[ "${MODE}" == "--fresh" && -d "${HM_MV}" ]]; then
    echo "==> --fresh: resetting ${HM_MV} to ${HM_REF} (unsaved edits are lost)"
    git -C "${HM_MV}" reset --quiet --hard "${HM_REF}"
    git -C "${HM_MV}" clean --quiet -fdx -e bin -e lib -e build
    NEED_PATCH=1
fi

if [[ ! -d "${HM_MV}" ]]; then
    echo "==> Creating worktree ${HM_MV} at ${HM_REF}"
    git -C "${HM_SRC}" worktree prune
    git -C "${HM_SRC}" worktree add --detach "${HM_MV}" "${HM_REF}"
    NEED_PATCH=1
fi

cd "${HM_MV}"
if [[ "${NEED_PATCH:-0}" == 1 ]]; then
    if [[ -s "${PATCH}" ]]; then
        echo "==> Applying $(basename "${PATCH}")"
        git apply --whitespace=nowarn "${PATCH}"
    else
        echo "==> No patch yet (${PATCH} missing or empty): building plain ${HM_REF}"
    fi
fi

build() {
    # "|| return 1" on every step: build() is called as "if ! build", and bash ignores
    # `set -e` inside a function called from an if-condition (week 1, lesson 1).
    rm -rf build && mkdir build && cd build || return 1
    cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_SEARCH_OPENSSL=off || return 1
    make -j"${JOBS}" || return 1
    cd ..
}

echo "==> Building HM-mv (Release, ${JOBS} jobs)"
if ! build; then
    # Same fallback as week1/build_hm.sh: GCC 16 warns in HM-18.0 code that is not
    # actually wrong, and HM builds with -Werror.
    echo
    echo "==> Build failed; retrying with -Werror disabled (common with very new GCC)"
    cd "${HM_MV}"
    sed -i 's/bb_enable_warnings( gcc warnings-as-errors/bb_enable_warnings( gcc/' CMakeLists.txt
    build
fi
if grep -q 'bb_enable_warnings( gcc warnings-as-errors' CMakeLists.txt; then
    WERROR="on"
else
    WERROR="off (local edit to CMakeLists.txt, not part of the patch)"
fi

ENC="${HM_MV}/bin/TAppEncoderStatic"
DEC="${HM_MV}/bin/TAppDecoderStatic"
[[ -x "${ENC}" && -x "${DEC}" ]] || { echo "ERROR: binaries not found in ${HM_MV}/bin"; exit 1; }

# Are there changes in HM-mv that hm-mv.patch does not contain?
git add --intent-to-add --all -- . "${PATCH_EXCLUDE[@]}"
CURRENT="$(git diff --binary "${HM_REF}" -- . "${PATCH_EXCLUDE[@]}")"
SAVED="$( [[ -f "${PATCH}" ]] && cat "${PATCH}" || true )"
if [[ -z "${CURRENT}" && -z "${SAVED}" ]]; then
    PATCH_STATE="plain ${HM_REF} (no changes, no patch)"
elif [[ "${CURRENT}" == "${SAVED}" ]]; then
    PATCH_STATE="matches hm-mv.patch"
else
    PATCH_STATE="DIFFERS from hm-mv.patch (run ./build_hm_mv.sh --save-patch before committing)"
fi

{
    echo "HM base          : ${HM_REF} ($(git rev-parse "${HM_REF}^{commit}"))"
    echo "Patch            : $( [[ -s "${PATCH}" ]] && md5sum "${PATCH}" | cut -d' ' -f1 || echo none)"
    echo "Source tree      : ${PATCH_STATE}"
    echo "Changed files    : $(git diff --name-only "${HM_REF}" -- . "${PATCH_EXCLUDE[@]}" | wc -l)"
    echo "-Werror          : ${WERROR}"
    echo "Compiler         : $(g++ --version | head -1)"
    echo "Built            : $(date -Iseconds)"
} | tee "${HM_MV}/hm_mv_build_info.txt"

echo
echo "==> Smoke test: encoding 5 synthetic frames"
TMP="$(mktemp -d)"
ffmpeg -loglevel error -y -f lavfi -i "testsrc2=size=416x240:rate=30,format=yuv420p" \
       -frames:v 5 "${TMP}/t.yuv"
"${ENC}" -c "${HM_MV}/cfg/encoder_lowdelay_P_main.cfg" -i "${TMP}/t.yuv" \
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
