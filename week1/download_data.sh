#!/usr/bin/env bash
# Week 1 - download test sequences and training datasets.
#
# Usage:  ./download_data.sh <set> [<set> ...]
#   small    Xiph CIF classics (foreman, bus, football, mobile)       ~150 MB
#   uvg      UVG 1080p 8-bit: the 7 sequences used in most papers     ~12 GB raw (+ the .7z archives)
#   xiphhd   Xiph/SVT 1080p50 (crowd_run, park_joy, ...)  optional    ~7.5 GB
#   chairs   FlyingChairs + official train/val split                  ~30 GB zipped, ~60 GB extracted
#   sintel   MPI-Sintel complete (training flow + test)               ~5.5 GB
#   all      small + uvg + chairs + sintel
#
# Downloads resume if interrupted (aria2c -c). Re-running skips finished files.
# Data goes to ./data/raw/<set>/ ; run prepare_sequences.py afterwards.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW="${PROJECT_DIR}/data/raw"
mkdir -p "${RAW}"

fetch() {  # fetch <url> <dest_dir>
    local url="$1" dest="$2" name
    name="$(basename "${url}")"
    mkdir -p "${dest}"
    if [[ -f "${dest}/${name}" && ! -f "${dest}/${name}.aria2" ]]; then
        echo "   skip (done) ${name}"
        return 0
    fi
    echo "   get ${name}"
    if ! aria2c -c -x 8 -s 8 --file-allocation=none --summary-interval=30 \
                --console-log-level=warn -d "${dest}" -o "${name}" "${url}"; then
        echo "   WARNING: failed to download ${url}" >&2
        FAILED+=("${url}")
    fi
}

check_space() {  # check_space <GB needed>
    local need="$1" avail
    avail="$(df --output=avail -BG "${RAW}" | tail -1 | tr -dc '0-9')"
    if (( avail < need )); then
        echo "WARNING: only ${avail} GB free in ${RAW}, this set needs about ${need} GB." >&2
        read -r -p "Continue anyway? [y/N] " ans
        [[ "${ans}" == [yY]* ]] || exit 1
    fi
}

get_small() {
    echo "==> Xiph CIF sequences"
    for s in foreman_cif bus_cif football_cif mobile_cif; do
        fetch "https://media.xiph.org/video/derf/y4m/${s}.y4m" "${RAW}/small"
    done
}

get_uvg() {
    echo "==> UVG 1080p 8-bit (license: CC BY-NC; cite Mercat et al., MMSys 2020)"
    check_space 40
    local base="https://tie-ultravideo.rd.tuni.fi/video"
    for s in Beauty Bosphorus HoneyBee Jockey ReadySetGo ShakeNDry YachtRide; do
        fetch "${base}/${s}_1920x1080_120fps_420_8bit_YUV_RAW.7z" "${RAW}/uvg"
    done
    echo "==> Extracting UVG archives"
    for a in "${RAW}"/uvg/*.7z; do
        [[ -e "${a}" ]] || continue
        local marker="${a}.extracted"
        [[ -f "${marker}" ]] && continue
        7z x -y -o"${RAW}/uvg" "${a}" >/dev/null && touch "${marker}"
    done
    echo "   You may delete ${RAW}/uvg/*.7z after extraction to save space."
}

get_xiphhd() {
    echo "==> Xiph/SVT 1080p50 sequences"
    check_space 10
    for s in crowd_run park_joy ducks_take_off old_town_cross in_to_tree; do
        fetch "https://media.xiph.org/video/derf/y4m/${s}_1080p50.y4m" "${RAW}/xiphhd"
    done
}

get_chairs() {
    echo "==> FlyingChairs (cite Dosovitskiy et al., ICCV 2015)"
    check_space 70
    fetch "https://lmb.informatik.uni-freiburg.de/data/FlyingChairs/FlyingChairs.zip" "${RAW}/chairs"
    fetch "https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs/FlyingChairs_train_val.txt" "${RAW}/chairs"
    if [[ ! -d "${RAW}/chairs/FlyingChairs_release" && -f "${RAW}/chairs/FlyingChairs.zip" ]]; then
        echo "   extracting (takes a while, 22k image pairs)"
        unzip -q "${RAW}/chairs/FlyingChairs.zip" -d "${RAW}/chairs"
    fi
    if [[ -d "${RAW}/chairs/FlyingChairs_release" ]]; then
        cp -n "${RAW}/chairs/FlyingChairs_train_val.txt" "${RAW}/chairs/FlyingChairs_release/" || true
    fi
}

get_sintel() {
    echo "==> MPI-Sintel (cite Butler et al., ECCV 2012)"
    check_space 12
    fetch "http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip" "${RAW}/sintel"
    if [[ ! -d "${RAW}/sintel/training" && -f "${RAW}/sintel/MPI-Sintel-complete.zip" ]]; then
        unzip -q "${RAW}/sintel/MPI-Sintel-complete.zip" -d "${RAW}/sintel"
    fi
}

[[ $# -ge 1 ]] || { sed -n '2,14p' "$0"; exit 1; }
command -v aria2c >/dev/null || { echo "aria2c missing: sudo pacman -S aria2"; exit 1; }

FAILED=()
for set in "$@"; do
    case "${set}" in
        small)  get_small ;;
        uvg)    get_uvg ;;
        xiphhd) get_xiphhd ;;
        chairs) get_chairs ;;
        sintel) get_sintel ;;
        all)    get_small; get_uvg; get_chairs; get_sintel ;;
        *) echo "unknown set: ${set}"; exit 1 ;;
    esac
done

if (( ${#FAILED[@]} )); then
    echo
    echo "Some downloads failed (servers move files occasionally):"
    printf '  %s\n' "${FAILED[@]}"
    echo "Check the dataset's web page for the current link and re-run."
    exit 1
fi
echo "All requested sets downloaded."
