#!/bin/bash
# Gisola-BaBoo installer for Linux (or WSL on Windows).
#
#   bash install.sh          checks what is needed and asks before installing anything missing
#   bash install.sh --yes    answers yes to every question (also for use without a terminal)
#   bash install.sh --no-gpu does not look for a GPU
#
# What it checks and, if you agree, installs (no administrator rights / sudo needed):
#   - conda          (Miniforge, into ~/miniforge3, if no conda is found)
#   - gfortran, make (from conda-forge, into the environment, if they are missing)
#   - the CUDA libraries for an NVIDIA GPU (from conda-forge, into the environment, only if
#     a GPU is found and Numba cannot use it; the NVIDIA driver itself is never installed)
# What it always does:
#   - creates the conda environment GisolaBaBoo from environment.yml (an existing one is kept)
#   - compiles AXITRA, pre-compiles the Numba kernels and adds ulimit to ~/.bashrc
# At the end it prints whether the GPU can be used. Without a GPU the program runs on the CPU.
set -eo pipefail

ENV_NAME="GisolaBaBoo"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BASE=""
ASSUME_YES=0
SKIP_GPU=0
GPU_STATUS=""

step() { echo; echo "==> $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }

usage() {
    sed -n '2,/^set -eo pipefail/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'
}

for arg in "$@"; do
    case "$arg" in
        -y|--yes)  ASSUME_YES=1 ;;
        --no-gpu)  SKIP_GPU=1 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $arg (use --help)" ;;
    esac
done

# ask "question": returns 0 for yes. Without --yes it asks on the terminal (default: no);
# without a terminal and without --yes it answers no.
ask() {
    if [ "$ASSUME_YES" = 1 ]; then
        echo "$1 [y/N] y (--yes)"
        return 0
    fi
    if [ -t 0 ]; then
        local reply
        read -r -p "$1 [y/N] " reply
        case "$reply" in y|Y|yes|YES) return 0 ;; *) return 1 ;; esac
    fi
    echo "$1 [y/N] n (no terminal to ask: run 'bash install.sh --yes' to agree)"
    return 1
}

# True when Numba can really use the GPU (driver, CUDA libraries and a small test kernel).
gpu_works() {
    (cd "$REPO_DIR/src" && python check_gpu.py)
}

# Looks for an NVIDIA GPU and, if Numba cannot use it, offers to install the CUDA libraries into
# the environment. The program also runs on the CPU, so nothing here stops the installation: the result is
# written to GPU_STATUS and printed at the end.
gpu_setup() {
    if [ "$SKIP_GPU" = 1 ]; then
        GPU_STATUS="not checked (--no-gpu): the CPU is used"
        return 0
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
        echo "  no NVIDIA GPU (or no NVIDIA driver) found"
        GPU_STATUS="no NVIDIA GPU found: the CPU is used"
        return 0
    fi
    local name cc cuda_max major minor target need cc_major
    name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)"
    cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ')"
    cuda_max="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n1)"
    echo "  found: $name (compute capability ${cc:-unknown}); the driver supports CUDA up to ${cuda_max:-unknown}"

    local result
    if result="$(gpu_works)"; then
        echo "  $result"
        GPU_STATUS="ready: $result"
        return 0
    fi
    echo "  Numba cannot use the GPU yet: $result"

    major="${cuda_max%%.*}"
    minor="${cuda_max#*.}"
    if [ -z "$cuda_max" ] || ! [ "$major" -ge 12 ] 2>/dev/null; then
        GPU_STATUS="not set up: the driver supports CUDA ${cuda_max:-(unknown)} and CUDA 12 or newer is needed. Update the NVIDIA driver, or see the manual."
        return 0
    fi
    # the newest CUDA 12 version that the driver supports (a newer driver also runs CUDA 12 libraries)
    if [ "$major" -eq 12 ]; then target="12.$(( minor > 9 ? 9 : minor ))"; else target="12.9"; fi
    # the oldest CUDA version that knows this GPU (the RTX 50 series and newer need 12.8)
    need="12.0"
    cc_major="${cc%%.*}"
    if [ -n "$cc_major" ] && [ "$cc_major" -ge 10 ] 2>/dev/null; then need="12.8"; fi
    if [ "${target#12.}" -lt "${need#12.}" ]; then
        GPU_STATUS="not set up: this GPU needs CUDA $need or newer, but the driver supports only CUDA $cuda_max. Update the NVIDIA driver."
        return 0
    fi

    if ask "  Install the CUDA $target libraries into the environment $ENV_NAME (from conda-forge, about 100 MB)?"; then
        if conda install -y -n "$ENV_NAME" -c conda-forge --override-channels                "cuda-version=$target" cuda-cudart cuda-nvvm-impl cuda-nvvm-tools; then
            if result="$(gpu_works)"; then
                echo "  $result"
                GPU_STATUS="ready: $result (CUDA $target libraries installed)"
            else
                GPU_STATUS="not working: the CUDA $target libraries were installed, but Numba still cannot use the GPU ($result). See the manual."
            fi
        else
            GPU_STATUS="not set up: the CUDA libraries could not be installed. See the manual."
        fi
    else
        GPU_STATUS="not set up (you answered no): the CPU is used. See the manual to set it up yourself."
    fi
    return 0
}

[ "$(uname -s)" = "Linux" ] || die "Gisola-BaBoo needs Linux (or WSL on Windows)."

# ---------------------------------------------------------------------------
step "Checking conda"
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    echo "  found: $CONDA_BASE"
else
    for d in "$HOME/miniforge3" "$HOME/mambaforge" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
        if [ -x "$d/bin/conda" ]; then CONDA_BASE="$d"; break; fi
    done
    if [ -n "$CONDA_BASE" ]; then
        echo "  found: $CONDA_BASE"
    else
        echo "  conda was not found."
        miniforge="Miniforge3-Linux-$(uname -m).sh"
        url="https://github.com/conda-forge/miniforge/releases/latest/download/$miniforge"
        if ask "  Install Miniforge (conda) into $HOME/miniforge3? About 100 MB; it also adds conda to ~/.bashrc."; then
            tmpdir="$(mktemp -d)"
            installer="$tmpdir/miniforge.sh"     # the installer only runs from a file named *.sh
            if command -v curl >/dev/null 2>&1; then
                curl -fsSL "$url" -o "$installer"
            elif command -v wget >/dev/null 2>&1; then
                wget -q "$url" -O "$installer"
            else
                die "neither curl nor wget is available to download Miniforge ($url)."
            fi
            bash "$installer" -b -p "$HOME/miniforge3"
            rm -f "$installer"
            rmdir "$tmpdir"
            CONDA_BASE="$HOME/miniforge3"
            "$CONDA_BASE/bin/conda" init bash >/dev/null
            echo "  conda installed (added to ~/.bashrc; open a new terminal to use it)"
        else
            echo
            echo "conda is needed. Install it yourself, for example Miniforge:"
            echo "    curl -L -O $url"
            echo "    bash $miniforge"
            echo "then open a new terminal and run this script again."
            exit 1
        fi
    fi
fi
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ---------------------------------------------------------------------------
step "Conda environment $ENV_NAME"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "  already exists, it is used as it is"
    echo "  (to build it again: conda env remove -n $ENV_NAME, then run this script again)"
else
    conda env create -f "$REPO_DIR/environment.yml"
fi
conda activate "$ENV_NAME"

# ---------------------------------------------------------------------------
step "Checking gfortran and make"
missing=()
command -v gfortran >/dev/null 2>&1 || missing+=(gfortran)
command -v make     >/dev/null 2>&1 || missing+=(make)
if [ "${#missing[@]}" -gt 0 ]; then
    echo "  missing: ${missing[*]}"
    if ask "  Install ${missing[*]} into the environment $ENV_NAME (from conda-forge)?"; then
        conda install -y -n "$ENV_NAME" -c conda-forge --override-channels "${missing[@]}"
        hash -r
    else
        echo
        echo "gfortran and make are needed to compile AXITRA. Install them yourself, for example:"
        echo "    conda install -n $ENV_NAME -c conda-forge gfortran make"
        echo "or with your system package manager (for Ubuntu: sudo apt install gfortran make),"
        echo "then run this script again."
        exit 1
    fi
else
    echo "  found: $(command -v gfortran), $(command -v make)"
fi

# ---------------------------------------------------------------------------
step "Compiling AXITRA (Green's functions)"
cd "$REPO_DIR/src/core/green"
# a fresh build: objects made by another compiler cannot be mixed with the new ones
rm -f ./*.o ./*.mod gr_xyz
make
[ -x gr_xyz ] || die "AXITRA was not compiled (src/core/green/gr_xyz is missing)."

# ---------------------------------------------------------------------------
step "Checking for an NVIDIA GPU (without one the program runs on the CPU)"
gpu_setup

# ---------------------------------------------------------------------------
step "Pre-compiling the Numba kernels (one time, about 20 s)"
cd "$REPO_DIR/src"
python precompile.py

# ---------------------------------------------------------------------------
step "Adding ulimit to ~/.bashrc"
touch ~/.bashrc
if ! grep -q "ulimit -s unlimited" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# Gisola: fix segmentation fault with large stack" >> ~/.bashrc
    echo "ulimit -s unlimited" >> ~/.bashrc
    echo "  -> Added 'ulimit -s unlimited' to ~/.bashrc"
else
    echo "  -> Already present in ~/.bashrc, skipping"
fi

echo ""
echo "============================================"
echo " Installation completed."
echo "============================================"
echo ""
echo " GPU: $GPU_STATUS"
echo ""
echo " To start using Gisola-BaBoo, open a new terminal"
echo " (so that conda and ulimit take effect) and run:"
echo ""
echo "   conda activate $ENV_NAME"
echo "   cd $REPO_DIR/src"
echo "   python GisolaBaBoo.py -c ../test/config.yaml --event-xml ../test/benchmark/event.xml"
echo ""
case "$GPU_STATUS" in
    ready*) echo " To use the GPU, set ComputeDevice: GPU in the configuration file." ;;
    *)      echo " The program runs on the CPU (ComputeDevice: CPU). GPU support is described in the manual." ;;
esac
echo ""
