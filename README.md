# Gisola-BaBoo

Gisola-BaBoo ([Velentzas et al., 2026](https://doi.org/10.5194/egusphere-egu26-21600)) is an open-source software for the automatic computation of earthquake moment tensors, with an estimate of the uncertainty of each solution. It processes single events, lists of events, catalogs and time ranges, and it can run in real time, using the data of standard seismological web services (FDSN/EIDA) or local data.

It is based on two earlier works:

- [Gisola](https://github.com/nikosT/Gisola) ([Triantafyllis et al., 2021](https://doi.org/10.1785/0220210153)): an application for real-time moment tensor inversion. It provides the workflow from the selection of the event, the stations and the data to the results and web pages.
- ISOLA-BaBoo ([Zahradník et al., 2026](https://doi.org/10.26443/seismica.v5i1.2135)): a code for moment tensor inversion with uncertainty estimation by a Bayesian bootstrap.

Gisola-BaBoo brings the Bayesian bootstrap into the Gisola workflow, with a new implementation of the inversion in Python and Numba that runs on the CPU or on an NVIDIA GPU.

For each event the program selects the stations and downloads their data, computes Green's functions, inverts for the centroid moment tensor, estimates the uncertainty with the bootstrap, and writes the results and the web pages.

Documentation: the [user manual](docs/GisolaBaBoo_manual.md) (also as [PDF](docs/GisolaBaBoo_User_Manual.pdf)).

## Installation

Linux (or WSL on Windows). No administrator rights are needed.

Get Gisola-BaBoo from GitHub:

```bash
git clone https://github.com/VasilisVel/Gisola-BaBoo.git ~/Gisola-BaBoo
cd ~/Gisola-BaBoo
```

### Automatic installation

```bash
bash install.sh
```

The script checks for conda, `gfortran` and `make`, and asks your permission before it downloads anything that is missing (the default answer is no). Then it creates the conda environment `GisolaBaBoo`, compiles AXITRA, checks for an NVIDIA GPU if you have one (without a GPU the program runs on the CPU; see the manual), and pre-compiles the Numba kernels. `bash install.sh --yes` answers yes to every question, for a computer without a terminal.

### Manual installation

You can also do the steps yourself:

1. **conda** (skip this if you already have it). Install Miniforge, and open a new terminal afterwards:
   ```bash
   curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh
   bash Miniforge3-Linux-$(uname -m).sh
   ```
2. **The environment** (in the main folder):
   ```bash
   cd ~/Gisola-BaBoo
   conda env create -f environment.yml
   conda activate GisolaBaBoo
   ```
3. **gfortran and make** (skip this if you already have them):
   ```bash
   conda install -c conda-forge --override-channels gfortran make
   ```
   or, on Ubuntu: `sudo apt install gfortran make`
4. **Compile AXITRA:**
   ```bash
   (cd src/core/green && make)
   ```
5. **Pre-compile the Numba kernels:**
   ```bash
   (cd src && python precompile.py)
   ```
6. **Optional:** run `ulimit -s unlimited` before the program (add it to `~/.bashrc`), if a run stops with a segmentation fault.

## Test the installation

Two examples, to run from the `src` folder:

```bash
conda activate GisolaBaBoo
cd ~/Gisola-BaBoo/src

# 1. Local run of the benchmark event: it uses local data, so it runs with a different configuration (see the manual).
python GisolaBaBoo.py -c ../test/config.yaml --event-xml ../test/benchmark/event.xml

# 2. Run that communicates with the server to get 10 events of 2013: it runs with the default configuration (see the manual).
python GisolaBaBoo.py --event-file ../test/greece2013_events.txt
```

The results are written to `test/results/` for the first run and to `output/` for the second one. The page of each solution is `index.html` in the `output` folder of its run.

## License and credits

GNU General Public License v3 (see `LICENSE`). Gisola-BaBoo is based on Gisola, which was written by Nikolaos Triantafyllis and co-authors. If you use Gisola-BaBoo, please cite:

- Velentzas, V., Zahradník, J., Psarakis, E., Evangelidis, C., Sokos, E. (2026). Bayesian bootstrapping extension of GISOLA automatic moment tensor inversion software. EGU General Assembly 2026 (EGUsphere). https://doi.org/10.5194/egusphere-egu26-21600

- Zahradník, J., Sokos, E., Turhan, F. (2026). ISOLA-BaBoo, a code for full moment tensor inversion with uncertainty estimation; application to the 2025 Anydros earthquakes. Seismica, 5(1). https://doi.org/10.26443/seismica.v5i1.2135

- Triantafyllis, N., Venetis, I. E., Fountoulakis, I., Pikoulis, E.-V., Sokos, E., Evangelidis, C. P. (2021). Gisola: A High-Performance Computing Application for Real-Time Moment Tensor Inversion. Seismological Research Letters. https://doi.org/10.1785/0220210153
