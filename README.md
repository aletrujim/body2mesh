# body2mesh

**body2mesh: From Monocular Video to 3D  Anthropometry Using Deep Learning**

body2mesh is a Deep Learning-based pipeline for low-cost three-dimensional (3D) human body reconstruction and anthropometric assessment from monocular video recordings. The framework generates 3D body meshes from conventional RGB videos and automatically estimates obesity-related anthropometric measurements, including waist circumference (WC), hip circumference (HC), waist-to-hip ratio (WHR), and waist-to-height ratio (WHtR).

The project was developed for applications in biological anthropology, body composition assessment, obesity research, and large-scale population studies.

---

## Features

* Human body reconstruction from monocular RGB videos
* Background segmentation
* Human pose estimation
* PIFuHD-based 3D mesh reconstruction
* Automatic mesh post-processing
* Automatic waist and hip circumference estimation
* WHR and WHtR computation
* HPC-compatible implementation (Narval Cluster, Digital Research Alliance of Canada)
* Batch processing of large datasets

---

## Pipeline

The body2mesh workflow consists of the following stages:

1. Video acquisition
2. Frame extraction
3. Background removal
4. Human pose estimation
5. 3D reconstruction using PIFuHD
6. Mesh post-processing

   * Scaling to real height
   * Head anonymization
   * Arm removal
   * Hole filling
   * Laplacian smoothing
7. Automatic anthropometric analysis

   * Waist circumference
   * Hip circumference
   * Waist-to-hip ratio (WHR)
   * Waist-to-height ratio (WHtR)

---

## Repository Structure

```text
body2mesh/
│
├── body2mesh_run/        # Main reconstruction pipeline
├── human-pose/           # OpenPose estimation
├── pifuhd/               # PIFuHD reconstruction model
├── notebooks/            # Analysis and visualization notebooks
├── tests/                # Validation scripts
├── run_body2mesh.sh      # HPC execution script
├── measures/             # Obtain 3D anthropometric measurements 
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/aletrujim/body2mesh.git
cd body2mesh
```

Create a Python environment:

```bash
conda create -n body2mesh python=3.10
conda activate body2mesh
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Input Data

The pipeline expects RGB video recordings of a single subject.

Recommended acquisition protocol:

* Full-body recording
* Front, back, left, and right views
* Standing anatomical posture
* Slight arm separation
* Tight-fitting clothing
* Natural or uniform lighting
* Conventional smartphone or digital camera

---

## Running the Pipeline

Example:

```bash
bash run_body2mesh.sh
```

---

## Computational Requirements

The pipeline was validated on the Narval Cluster (FIR-Supercomputer, Simon Fraser University) using:

* NVIDIA A100 GPU
* 16–32 GB RAM
* 4–8 CPU cores

Average runtime:

* ~157 seconds per subject

The pipeline can also be executed on standard GPU-enabled workstations.

---

## Citation

If you use body2mesh in your research, please cite:

```bibtex

```

---

## Related Publications

```bibtex

```

---

## License

This repository is released for academic and non-commercial research purposes.

Please cite the corresponding publication when using the code, data, or derived models.

---

## Contact

**Magda Alexandra Trujillo-Jiménez**

Imaging Science Laboratory (LCI-UNS)
Patagonian Institute of Social and Human Sciences (IPCSH-CONICET)
Puerto Madryn, Argentina

GitHub: https://github.com/aletrujim

