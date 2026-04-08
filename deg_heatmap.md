{\rtf1\ansi\ansicpg1252\cocoartf2761
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\froman\fcharset0 Times-Roman;}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;}
{\*\expandedcolortbl;;\cssrgb\c0\c0\c0;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\sa321\partightenfactor0

\f0\fs22 \cf0 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 # **CAD Degradation Heatmap Pipeline**\
\
## **Overview**\
\
`cad_degradation_heatmap.py` is a prototype tool for mapping degradation behavior onto a CAD-derived device geometry. It converts a 3D mesh into a voxel grid, simulates degradation using a simplified reaction\'96diffusion model, and outputs spatial heatmaps of material properties and potential weak points.\
\
This is intended for **early-stage design screening**, not validated prediction or final analysis.\
\
---\
\
## **Objective**\
\
The goal of this pipeline is to link **geometry + material degradation** to identify where a device may weaken over time.\
\
Specifically, it allows us to:\
- visualize how degradation progresses spatially (not just bulk)  \
- estimate changes in molecular weight, porosity, and modulus  \
- highlight regions that may be more failure-prone due to geometry + degradation  \
\
---\
\
## **Workflow**\
\
The script follows the pipeline:\
\
1. **Load mesh** (STL/OBJ/PLY/GLB)  \
2. **Voxelize geometry** into a 3D grid  \
3. **Run degradation simulation** (reaction\'96diffusion + kinetics)  \
4. **Compute derived fields** (modulus, porosity, risk)  \
5. **Export results** (data + visualizations)  \
\
---\
\
## **Model Summary**\
\
The simulation evolves several fields over time inside the device:\
\
- **Water concentration (`W`)**  \
- **Acid/oligomer concentration (`A`)**  \
- **Relative molecular weight (`Mn_rel`)**  \
- **Porosity (`phi`)**  \
- **Relative modulus (`E_rel`)**  \
\
Key processes included:\
- water ingress from exposed surfaces  \
- diffusion of water and acid  \
- hydrolysis + autocatalytic degradation  \
- porosity growth with degradation  \
- modulus decay as material weakens  \
\
Geometry is incorporated through:\
- **distance to surface** (depth)  \
- **curvature proxy** (edges/corners)  \
- **thinness proxy** (ligament-like regions)  \
\
---\
\
## **Inputs**\
\
- Mesh file (`.stl`, `.obj`, `.ply`, `.glb`)  \
- Key parameters:\
  - **voxel size** (`--voxel-pitch`)  \
  - **simulation time** (`--days`)  \
  - **time step** (`--dt`)  \
\
**Example:**\
\
    python cad_degradation_heatmap.py device.stl --voxel-pitch 0.4 --days 180 --dt 0.05 --outdir out_device\
\
---\
\
## **Outputs**\
\
All outputs are saved to the specified directory:\
\
### **Data**\
- `fields_final.csv` \uc0\u8594  voxel coordinates + field values  \
- `fields_final.npz` \uc0\u8594  compressed arrays  \
\
### **3D Visualization**\
- `Mn_rel_final.ply`  \
- `porosity_final.ply`  \
- `modulus_final.ply`  \
- `risk_final.ply`  \
\
*(Open in MeshLab, ParaView, or Blender)*  \
\
### **2D Slices**\
- Mid-plane heatmaps for each field  \
\
### **Time Trends**\
- Plots of mean Mn, porosity, modulus, and risk over time  \
\
### **Summary**\
- `summary.txt` with key final values and interpretation  \
\
---\
\
## **Key Assumptions & Limitations**\
\
- Simplified reaction\'96diffusion model (parameters not fully calibrated)  \
- No full mechanical modeling (no FEA)  \
- Stress represented by geometric proxies  \
- Accuracy depends on voxel size and time step  \
- Requires watertight mesh for best results  \
\
This should be treated as a **conceptual and comparative tool**, not a predictive simulator.\
\
---\
\
## **Use in Project Context**\
\
This tool can be used to:\
- compare design iterations (e.g., clip geometries)  \
- identify potential weak regions before fabrication  \
- guide where more detailed simulation or testing is needed  }