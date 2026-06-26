UDCT v4.5 Reproducibility Package
==================================

This package contains the official Gold-Standard Python script 
for reproducing the core quantitative results of the UDCT v4.5 paper.

Paper Title:
UDCT v4.5: Dynamical Emergence of Born Rule Statistics 
from the DSBB Stability Landscape

Key Result:
- Minimum L2 Relative Error: 2.41%
- Optimized Parameters: λ* ≈ 1.343, k_eff* ≈ 0.04185

Contents:
- UDCT_v4.5_Reproducibility_Package_v1.py : Main reproducibility script

How to Run (Recommended - Mode A):
1. Install required packages:
   pip install numpy scipy matplotlib

2. Place the script and the original data file 
   (udct_v4.4_born_test.npz) in the same folder.

3. Run the script:
   python UDCT_v4.5_Reproducibility_Package_v1.py

4. When prompted, choose 'N' to skip Mode B 
   (full 2D simulation is optional and slower).

The script will automatically reproduce the paper's 
reported results with high accuracy.

Mode A (Default):
- Uses the original paper dataset
- Reproduces the exact 2.41% L2 error reported in the paper

Mode B (Optional):
- Runs the full 2D DSBB simulation with Detector Freezing
- For advanced users who want to regenerate results from scratch

Author: Won Shik Paik
Affiliation: Independent Researcher, Auckland, New Zealand
Email: wspaik5@gmail.com

This code is intended to serve as a stable, long-term 
baseline for future research on the dynamical emergence 
of Born Rule statistics.
