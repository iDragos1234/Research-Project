# Research-Project
Code for the final Research Project of my Bachelor of Science in Computer Science and Engineering. Topic: "Deep Learning for Automated segmentation of the hip joint in X-ray images".

## How to run?

### Preprocessing step

To run the the preprocessing step on a regular computer, run the following command in the bash terminal:
```
python ./dicom_preprocessor/dicom_preprocessor.py \
    --input "./data" \
    --output "output.h5" \
    --target-pixel-spacing 1
```

To run the preprocessing step on the DelftBlue supercomputer, run the following command in the bash terminal of your DelftBlue login node:
```
sbatch run.sh
```

The preprocessing step will perform specific transformations on all data from the input folder and write it to an HDF5 file as output.

