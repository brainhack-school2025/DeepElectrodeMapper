# Deep Electrode Mapper [In Development]
Project that started out of BrainHack Toronto School

## What is DeepElectrodeMapper?
Finding electrode coordinates currently is either a manual and time consuming task, that can be prone to human error, or requires specialized and expensive tools. DeepElectrodeMapper uses a deep learning algorithm, specifically PointNet++, to segment a 3D PointCloud file of scanned EEG electrodes cap, and extract the electrodes' coordinates, presenting a more accessible, efficient and generalizable strategy to acquire the EEG elctrode coordinates. 

## How does DeepElectrodeMapper functions?
DeepElectrodeMapper takes multiple file formats that includes the EEG cap, including, surface data formats('.obj', '.ply', '.stl', '.off') and volume data formats ('.nii', '.nii.gz'), standardize them into mesh (.obj) to preprocess, and transforms into 3D PointCloud to input to the trained PointNet++. With the points, that refer to the electrodes segmented and extracted, it is possible to find their centroids via clustering. DeepElectrodeMapper finds the coordinates of the centroids, which should align with the electrodes coordinates. 

## Resources
Digitization tools for 3D scanned EEGs:
- get_chanlocs EEGLAB plugin [https://github.com/sccn/get_chanlocs]
- Brainstorm3 [https://neuroimage.usc.edu/brainstorm/Tutorials/TutDigitize3dScanner]
- Meshlab PickPoints [https://www.protocols.io/view/spatial-localization-of-3d-scanned-eeg-electrodes-bf7ejrje]
