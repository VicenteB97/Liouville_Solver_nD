# adaptivecartesianMeshRefinement
## Summary
This class is meant to work as an *interface* between a compression algorithm (such as a wavelet transform) and the full solution information of the Liouville PDE.

## Usage and properties
In the current use case, the core functionality is the following:
- Set the initial signal and the cartesianMesh information
- Create instance of specific compression algorithm
- Recover compressed information
- Translate information to cartesianMesh-Particle information