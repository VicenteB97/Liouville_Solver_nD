# adaptiveMeshRefinement
## Summary
This class is meant to work as an *interface* between a compression algorithm (such as a wavelet transform) and the full solution information of the Liouville PDE.

## Usage and properties
In the current use case, the core functionality is the following:
- Set the initial signal and the Mesh information
- Create instance of specific compression algorithm
- Recover compressed information
- Translate information to Mesh-Particle information