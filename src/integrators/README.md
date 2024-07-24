# integrators
## Summary
This class contains all information on the specific time integrators used for the Liouville PDE solution.

## Usage and properties
The use case of this class is the following:
- Gather initial position of particles.
- Forward-advect particles (from $$t_0$$ to $$t_0 + \Delta t$$).
- Inverse-advect particles using the composition of the vector field function ($$\mathbf{v}_t$$) with a smooth function in time.
- Inverse-integrate the *rescaling* term of the PDE in lagrangian form.

## Data requirements
The data/simulation information needed to successfully launch would be:
- Initial time and timestep (fixed, for now)