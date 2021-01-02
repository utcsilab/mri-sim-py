## EPG Algorithm
The Extended Phase Graph Algorithm is a powerful tool for MRI sequence simulation and quantitative fitting, but such simulators are mostly written to run on CPU only and (with some exception) are poorly parallelized. A parallelized simulator compatible with other learning-based frameworks would be a useful tool to optimize scan parameters. Thus, we created an open source, GPU-accelerated EPG simulator in PyTorch. Since the simulator is fully differentiable by means of automatic differentiation, it can be used to take derivatives with respect to sequence parameters, e.g. flip angles, as well as tissue parameters, e.g. T1 and T2.  

## This is a port of Brian Hargreaves' Matlab EPG simulation scripts.
http://web.stanford.edu/~bah/software/epg/
This repo also includes the torch implementation of epg, which can leverage GPU and matrix computation to accelerate epg simulation. In general, the computation time of a 320 by 320 image is about 0.01 second comparing to about 1 minute using cpu epg without matrix computation.

## This is a port of Brian Hargreaves' Matlab EPG simulation scripts.
http://web.stanford.edu/~bah/software/epg/
This repo also includes the torch implementation of epg, which can leverage GPU and matrix computation to accelerate epg simulation. In general, the computation time of a 320 by 320 image is about 0.01 second comparing to about 1 minute using cpu epg without matrix computation.
