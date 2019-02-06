## This is a port of Brian Hargreaves' Matlab EPG simulation scripts.
http://web.stanford.edu/~bah/software/epg/
This repo also includes the torch implementation of epg, which can leverage GPU and matrix computation to accelerate epg simulation. In general, the computation time of a 320 by 320 image is about 0.01 second comparing to about 1 minute using cpu epg without matrix computation.
