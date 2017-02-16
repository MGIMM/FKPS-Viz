# FKPS-Viz
Feynman-Kac Particle System Visulization


* run `python particle-system-generator.py` to generate a tree
  structured `.json` file

* run `python -m SimpleHTTPServer 8000` to set up

* got to the page `http://localhost:8000/` in the browser 
 
### particle-system-generator.py

* `N_test` :  number of particles

* `p_0_test` : survive probability

For the visulization, we take `N_test = 100`. As the number of
particles is relatively small, the quality of variance estimator may
be affected.

### `var_estimation.py` 

This file is in the same structure of `particle-system-generator.py`.

run `python var_estimation.py` to test the variance estimator. 
