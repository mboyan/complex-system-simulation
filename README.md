# Catching the Drift:<br/>Modelling Coral Growth with Diffusion Limited Aggregation (DLA)
## _Course Project | Complex System Simulation | MSc Computational Science (UvA/VU)_

This repository contains the Python framework for modelling coral growth using Diffusion Limited Aggregation (DLA)<sup>11</sup>. It was constructed during a two-week university project for the course Complex System Simulation in January 2024.

![background_base](https://github.com/mboyan/complex-system-simulation/blob/boyan-final-cleanup/Images/background_base_cropped.png)


### Introduction
[Corals](https://kids.frontiersin.org/articles/10.3389/frym.2019.00143) are colonial marine organisms of great environmental importance.
Understanding their behaviour, morphogenesis and lifecycle are essential for their preservation.
It has been recognised<sup>2</sup> that their growth can be modelled using DLA - an algorithm which generates branching clusters that can closely mimic various natural phenomena, from urban structures<sup>10</sup> to fluid interface instabilities<sup>7</sup>.

The building blocks of corals are polyps - small tentacled invertebrates that live in a symbiotic relationship with photosynthesising microalgae called zooxanthellae.
The algae hosted in their tissues absorb sunlight to produce most of the energy required for the coral to survive.
The rest is acquired by the polyps by capturing and digesting nutrients carried by the underwater currents, which can range from zooplankton to small fish.
Throughout their lifecycle, corals excrete calcium carbonate, which becomes the main component in coral reefs.
This branching skeleton remains a long-lasting trace of the organisms that once inhabited it, allowing scientists to identify species by the morphological properties of the calcified structures<sup>3</sup>.

As any living system, corals have a complex lifecycle, with a lot of environmental and biological factors determining their growth and proliferation.
Nonetheless, in a highly abstracted depiction of the conditions under which coral skeletons form, it is easy to spot some parallels with the elementary principles of the DLA algorithm.
A diffuse transport of elements required for sustenance (analogical to random walks of particles in a discretised space) ultimately results in the permanent deposition of solid matter (assigning a fixed value to regions of this discrete space) and the formation of dendritic aggregations (the DLA clusters).

Existing research has gone into great depth in incorporating more realistic representations into coral growth models, including the modelling of fluid dynamics<sup>5</sup> or alternative, more sophisticated growth principles<sup>6</sup>. Additionally, various measures have been highlighted for quantifying the emergent morphological properties of actual corals, including fractal dimension<sup>4</sup>, compactness<sup>3</sup> and curvature<sup>9</sup>.

On the other hand, quantitative measures of similar nature have been applied to the phenomenon of DLA as a complex system, analysing its fractal dimension<sup>1</sup> and the distribution of branch lengths<sup>8</sup>. The latter is evidently governed by a power-law, characteristic for many emergent phenomena found in natural systems and in computational models.

### The Model

Given the short time frame of the course, we aimed to produce a toy model which builds up on the elementary implementation of DLA and encompasses many of these aspects in a relatively simple form. The goal was to have a versatile simulation framework which enables a range of inputs - either spatial / topological parameters of the simulation space or special variables which can be seen as analogies to environmental influences. As an output, the framework is meant to provide, on one hand, qualitative visuals of the emerging DLA-made "corals", and on the other, a comparative analysis of the variation in selected morphological measures. This framework was used to address the main research questions posed in this project:
- Are there consistent measures to evaluate the emerging morphological complexity of DLA-generated “coral” growth?
- How do drift, growth bias (preferential growth toward sunlight) and surrounding “nutrition” density influence the outcomes of the growth model?
- How can inhibitory influences (e.g. macroalgae blocking the sunlight, competing species, local spatial constraints) affect the outcomes of the growth model - as variants in the environmental parameterisations?

The code is structured in four .py modules. The DLA model containing all functions defining the movement, aggregation, initialisation and regeneration of particles can be found in `dla_model.py`. The analysis of the fractal dimension and the branch distribution of DLA-generated "corals" are designated functions in `cs_measures.py`. The experimental setups for executing the simulations and analysing the results are defined as functions in `dla_simulation.py`. Matplotlib functions tend to create a bit of a mess, which is why most of the plotting procedures are contained in a separate module, `vis_tools.py`.

A more detailed explanation of the code can be found, together with a documentation of the experimental process, in the [Project Notebook](https://github.com/mboyan/complex-system-simulation/blob/main/Code/ProjectNotebook.ipynb).
A more concise documentation of all functions can be found in the [Documentation](https://github.com/mboyan/complex-system-simulation/tree/main/Documentation) directory.



https://github.com/mboyan/complex-system-simulation/assets/29741948/8cfe2fe6-c0c4-40e4-9753-c148ec806d5b


### Python Module

The framework is also uploaded as a module on PyPi (https://pypi.org/project/dlacorals/). To install it run the following from the command line:

`pip install dlacorals`

### Prerequisites

The code makes use of the following packages:
- _numpy_ (https://numpy.org/);
- _scipy_ (https://scipy.org/);
- _pandas_ (https://pandas.pydata.org/);
- _matplotlib_ (https://matplotlib.org/);
- _seaborn_ (https://seaborn.pydata.org/);
- _networkx_ (https://networkx.org/);
- _powerlaw_ (https://pypi.org/project/powerlaw/).

The relevant versions are indicated in `requirements.txt`.

Additionally, to render the animations it may be necessary to install _ffmpeg_ (https://ffmpeg.org/download.html).


## References
[1] Jiang, Minhui, and Zhouqi Zhong. ‘Research on Fractal Dimension and Growth Control of Diffusion Limited Aggregation’. _International Conference on Statistics, Applied Mathematics, and Computing Science (CSAMCS 2021)_, vol. 12163, SPIE, 2022, pp. 778–86.  
[2] Kaandorp, Jaap A., Christopher P. Lowe, et al. ‘Effect of Nutrient Diffusion and Flow on Coral Morphology’. _Physical Review Letters_, vol. 77, no. 11, 1996, p. 2328.  
[3] Kaandorp, Jaap A., Peter MA Sloot, et al. ‘Morphogenesis of the Branching Reef Coral Madracis Mirabilis’. _Proceedings of the Royal Society B: Biological Sciences_, vol. 272, no. 1559, 2005, pp. 127–33.  
[4] Martin-Garin, Bertrand, et al. ‘Use of Fractal Dimensions to Quantify Coral Shape’. _Coral Reefs_, vol. 26, no. 3, 2007, pp. 541–50.  
[5] Merks, RMH, et al. ‘Diffusion-Limited Aggregation in Laminar Flows’. _International Journal of Modern Physics C_, vol. 14, no. 09, 2003, pp. 1171–82.  
[6] Merks, Roeland, et al. ‘Models of Coral Growth: Spontaneous Branching, Compactification and the Laplacian Growth Assumption’. _Journal of Theoretical Biology_, vol. 224, no. 2, 2003, pp. 153–66.  
[7] Paterson, Lincoln. ‘Diffusion-Limited Aggregation and Two-Fluid Displacements in Porous Media’. Physical Review Letters, vol. 52, no. 18, 1984, p. 1621.  
[8] Pastor-Satorras, Romualdo, and Jorge Wagensberg. ‘Branch Distribution in Diffusion-Limited Aggregation: A Maximum Entropy Approach’. _Physica A: Statistical Mechanics and Its Applications_, vol. 224, no. 3–4, 1996, pp. 463–79.  
[9] Ramírez-Portilla, Catalina, et al. ‘Quantitative Three-Dimensional Morphological Analysis Supports Species Discrimination in Complex-Shaped and Taxonomically Challenging Corals’. _Frontiers in Marine Science_, 2022.  
[10] Rybski, Diego, et al. ‘Modeling Urban Morphology by Unifying Diffusion-Limited Aggregation and Stochastic Gravitation’. Findings, 2021.  
[11] Witten Jr, Thomas A., and Leonard M. Sander. ‘Diffusion-Limited Aggregation, a Kinetic Critical Phenomenon’. _Physical Review Letters_, vol. 47, no. 19, 1981, p. 1400.  

