.. include:: ../../substitutions.rst

=================
Practical Example
=================

The following outlines potential applications for template matching using ``match_template.py``.

.. toctree::
   :maxdepth: 2

   particle_picking
   fitting
   alignment


You will learn how to analyze the output of these experiments in the Postprocessing section.


Configuration
-------------

The following provides a background on some of the parameters of ``match_template.py``, classified based on their area of influence. An overview of all available parameters can be acquired using

.. code-block:: bash

   match_template.py -h


Input
^^^^^

- **target/template**: Specifies the paths to your target and template files, which are essential inputs for `match_template`.

- **target_mask/template_mask**: Define masks for the target and template, allowing the tool to focus only on regions of interest and potentially improve accuracy.

- **invert_target_contrast**: Whether to invert the target's contrast and perform linear rescaling between zero and one. This option is intended for targets, where the object to-be-matched has negative values, i.e. tomograms with black white contrast.

Performance
^^^^^^^^^^^

- **cutoff_target/cutoff_template**: By defining a minimal enclosing box containing all significant density values, the execution time can be significantly improved.

- **pad_edges**: Pad each dimension of the target by the template shape. This helps to avoid erroneous scores at the boundaries. |project| will automatically set this flag if the target has to be split to fit into memory, as otherwise grid-like artifacts will be present in the scores. Setting this option will typically yield a longer runtime.

- **pad_fourier**: Zero pad the target to the full convolution shape, which is defined as cumulative box size of template and target minus one. Setting this option improves numerical stability, but yields a longer runtime. When working with large data such as tomograms this flag does not need to be set.

- **no_centering**: Omit moving the template's center of mass to the center of a new box that is sufficiently sized to represent all rotations.

- **interpolation_order**: Defines the spline interpolation order for rotations. While higher values ensure more accuracy reducing it can lead to performance improvements, especially on CPU, at a slight accuracy trade-off.

- **use_mixed_precision**: By utilizing float16 for GPU operations, memory consumption is lowered, and certain hardware might observe a performance boost.

- **use_memmap**: When working with large inputs, this option lets the tool offload objects to the disk, making computations feasible even with limited RAM, though this comes with a slight IO overhead.


Accuracy
^^^^^^^^

- **score**: The choice of the scoring function plays a pivotal role in the accuracy of results. Note, that if your mask is not symmetric nor encompasses all possible rotations of the template you have to use a scoring method that also rotates the mask, such FLC or MCC.

- **angular_sampling**: Granularity of rotations. A lower value will sample more rotations, potentially yielding more accurate results but at a computational cost.

- **scramble_phases**: This option scrambles the phases of the templates to simulate a noise background, aiding in refining the score space.


Computation
^^^^^^^^^^^

- **memory_scaling**: Fraction of available memory that will be used. Corresponds to total amount of RAM for CPU backends, and sum of available GPU memory for GPU backends.

- **ram**: Memory in bytes that should be used for computations. If set will overwrite the value determined by **memory_scaling**.

- **use_gpu**: Wheter to perform computations using a GPU backend.

- **gpu_indices**: GPU device indices to use for computation, e.g. 0,1. If this parameter is provided, the shell environment CUDA_VISIBLE_DEVICES that otherwise specifies the GPU device indices will be ignored.
