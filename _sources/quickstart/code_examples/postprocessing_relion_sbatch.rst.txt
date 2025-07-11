RELION cluster execution
========================

The following code outlines how to use RELION on a SLURM cluster. Make sure to adapt the parameters to your specific use case.

.. code-block:: bash

	#!/bin/bash
	#SBATCH -p gpu-el8
	#SBATCH -J Ref6
	#SBATCH --ntasks=15
	#SBATCH -o run.out
	#SBATCH -e run.err
	#SBATCH --open-mode=append
	#SBATCH --gres=gpu:2
	#SBATCH -C gpu=3090
	#SBATCH -N 1
	#SBATCH --mem=60GB
	#SBATCH --time=1-00:00:00
	#SBATCH --export=NONE
	#SBATCH --ntasks-per-core=1


	module purge
	module load RELION/4.0.1-EMBLv.0011_20230510_01_e16f796_a-foss-2022a-CUDA-11.7.0

	mpirun \
		-n 3 `which relion_refine_mpi` \
		--i ctf_test.star \
		--ref template52.mrc \
		--o ./run_1/ \
		--ctf \
		--auto_refine \
		--split_random_halves \
		--firstiter_cc \
		--ini_high 60 \
		--dont_combine_weights_via_disc \
		--pool 3 \
		--pad 2 \
		--particle_diameter 250 \
		--flatten_solvent \
		--zero_mask \
		--oversampling 1 \
		--healpix_order 2 \
		--auto_local_healpix_order 4 \
		--offset_range 5 \
		--offset_step 2 \
		--sym C1 \
		--low_resol_join_halves 40 \
		--norm \
		--scale \
		--j 7 \
		--gpu

Assuming you copied the code above to a file ``run_relion_refine.sbatch`` you scan submit that to the cluster via:

.. code-block:: bash

	sbatch run_relion_refine.sbatch

