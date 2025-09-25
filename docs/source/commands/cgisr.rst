Variational optimization of UCG states

Description
-----------

The ``cgisr`` command is used to variationally optimize UCG internal states according to the UCG-ISR method. This command is designed to construct variationally optimal state definitions for one given two states for CG bead type given its self-environment.

Notes
-----

`cgisr` generates a file which contains UCG arguments to employed when calling the ``RLECUSTOM`` class, ``isr.dat``. The ``VAL`` argument should be replaced with the appropriate CG bead type after optimization.

More information on the ISR method can be found in the corresponding `paper <https://doi.org/10.1063/5.0244427>`_.

Examples
--------

::

	cgisr --top top.in \
     	  --traj methanol_interface.lammpstrj \
     	  --traj_class interface_types.lammpstrj \
     	  --names CH3,OH --cut 10.0 --chi 8e-3 \
     	  --epochs 100 \
     	  --batch_size 10 
