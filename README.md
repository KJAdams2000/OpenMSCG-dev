# OpenMSCG developer version

This contains some of my modifications on the OpenMSCG package (Both in Python and C++ codes) to consider my new UCG projects.
This is by far the latest preserved version when the GitLab link went dead.

Changes mainly include: 
- `mscg.cli.ucg.wf_ld.py`: Using `ucgl` as UCG weights in Lambda Dynamics REM.
- Changes to several trajectory reader files, to allow directly reading `ucgstate`, `ucgl`, `ucgp` from LAMMPS dump file. 
- `Adam` optimizer in `cgrem`.
- Padding options in `cgrem`.
