import pytest
from mscg.cli import cgisr

def test_isr(datafile):
    
    phi, alpha, psi, beta = cgisr.main(
        top    = datafile("methanol_ucg.data"),
        traj   = datafile("UCG.lammpstrj"),
        traj_class   = datafile("UCG.lammpstrj"),
        alpha  = 2.5,
        cut    = 12.0,
        beta   = 0.1,
        psi    = 0.1,
        phi    = 3.0,
        epochs = 10,
        chi    = 8e-3,
        batch_size = 1728,
        save   = 'return'
    )

    vals = [phi, alpha, psi, beta]
    ref_vals = [3.15, 2.33, 0.25, 0.25]

    for i in range(len(vals)):
        assert abs(vals[i] - ref_vals[i]) < 0.05

    
