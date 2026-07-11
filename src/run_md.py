"""Standalone MD production for alanine dipeptide (implicit solvent, 300 K).
Usage: python run_md.py <seed> <production_ns> <outdir>
Runs with 1 CPU thread (fastest for this 22-atom system). Saves DCD every 1 ps.
"""
import sys, os, time
import numpy as np
from openmm import unit, LangevinMiddleIntegrator, Platform
from openmm.app import Simulation, DCDReporter, StateDataReporter, PDBFile
from openmmtools import testsystems

def main():
    seed = int(sys.argv[1])
    prod_ns = float(sys.argv[2])
    outdir = sys.argv[3]
    os.makedirs(outdir, exist_ok=True)

    dt_fs = 2.0
    save_ps = 1.0
    equil_ns = 0.5
    steps_per_ps = int(1000 / dt_fs)          # 500
    save_interval = int(save_ps * steps_per_ps)  # 500 steps = 1 ps
    equil_steps = int(equil_ns * 1e6 / dt_fs)
    prod_steps = int(prod_ns * 1e6 / dt_fs)

    ts = testsystems.AlanineDipeptideImplicit(hydrogenMass=3.0*unit.amu)
    system, topology, positions = ts.system, ts.topology, ts.positions

    integ = LangevinMiddleIntegrator(300*unit.kelvin, 1.0/unit.picosecond, dt_fs*unit.femtosecond)
    integ.setRandomNumberSeed(seed)
    plat = Platform.getPlatformByName("CPU")
    sim = Simulation(topology, system, integ, plat, {"Threads": "1"})
    sim.context.setPositions(positions)

    # Save topology once (from seed 1 only, but harmless to write per seed)
    with open(os.path.join(outdir, "topology.pdb"), "w") as f:
        PDBFile.writeFile(topology, positions, f)

    sim.minimizeEnergy()
    sim.context.setVelocitiesToTemperature(300*unit.kelvin, seed)

    # Equilibration
    t0 = time.time()
    sim.step(equil_steps)
    print(f"[seed {seed}] equilibration {equil_ns} ns done in {(time.time()-t0)/60:.1f} min", flush=True)

    # Production with reporters
    dcd = os.path.join(outdir, f"seed{seed}.dcd")
    sim.reporters.append(DCDReporter(dcd, save_interval))
    sim.reporters.append(StateDataReporter(
        os.path.join(outdir, f"seed{seed}.log"), save_interval*10,
        step=True, time=True, potentialEnergy=True, temperature=True, speed=True))
    t0 = time.time()
    sim.step(prod_steps)
    wall = (time.time()-t0)/60
    n_frames = prod_steps // save_interval
    print(f"[seed {seed}] production {prod_ns} ns done in {wall:.1f} min, ~{n_frames} frames -> {dcd}", flush=True)

if __name__ == "__main__":
    main()
