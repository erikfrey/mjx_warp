# mjWarp

MuJoCo implemented in Warp.

# Installing for development

```bash
git clone https://github.com/erikfrey/mjx_warp.git
cd mjx_warp
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -e .
```

During early development mjWarp is on the bleeding edge - you should install warp nightly:

```
pip install warp-lang --pre --upgrade -f https://pypi.nvidia.com/warp-lang/
```

Now make sure everything is working:

```bash
pytest
```

Should print out something like `XX passed in XX.XXs` at the end!

# Benchmarking

Benchmark as follows:

```bash
mjx-testspeed --function=step --mjcf=humanoid/humanoid.xml --batch_size=8192
```

To get a full trace of the physics steps (e.g. timings of the subcomponents) run the following:

```bash
mjx-testspeed --function=step --mjcf=humanoid/humanoid.xml --batch_size=8192 --event_trace=True
```

`humanoid.xml` has been carefully optimized for MJX in the following ways:

* Newton solver iterations are capped at 1, linesearch iterations capped at 4
* Only foot<>floor collisions are turned on, producing at most 8 contact points
* Adding a damping term in the Euler integrator (which invokes another `factor_m` and `solve_m`) is disabled

By comparing MJWarp to MJX on this model, we are comparing MJWarp to the very best that MJX can do.

For many (most) MuJoCo models, particularly ones that haven't been carefully tuned, MJX will
do much worse.

## physics steps / sec

NVIDIA GeForce RTX 4090, 27 dofs, ncon=8, 8k batch size.

```
Summary for 8192 parallel rollouts

 Total JIT time: 1.25 s
 Total simulation time: 4.46 s
 Total steps per second: 1,838,765
 Total realtime factor: 9,193.83 x
 Total time per step: 543.84 ns

Event trace:

step: 540.35                 (MJX: 316.58 ns)
  forward: 538.04
    fwd_position: 239.87
      kinematics: 12.83      (MJX:  16.45 ns)
      com_pos: 8.02          (MJX:  12.37 ns)
      crb: 12.32             (MJX:  27.91 ns)
      factor_m: 6.50         (MJX:  27.48 ns)
      collision: 197.79      (MJX:   1.23 ns)
      make_constraint: 6.42  (MJX:  42.39 ns)
      transmission: 1.22     (MJX:   3.54 ns)
    fwd_velocity: 29.75
      com_vel: 8.61          (MJX:   9.38 ns)
      passive: 1.12          (MJX:   3.22 ns)
      rne: 14.00             (MJX:  16.75 ns)
    fwd_actuation: 2.76      (MJX:   3.93 ns)
    fwd_acceleration: 11.75
      xfrc_accumulate: 3.76  (MJX:   6.81 ns)
      solve_m: 6.92          (MJX:   8.88 ns)
    solve: 252.87            (MJX: 153.29 ns)
      mul_m: 6.08
      _linesearch_iterative: 40.41
        mul_m: 3.66
  euler: 1.79               (MJX:    3.78 ns)
```
