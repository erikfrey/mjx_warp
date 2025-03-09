# MJWarp

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

During early development MJWarp is on the bleeding edge - you should install warp nightly:

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
mjw-testspeed --function=step --mjcf=humanoid/humanoid.xml --batch_size=8192
```

To get a full trace of the physics steps (e.g. timings of the subcomponents) run the following:

```bash
mjw-testspeed --function=step --mjcf=humanoid/humanoid.xml --batch_size=8192 --event_trace=True
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

 Total JIT time: 0.82 s
 Total simulation time: 2.98 s
 Total steps per second: 2,753,173
 Total realtime factor: 13,765.87 x
 Total time per step: 363.22 ns

Event trace:

step: 361.41                 (MJX: 316.58 ns)
  forward: 359.15
    fwd_position: 52.58
      kinematics: 19.36      (MJX:  16.45 ns)
      com_pos: 7.80          (MJX:  12.37 ns)
      crb: 12.44             (MJX:  27.91 ns)
      factor_m: 6.40         (MJX:  27.48 ns)
      collision: 4.07        (MJX:   1.23 ns)
      make_constraint: 6.32  (MJX:  42.39 ns)
      transmission: 1.30     (MJX:   3.54 ns)
    fwd_velocity: 26.52
      com_vel: 8.44          (MJX:   9.38 ns)
      passive: 1.06          (MJX:   3.22 ns)
      rne: 10.96             (MJX:  16.75 ns)
    fwd_actuation: 2.74      (MJX:   3.93 ns)
    fwd_acceleration: 11.90
      xfrc_accumulate: 3.83  (MJX:   6.81 ns)
      solve_m: 6.92          (MJX:   8.88 ns)
    solve: 264.38            (MJX: 153.29 ns)
      mul_m: 5.93
      _linesearch_iterative: 43.15
        mul_m: 3.66
  euler: 1.74               (MJX:    3.78 ns)
```
