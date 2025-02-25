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
mjx-testspeed --function=forward --is_sparse=True --mjcf=humanoid/humanoid.xml --batch_size=8192
```

Some relevant benchmarks on an NVIDIA GeForce RTX 4090:

## forward steps / sec (smooth dynamics only)

27 dofs per humanoid, 8k batch size.

| Num Humanoids   | MJX    |  mjWarp dense |  mjWarp sparse |
| ----------------| -------| ------------- | -------------- |
| 1               | 7.9M   |  15.6M        | 13.7M          |
| 2               | 2.6M   |  7.4M         | 7.8M           |
| 3               | 2.2M   |  4.6M         | 5.3M           |
| 4               | 1.5M   |  3.3M         | 4.1M           |
| 5               | 1.1M   |  ❌           | 3.2M           |

# Ideas for what to try next

## 1. Unroll steps

In the Pure JAX benchmark, we can tell JAX to unroll some number of FK steps (in the benchmarks above, `unroll=4`).  This has a big impact on performance.  If we change `unroll` from 4 to 1, pure JAX performance at 8k batch drops from 50M to 33M steps/sec.

Is there some way that we can improve Warp performance in the same way?  If I know ahead of time that I am going to call FK in a loop 1000 times, can I somehow inject unroll primitives?

## 2. Different levels of parallelism

The current approach parallelizes over body kinematic tree depth.  We could go either direction: remove body parallism (fewer kernel launches), or parallelize over joints instead (more launches, more parallelism).

## 3. Tiling

It looks like a thing!  Should we use it?  Will it help?

## 4. Quaternions

Why oh why did Warp make quaternions x,y,z,w?  In order to be obstinate I wrote my own quaternion math.  Is this slower than using the Warp quaternion primitives?

## 5. `wp.static`

Haven't tried this at all - curious to see if it helps.

## 6. Other stuff?

Should I be playing with `block_dim`?  Is my method for timing OK or did I misunderstand how `wp.synchronize` works?  Is there something about allocating that I should be aware of?  What am I not thinking of?
