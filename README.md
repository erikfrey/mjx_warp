# mjx_warp

Trying to learn Warp - this is MuJoCo's FK implemented in Warp.

# Installing for development

```bash
git clone https://github.com/erikfrey/mjx_warp.git
cd mjx_warp
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -e .
```

Now make sure everything is working:

```bash
pytest
```

Should print out something like `1 passed` at the end!

# Benchmarking

Benchmark as follows:

```bash
mjx-testspeed --mjcf=humanoid/humanoid.xml
```

Some relevant benchmarks on an NVIDIA GeForce RTX 4090:

## kinematics

| Implementation   | Batch Size |  Steps / Second |
| ---------------- | ---------- | --------------- |
| JAX<>CUDA FFI    | 4096       |  46M            |
| Pure JAX         | 4096       |  47M            |
| Warp             | 4096       |  79M            |
| Pure CUDA        | 4096       |  86M            |
| JAX<>CUDA FFI    | 8192       |  68M            |
| Pure JAX         | 8192       |  82M            |
| Taichi           | 8192       |  82M            |
| Warp             | 8192       |  107M           |
| Pure CUDA        | 8192       |  125M           |

## com_pos

```bash
mjx-testspeed --function=com_pos --mjcf=humanoid/humanoid.xml --batch_size=8192
```

| Implementation   | Batch Size |  Steps / Second |
| ---------------- | ---------- | --------------- |
| JAX<>CUDA FFI    | 8192       |  65M            |
| Pure CUDA        | 8192       |  87M            |
| Pure JAX         | 8192       |  94M            |
| Warp             | 8192       |  124M           |

## crb

```bash
mjx-testspeed --function=crb --mjcf=humanoid/humanoid.xml --batch_size=8192
```

| Implementation   | Batch Size |  Steps / Second |
| ---------------- | ---------- | --------------- |
| JAX<>CUDA FFI    | 8192       |  26M            |
| Pure CUDA        | 8192       |  67M            |
| Pure JAX         | 8192       |  37M            |
| Warp             | 8192       |  95M           |

## factor_m

```bash
mjx-testspeed --function=factor_m --mjcf=humanoid/humanoid.xml --batch_size=8192
```

| Implementation   | Format | Batch Size |  Steps / Second |
| ---------------- | ------ | ---------- | --------------- |
| Pure JAX         | Sparse | 8192       |  17M            |
| Warp             | Sparse | 8192       |  33M            |
| Pure JAX         | Dense  | 8192       |  41M            |
| CUDA (cuSOLVER)  | Dense  | 8192       |  50M            |

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
