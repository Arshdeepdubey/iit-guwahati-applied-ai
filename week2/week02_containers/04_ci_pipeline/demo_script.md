# Week 2 · Session 4 — On-Camera Demo Script
## Schedulers & GPU Allocation — Lab 1 + Lab 2

---

## Before you hit Record

Open **three PowerShell / WSL terminals** side by side:
- **T1** — commands you type for the audience
- **T2** — live GPU metrics (`dmon`)
- **T3** — (optional) per-process accounting (`pmon`)

Copy `busy_train.py` to your demos folder:
```powershell
# On Windows host — copy the file to your demos folder
copy busy_train.py C:\Users\chick\Desktop\demos\
```

---

## Lab 1 — Inspect MIG partitions (Slide 8, ~8 min)

> **On-camera intro (say this):**
> "The first thing any admin does when setting up a shared GPU cluster is check
>  whether MIG is available. Let's run the exact command we have on the slide."

### T1 — Step by step

```powershell
# Step 1 — Is MIG enabled on this GPU?
nvidia-smi -i 0 --query-gpu=mig.mode.current --format=csv
```

**Expected output on RTX 3050 Ti:**
```
mig.mode.current
[N/A]
```

> **Say:** "You will see [N/A] — that is not an error. MIG is a data-centre-only
>  feature, available on A100, H100, and the Blackwell B200 we talk about in the
>  slide. Your laptop GPU correctly reports it has no MIG capability.
>  If you were on an A100 you would see 'Disabled' here, which you could then
>  enable with `sudo nvidia-smi -i 0 --mig-mode=1 -pm 1`."

```powershell
# Step 2 — What profiles would be available on a MIG-capable GPU?
#           (This command is safe to run; it will error cleanly on consumer GPUs)
nvidia-smi mig -lgip
```

**Expected output on RTX 3050 Ti:**
```
Unable to determine the device handle for GPU 00000000:01:00.0:
Not Supported
```

> **Say:** "Again — expected. Flip to the slide: those partition profiles
>  (1g.23gb × 7, 2g.45gb × 3 …) are what you would see on a B200.
>  Each slice behaves like a smaller, fully isolated GPU — same driver,
>  same CUDA, but its own memory pool and SM allocation.
>  A scheduler targets it by UUID just like a whole GPU."

```powershell
# Step 3 — Show what nvidia-smi -L reports (works everywhere)
nvidia-smi -L
```

**Expected output:**
```
GPU 0: NVIDIA GeForce RTX 3050 Ti Laptop GPU (UUID: GPU-xxxxxxxx-...)
```

> **Say:** "On a MIG system, each slice would appear as a separate line here —
>  GPU 0: MIG 2g.45gb Device 0 — and the scheduler would hand out those UUIDs
>  to containers via CUDA_VISIBLE_DEVICES. The container never knows it's a slice;
>  it just sees a smaller GPU."

**→ Transition:** "Now let's do something more visceral — watch a GPU actually work."

---

## Lab 2 — Watch a GPU work (Slide 11, ~8 min)

> **On-camera intro:**
> "We're going to run a small training loop in our Week 2 container and watch
>  the GPU metrics change in real time. This shows you what 'utilisation' actually
>  means — and then we'll deliberately starve it."

### Setup — copy busy_train.py into the container

```powershell
# T1 — confirm the file is in your demos folder
ls C:\Users\chick\Desktop\demos\busy_train.py
```

### Part A — Well-fed training (workers=4)

**T2 — start the metrics watcher FIRST:**
```powershell
# T2 — live GPU metrics, refresh every 1 second
nvidia-smi dmon -s um -d 1
```

You will see a header and then a new row every second:
```
# gpu   sm  mem  enc  dec   fb    bar1
# Idx    %    %    %    %   MiB   MiB
    0    3    1    0    0   378      4
```
> **Say:** "sm = shader multiprocessor utilisation — how busy the compute units are.
>  mem = memory bandwidth utilisation. fb = framebuffer used in MiB.
>  Right now it's idle. Watch what happens when I start training."

**T1 — run the training loop:**
```powershell
docker run --rm --gpus all \
  -v C:\Users\chick\Desktop\demos:/demos \
  gpu-hello \
  python3 /demos/busy_train.py
```

**Expected dmon output (T2) after a few seconds:**
```
    0   96   64    0    0  2048      4
    0   97   63    0    0  2048      4
```

> **Say:** "sm% is 96–97. The GPU is nearly fully occupied. fb shows ~2 GB of model
>  and activation memory. This is healthy training — compute-bound, not waiting."

**Optional — T3 per-process accounting (open a third terminal):**
```powershell
nvidia-smi pmon -c 5
```

```
# gpu        pid  type    sm   mem   enc   dec   command
    0      12345    C     96    64     0     0   python3
```

> **Say:** "pmon shows us which PID owns that utilisation. In a shared cluster,
>  this is how you prove your job is actually running — or catch someone else's
>  runaway process eating your allocation."

---

### Part B — Starved training (the important part)

**T1 — stop the current container (Ctrl-C), then rerun with --starve:**
```powershell
docker run --rm --gpus all \
  -v C:\Users\chick\Desktop\demos:/demos \
  gpu-hello \
  python3 /demos/busy_train.py --starve
```

**Expected dmon output (T2):**
```
    0    8    2    0    0  2048      4
    0    9    1    0    0  2048      4
```

> **Say:** "sm% collapsed from 97 to 8. The model and batch size are identical.
>  The only change: DataLoader workers dropped from 4 to 0, so the main process
>  has to fetch and preprocess each batch on the CPU before the GPU can touch it.
>  The GPU is sitting idle 90% of the time, waiting on data.
>  That dip is money. On a cloud instance at £3/hour, you're burning £2.70/hour
>  to move tensors slowly. This is the most common waste pattern in production ML."

> **Transition to slide 10 (Cost-Efficiency):**
> "This is exactly the 'data-loading bottleneck leaving SMs starved' waste pattern
>  on the slide. The fix: more DataLoader workers, prefetching, or caching
>  preprocessed tensors — topics we'll revisit in Week 8."

---

## Wrap (Slide 12 → 13)

Back to slides for the capstone exercise and week summary. No more typing needed.

> **Close:** "You've now run two real diagnostics — MIG availability and live
>  utilisation measurement. These two commands — dmon and pmon — are the first
>  things any infra engineer runs when someone says 'the GPU is slow'.
>  Next week: Kubernetes, and how these same scheduling principles translate
>  into declarative YAML."

---

## Quick-reference cheat sheet

| Command | What it tells you |
|---|---|
| `nvidia-smi -i 0 --query-gpu=mig.mode.current --format=csv` | MIG enabled? (N/A = consumer GPU) |
| `nvidia-smi mig -lgip` | Available MIG profiles on this GPU |
| `nvidia-smi -L` | All GPUs / MIG instances with UUIDs |
| `nvidia-smi dmon -s um -d 1` | Live SM% + memory BW% per second |
| `nvidia-smi pmon -c 5` | Per-process GPU utilisation (5 samples) |
| `nvidia-smi` (plain) | Snapshot: driver, memory, processes |

