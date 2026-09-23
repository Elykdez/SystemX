# Decision-model acceleration and int8 validation

Model configuration for these measurements: `StartLux-Decision-v1.1-4B-Instant`,
revision `416088f77cfce09633906a3675c7cd2e5dabcf10`.

The CUDA default is now int8 weights with FP16 computation, accelerated by
`fla-core` 0.5.2 and Triton 3.5.1 (Windows wheel 3.5.1.post24), using
bitsandbytes 0.50.2. Model/revision and Instant/no-thinking behavior are unchanged.
The upstream reference decoder otherwise remains in use. The unquantized BF16
configuration is available with `start.bat -Quantization none`.

## Measured speed

RTX 4090, Windows, PyTorch 2.9.0+cu126, Transformers 5.8.1. Replayed the same
26 captured Minesweeper states and candidate sets three times (78 requests per
configuration), excluding warmup. These are inference tests, not live game moves.

| Configuration | Median model time |
| --- | ---: |
| Original BF16, plain PyTorch linear attention | 219.26 ms |
| BF16 with FLA linear attention | 58.84 ms |
| Int8 + FP16 with FLA, three workers loaded | 71.12 ms |

The deployed int8 configuration is about 3.1 times faster than the original
configuration on this workload. FLA removes the expensive plain PyTorch
Gated DeltaNet fallback. The separate full-attention SDPA math fallback remains
enabled on Windows; the convolution also retains its PyTorch implementation.
First-run Triton compilation adds startup time. More workers do not make an
individual request faster.

## Memory and concurrency

One int8/FP16 worker added approximately 5.3 GiB to whole-GPU usage on the replay
and 6.1 GiB for a 1,956-token request, measured relative to the pre-load desktop
baseline. This includes CUDA overhead and allocator reservations. An additional
test passed startup and a full 2,048-token request with a 6 GiB PyTorch allocator
limit, leaving context overhead outside it. Startup peak reserved memory was
5,266 MiB; peak reserved memory including inference was 5,898 MiB.

Three workers passed five concurrent full-limit requests (three 2,048-token
questions per request). Peak sampled whole-GPU usage was 20,793 MiB, including
other GPU applications. All three workers answered, generated zero tokens, and returned
the expected choice. Warm request latency was approximately 565-569 ms for all
three long questions together. These samples are not a universal worst-case bound.

The RTX 3070 uses one worker by default. The memory measurements support an
8 GB deployment with room reserved for other applications, but were made on a
4090. Actual 3070 latency and combined game/desktop usage remain untested.
CUDA startup defaults to three workers on 24 GB cards and one on smaller cards;
explicit overrides are capped at three. BF16 retains the previous two-worker
default on 24 GB cards. CPU remains single-worker FP32.

A separate alternating serial/concurrent replay (78 requests each) measured
11.85 decisions/second serially and 12.86 with three concurrent workers: only
1.085 times the throughput. Median request latency increased from 66.39 ms to
166.19 ms under concurrency. Three replicas share the GPU and do not promise
three times the throughput. Minesweeper's `start-server.bat` therefore selects
one worker; game moves remain sequential regardless of server worker count.

## Correctness and limits

Int8 is not numerically equivalent to BF16. The initial bitsandbytes outlier
threshold of 6 failed upstream padding invariance, so it was rejected. The
deployed threshold is zero, with all upstream checks retained: causal-loop
equivalence, padding invariance, and late-token reach all pass. No attention
checks were relaxed. The API still enforces the 2,048-token per-question limit.

With FLA alone, 24 of 26 replay states selected the same move as the original
fallback; maximum option-probability change was 0.0077. Int8/FP16 matched FLA
BF16's top move on 15 of 26 states; maximum probability change was 0.0458.
Minesweeper offers multiple proved-safe candidates, so different rankings are
not by themselves failures, but this is not evidence of unchanged general
decision accuracy. Task-specific accuracy testing remains necessary. Choice,
Noul, and Score smoke tests passed, including state-sensitive routing.
The complete regression suite passed: 1,089 tests, with one upstream
Starlette/AnyIO deprecation warning. Dependency and whitespace checks passed.
Subsequent user-requested live tests also passed, as described below.

Local raw reports and replay inputs are under `.cache/startlux-speed/`.

## Live Minesweeper follow-up

Replay inference speed did not translate directly into live move speed. The
input loop included a fixed 350 ms delay after parking the pointer. That delay
is now removed: existing visual postcondition checks wait for two stable frames
and retry transient cursor/animation occlusion without repeating the click.
The pointer parks beyond the nearest available board edge. The three 50 ms
movement/button-edge waits remain for frame-polled game input.

| Live run | Verified moves | Result | Elapsed | Median move time |
| --- | ---: | --- | ---: | ---: |
| Before input-loop change | 30 | Won | 26.969 s | 878.63 ms |
| After, first fresh board | 27 | Won | 14.828 s | 575.81 ms |
| After, second fresh board | 27 | Won | 15.125 s | 576.79 ms |

These were different boards, not a controlled same-board speedup benchmark.
The first optimized game's visible timer showed 13 seconds. Input dispatch
dropped from roughly 538 ms to 183 ms; visual verification still gates every
subsequent move. Reports are under
`.cache/minesweeper/startlux-validation/int8-{live1,fast1,fast2}/`.

Live inference still alternates between roughly 70-90 ms and 240 ms. Replaying
the first optimized run's requests measured an 81.69 ms median. Slow calls also
occur without `--record-vram`, so polling alone does not explain them. Sampled
GPU clocks vary but do not consistently distinguish slow from fast calls.
The cause of that remaining latency variation is unresolved; the live results
do not support promising 70 ms per move or threefold gameplay throughput.

## Running on the 3070

Run `setup.bat` once after updating, then:

```powershell
.\start.bat -Device cuda -Quantization int8 -Workers 1
```

`/health` reports `quantization: int8` and one ready worker. Keep the game and
other GPU applications within the remaining memory. The cached BF16 files are
still needed on disk; quantization happens during loading.

References: [upstream reference backend](https://github.com/StartLuxLabs/StartLux-Decision/blob/cb1d11f1ac9e9b43b95ef3a6468fa29b607c40fd/startlux_decide/backends/transformers.py),
[Triton Windows compatibility](https://github.com/triton-lang/triton-windows),
[bitsandbytes installation](https://huggingface.co/docs/bitsandbytes/main/en/installation).
