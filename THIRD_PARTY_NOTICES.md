# Third-party notices

## StartLux v1.1 4B Instant

SystemX uses [StartLux-Decision](https://github.com/StartLuxLabs/StartLux-Decision)
and the [4B Instant checkpoint](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-4B-Instant).
Source and checkpoint revisions are pinned in `model_config.py`. AutoThink is not used.

The model weights are published under CC BY-NC 4.0. Commercial use requires
separate licensing from StartLux. Consult the upstream repositories for the full
code and model license terms before redistribution. Model files are cached in
the ignored `.cache` directory; the original weights are not modified.

The CUDA runtime uses [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention)
(`fla-core`), [Triton](https://github.com/triton-lang/triton-windows), and
[bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes).
Their distributions retain their respective licenses. The int8 adapter derives
its loading sequence from the pinned StartLux Transformers backend and quantizes
the weights in memory; the model's usage license still applies.

Other dependencies retain their own licenses and distribution notices. These
notices do not assign a license to the independently maintained SystemX wrapper.
