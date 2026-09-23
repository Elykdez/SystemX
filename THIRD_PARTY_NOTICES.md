# Third-party notices

## Von SDK and OptionMarker model

Upstream project: https://github.com/wfzyx/von

SDK revision: `2656a69be2ef0ebf2f3058302e52791f99cbd8de`.

Model repository: https://huggingface.co/wfzyx/von-1.0

Model revision: `aa2fdc9630ecdadef32c56073553b3a69bed38bf`.

The SDK and model are published upstream under Apache License 2.0. The upstream license text is included verbatim at [licenses/Von-LICENSE.md](licenses/Von-LICENSE.md), copied from the pinned SDK distribution. Model files reside in the ignored `.cache` directory; the SDK is installed in the ignored `.venv` directory. Preserve this attribution and applicable upstream notices when redistributing either.

The local adapter constructs the upstream architecture from configuration, loads the full pinned checkpoint, selects SDPA attention, disables optional ModernBERT compilation, adapts tokenizer loading to Transformers 4, and exposes a bounded HTTP interface. The upstream model weights have not been modified.

Other installed dependencies, including PyTorch, Transformers, FastAPI, and Hugging Face libraries, retain their own license terms and notices in their installed distributions. This notice does not assign a license to the independently maintained server wrapper.
