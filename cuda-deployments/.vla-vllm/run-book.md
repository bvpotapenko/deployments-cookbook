# Run Book: vla-vllm

This deployment serves OpenVLA-7B with vLLM 0.33.0. It uses the vLLM class
`OpenVLAForActionPrediction`. One process serves `GET /kot/info`,
`POST /kot/infer` and `POST /kot/score` on one port. There is no `/v1` route,
because the `vla_tokens` speed run times `/kot/infer`. The full-vocabulary tap is
version-pinned, so the server refuses to start on any other vllm version.

## Model parameters

The table holds the launch parameters per model. Every value not named stays
at the deployment default. Read a model's row against the same row in the
other engine run books. A mismatch between engines is the first suspect when
a comparison fails.

| model | revision | cards | `KOT_VLLM_GPU_MEM` | `KOT_VLLM_MAX_BATCHED_TOKENS` | `KOT_VLLM_MAX_MODEL_LEN` | dtype | decoding |
|---|---|---|---|---|---|---|---|
| openvla/openvla-7b | `47a0ec7fc4ec123775a391911046cf33cf9ed83f` | 1 | 0.6 | 2048 | unset | bfloat16 | greedy, 7 tokens |

One stack serves every model: CPython 3.12, vLLM 0.33.0, the torch build that
carries sm_120, the transformers line the vLLM 0.33.0 resolver takes, and
timm 0.9.16. Greedy decoding uses no seed.

## Server: NVIDIA RTX 5090 (sm_120)

### Environment

The stack runs on CPython 3.12.

```bash
# networked host
uv venv /root/kotvenv-vllm-vla --python 3.12
uv pip install --python /root/kotvenv-vllm-vla/bin/python "vllm==0.33.0" "timm==0.9.16"

# offline host: build the closure, then install from it
./make_wheels.sh
uv venv --seed /scratch/kot-vllm-vla-venv --python 3.12
/scratch/kot-vllm-vla-venv/bin/pip install --no-index --no-cache-dir \
  --find-links ./wheels "vllm==0.33.0" "timm==0.9.16"
```

The torch build must carry sm_120. The vLLM 0.33.0 resolver pairs one on its
own. A torch without Blackwell support dies at the first request on these
cards.

vLLM does not install timm. vLLM needs timm to build the vision towers. The
server and the launcher refuse any timm other than 0.9.16.

`make_wheels.sh` builds the closure into `wheels/` for linux x86_64, cp312. The
exact pins live in `wheels-requirements.txt`: the same closure as `llm-vllm`,
plus timm. The script downloads one wheel per pin, eight in parallel, each with
`--no-deps`. Files already present are skipped, so a rerun resumes. Rerun the
script until the two counts it prints match. The repository does not carry
`wheels/`.

Without `KOT_PYTHON`, and with no ambient vllm, the launcher runs the offline
install itself, into `KOT_VENV`.

### Layout

```text
/opt/engines/vla-vllm     this directory
  wheels/                 the offline closure, built by make_wheels.sh, not committed
/models/hub               hub cache; the weights are one snapshot in it
```

### Transfer

Push and pull over the provider-mapped ssh port. rsync makes no parent
directories and drops execute bits; `--partial` resumes an interrupted
transfer. The excludes keep `__pycache__` and `.DS_Store` off the box.

```bash
ssh -p $PORT user@$HOST 'mkdir -p /opt/engines/vla-vllm'
rsync -a --partial -e "ssh -p $PORT" \
  --exclude __pycache__ --exclude .DS_Store \
  ./.vla-vllm/ user@$HOST:/opt/engines/vla-vllm/
ssh -p $PORT user@$HOST 'chmod +x /opt/engines/vla-vllm/*.sh'
rsync -a --partial -e "ssh -p $PORT" user@$HOST:/outputs/ ./outputs/
```

Download the weights on the box into the hub cache. The snapshot directory
carries the revision. `/kot/info` reports it. On vast, huggingface.co times
out, so the download goes through the mirror endpoint:

```bash
HF_ENDPOINT=https://hf-mirror.com /root/kotvenv-vllm-vla/bin/hf download \
  openvla/openvla-7b --revision 47a0ec7fc4ec123775a391911046cf33cf9ed83f \
  --cache-dir /models/hub
export MODEL=/models/hub/models--openvla--openvla-7b/snapshots/47a0ec7fc4ec123775a391911046cf33cf9ed83f
ls $MODEL
```

Verify: the directory holds three `model-*-of-00003.safetensors` shards,
`config.json`, `preprocessor_config.json` and the tokenizer files. The weights
take 15.1 GB. Do not change any file in the snapshot.

### Start

```bash
cd /opt/engines/vla-vllm
mkdir -p /outputs
CUDA_VISIBLE_DEVICES=0 KOT_PYTHON=/root/kotvenv-vllm-vla/bin/python \
  ./kot_serve.sh --port 8090 --model $MODEL --device cuda 2>&1 | tee /outputs/vla-vllm.log
```

The engine opens tensor parallel over every visible card. This model fits on
one card. Pin one with `CUDA_VISIBLE_DEVICES`. The process listens on 127.0.0.1
only.

The start loads the weights, then runs one warm-up decode. The port opens after
the warm-up. The startup line `[kot-vllm-vla] serving on ...` holds the
`/kot/info` content.

The server exits non-zero when the load or the warm-up fails. The server also
exits non-zero when `--device` is not the platform that vLLM detects.

Each request writes one log line. The line holds the prompt id, the route or
the mode, the time in ms and the tokens.

The launcher raises the file-descriptor limit to the hard cap itself.

Prefix caching is disabled in the server. A cache hit leaves the requested rows
of the cached prefix not computed.

The multimodal processor cache is disabled in the server. With both caches off,
vLLM names each image per request, so the vision towers run for every request.

### Parameters

| variable | default | effect |
|---|---|---|
| `KOT_PYTHON` | ambient `python3` | interpreter that carries vllm 0.33.0 and timm 0.9.16. The launcher's search never finds `/root/kotvenv-vllm-vla`; set the variable. |
| `KOT_VENV` | `/scratch/kot-vllm-vla-venv` | where the launcher builds the offline venv; it falls back to `/tmp/kot-vllm-vla-venv`. |
| `KOT_VLLM_GPU_MEM` | `0.6` | `gpu_memory_utilization`. The value includes the 14 GiB of weights. On one card, 0.6 gives about 4.5 GiB of KV cache, and a value below 0.45 gives none. |
| `KOT_VLLM_MAX_BATCHED_TOKENS` | `2048` | caps the per-step transient of the log-probability pass. |
| `KOT_VLLM_MAX_MODEL_LEN` | the checkpoint's 2048 | caps `max_model_len`. One request at 2048 positions needs about 1 GiB of KV cache. |
| `KOT_VLLM_DTYPE` | engine auto | compute dtype, for example `float32` for a cross-check. |
| `KOT_VLLM_COMPILATION` | engine default | `eager` disables torch.compile. |

## Models

### openvla/openvla-7b

The model is a Prismatic VLM. DINOv2 ViT-L/14 and SigLIP SO400M/14 read the
frame. An MLP projects their patches into Llama-2-7B. Llama-2-7B writes 7 action
tokens. The dataset statistics change the tokens into robot units.

vLLM 0.33.0 has its own config and processor classes for `model_type: openvla`.
`trust_remote_code` stays False. The server does not import the checkpoint's
Python.

The table holds the pinned parameters. A changed value changes the action.

| parameter | value | source |
|---|---|---|
| action statistics | `bridge_orig` (WidowX, BridgeData V2) | `UNNORM_KEY` in `kot_serve.py`; the model card's example |
| prompt | `In: What action should the robot take to {task}?\nOut:`, task lower-cased, `<s>` first, token 29871 appended | openvla `vla-scripts/deploy.py`; `predict_action` |
| image slot | 256 tokens after `<s>`, id 32000 in the expanded prompt | vLLM's processor |
| image | resize 224x224 bicubic (PIL), /255, mean/std per tower, cast to bfloat16 | vLLM's processor; the `tvf_*` fields of `preprocessor_config.json` |
| camera to 8-bit | round to nearest level, saturate at 0 and 255 | `kot_serve.py` |
| decoding | argmax over all 32064 logits, 7 steps, no EOS stop | `predict_action` with `do_sample=False` |
| vocabulary | 32064 columns | `text_config.vocab_size` |

The action holds x, y, z, roll, pitch, yaw and gripper. `bridge_orig` maps the
first 6 from [-1, 1] to its q01..q99 range. The gripper stays in [-1, 1]. The
action tokens are ids 31744 to 31999. The checkpoint has 255 bin centres. A token
outside the centres clips to the nearest one.

The observation is a kot `.npz`. It holds `prompt`, a string, and one camera,
float32 `[3, H, W]` with 1.0 at full scale. The server ignores `state` and
`noise`. Any other key is a camera. Two cameras stop the request with an error.

`/kot/score` decode builds the prompt from the observation's task when
`input_ids` is empty. It writes 7 rows, one per action token, and the ids
prompt + tokens. `max_new_tokens` must be 7. The server reads no seed and no
sampling.

`/kot/score` prefill takes those ids back with the same observation. The ids
must start with `<s>` and hold no image tokens. It writes the last `n_rows`, or
n - 1 rows. Every row is the float32 log-softmax of the raw logits over all 32064
columns.

## Check

Make two observations from two photos, with one task. A frame from the robot's
third-person camera gives a meaningful action:

```bash
mkdir -p /tmp/kot-vla
for pair in "frame.jpg obs001" "other.jpg obs002"; do
  set -- $pair
  /root/kotvenv-vllm-vla/bin/python - "$1" /tmp/kot-vla/$2.npz <<'EOF'
import sys
import numpy as np
from PIL import Image
photo, out = sys.argv[1:]
pixels = np.asarray(Image.open(photo).convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255
np.savez(out, prompt=np.asarray("put the eggplant in the basket"), image=pixels)
EOF
done
```

Read the identity:

```bash
curl -s http://127.0.0.1:8090/kot/info | python3 -m json.tool
```

Expect `model_id` `openvla/openvla-7b`, the pinned `model_revision`, `dtype`
`bfloat16`, `device` `cuda:0`, `vocab_size` 32064 and `unnorm_key` `bridge_orig`.

Send one observation and read the action:

```bash
curl -s http://127.0.0.1:8090/kot/infer \
  -H 'Content-Type: application/json' \
  -d '{"prompt_id": "obs001", "out_dir": "/tmp/kot-vla/out/obs001",
       "attachments": {"observation": "/tmp/kot-vla/obs001.npz"}}' \
  | python3 -m json.tool
/root/kotvenv-vllm-vla/bin/python -c \
  'import numpy as np; a = np.load("/tmp/kot-vla/out/obs001/actions.npy"); print(a.shape, a.dtype, a)'
```

Expect `(1, 1, 7) float32`. The server log shows the 7 tokens. Each token must lie
in 31744..31999.

Score both observations. The two frames must give different log-probabilities:

```bash
for id in obs001 obs002; do
  curl -s http://127.0.0.1:8090/kot/score -H 'Content-Type: application/json' \
    -d "{\"prompt_id\": \"$id\", \"mode\": \"decode\", \"input_ids\": [],
         \"max_new_tokens\": 7, \"out_dir\": \"/tmp/kot-vla/score/$id\",
         \"attachments\": {\"observation\": \"/tmp/kot-vla/$id.npz\"}}"
done
/root/kotvenv-vllm-vla/bin/python -c 'import numpy as np
a, b = (np.load(f"/tmp/kot-vla/score/{i}/log_probs.npy") for i in ("obs001", "obs002"))
print(a.shape, a.dtype, "max |diff|", abs(a - b).max())'
```

Expect `(7, 32064) float32` and a max |diff| above 0. A difference of 0 means the
frame does not reach the model.

Replay the decode as a prefill over its stored ids:

```bash
IDS=$(/root/kotvenv-vllm-vla/bin/python -c \
  'import numpy as np; print(np.load("/tmp/kot-vla/score/obs001/input_ids.npy").tolist())')
curl -s http://127.0.0.1:8090/kot/score -H 'Content-Type: application/json' \
  -d "{\"prompt_id\": \"obs001\", \"mode\": \"prefill\", \"input_ids\": $IDS,
       \"n_rows\": 7, \"out_dir\": \"/tmp/kot-vla/replay/obs001\",
       \"attachments\": {\"observation\": \"/tmp/kot-vla/obs001.npz\"}}"
/root/kotvenv-vllm-vla/bin/python -c 'import numpy as np
d = np.load("/tmp/kot-vla/score/obs001/log_probs.npy")
p = np.load("/tmp/kot-vla/replay/obs001/log_probs.npy")
print("same tokens", (d.argmax(-1) == p.argmax(-1)).all(), "max |diff|", abs(d - p).max())'
```

Expect `same tokens True` and a small max |diff|. The decode reads a KV cache. The
prefill does not.

Then check these items:

- The log line `Available KV cache memory` shows about 4.5 GiB.
- A kot capture against `goldens/openvla-7b-rtx5090-transformers` meets its gates.
- With `CUDA_VISIBLE_DEVICES=0,1`, `/kot/info` reports `cuda:0+cuda:1`. The rows stay
  close to the one-card rows.
- One observation sent twice takes about the same `/kot/infer` time both times.
