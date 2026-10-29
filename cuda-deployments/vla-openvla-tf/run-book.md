# Run Book: vla-openvla-tf

This deployment serves openvla/openvla-7b with transformers 5.20.2. One
process serves `GET /kot/info`, `POST /kot/infer` and `POST /kot/score` on
one port. One request runs at a time: one robot, one card. A request names
one observation file. `/kot/infer` predicts one action and writes it as
`actions.npy`, float32, shape `[1, 1, 7]`. `/kot/score` writes the
log-probabilities of the 7 action tokens over the full vocabulary.

The server does not run the checkpoint's remote code. That code needs
transformers 4.40.1. `openvla.py` builds the same network and loads the same
weights; its docstring gives the reason.

## Model parameters

| model | revision | cards | dtype | decoding |
|---|---|---|---|---|
| openvla/openvla-7b | `47a0ec7fc4ec123775a391911046cf33cf9ed83f` | 1 | bfloat16 | greedy, 7 tokens |

One stack serves the model: CPython 3.12, transformers 5.20.2, timm 0.9.16,
torch 2.14.1+cu130, torchvision 0.29.1+cu130. Greedy decoding uses no seed.

## Server: NVIDIA RTX 5090 (sm_120)

### Environment

Build the venv at `/root/kotvenv-vla`. The launcher uses this path.

```bash
uv venv /root/kotvenv-vla --python 3.12
uv pip install --python /root/kotvenv-vla/bin/python \
  torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
uv pip install --python /root/kotvenv-vla/bin/python --no-deps \
  torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
uv pip install --python /root/kotvenv-vla/bin/python \
  transformers==5.20.2 timm==0.9.16 numpy pillow
```

The torch build must carry sm_120; cu130 does. torchvision 0.29.x pairs with
torch 2.14.x. `--no-deps` keeps the torch build. timm builds the two vision
towers and imports torchvision. Use timm 0.9.x only: timm 1.0 changed
`get_intermediate_layers`, and the reference code refuses it.

Verify the stack:

```bash
/root/kotvenv-vla/bin/python -c 'import torch, transformers, timm; \
  print(torch.__version__, transformers.__version__, timm.__version__, \
  torch.cuda.get_device_name(0))'
```

### Layout

```text
/opt/engines/vla-openvla-tf   this directory
/models/hub                   hub cache; the weights are one snapshot in it
```

### Transfer

Push and pull over the provider-mapped ssh port. rsync makes no parent
directories and drops execute bits; `--partial` resumes an interrupted
transfer. The excludes keep `__pycache__` and `.DS_Store` off the box.

```bash
ssh -p $PORT user@$HOST 'mkdir -p /opt/engines/vla-openvla-tf'
rsync -a --partial -e "ssh -p $PORT" \
  --exclude __pycache__ --exclude .DS_Store \
  ./vla-openvla-tf/ user@$HOST:/opt/engines/vla-openvla-tf/
ssh -p $PORT user@$HOST 'chmod +x /opt/engines/vla-openvla-tf/*.sh /opt/engines/vla-openvla-tf/*.py'
rsync -a --partial -e "ssh -p $PORT" user@$HOST:/outputs/ ./outputs/
```

Download the weights on the box into the hub cache. The snapshot directory
carries the revision; `/kot/info` reports it. On vast, huggingface.co times
out, so the download goes through the mirror endpoint:

```bash
HF_ENDPOINT=https://hf-mirror.com /root/kotvenv-vla/bin/hf download \
  openvla/openvla-7b --revision 47a0ec7fc4ec123775a391911046cf33cf9ed83f \
  --cache-dir /models/hub
export MODEL=/models/hub/models--openvla--openvla-7b/snapshots/47a0ec7fc4ec123775a391911046cf33cf9ed83f
ls $MODEL
```

Verify: the directory holds three `model-*-of-00003.safetensors` shards,
`config.json`, `preprocessor_config.json` and the tokenizer files. The
weights take 15.1 GB.

### Start

```bash
cd /opt/engines/vla-openvla-tf
mkdir -p /outputs
export CUDA_VISIBLE_DEVICES=0
./kot_serve.sh --port 8090 --model $MODEL --device cuda 2>&1 | tee /outputs/vla-openvla-tf.log
```

The model needs one card. The server uses the first visible card only, so
set CUDA_VISIBLE_DEVICES to one card. The process listens on 127.0.0.1 only.

The start loads the weights, then runs one warm-up prediction. The port
opens after the warm-up. The startup line `[vla-openvla-tf] serving on ...`
holds the `/kot/info` content. A load error stops the start and names the
tensor or the parameter.

Each request writes one log line. A `/kot/infer` line holds the prompt id,
the prediction time in ms, the 7 tokens and the action. A `/kot/score` line
holds the prompt id, the mode, the time in ms, the row count and the tokens.

### Parameters

| variable | effect |
|---|---|
| `KOT_VLA_PYTHON` | the interpreter the launcher execs; default `/root/kotvenv-vla/bin/python` |
| `CUDA_VISIBLE_DEVICES` | the card set; the server uses card 0 of it |

| argument | effect |
|---|---|
| `--port` | listen port on 127.0.0.1 |
| `--model` | the checkpoint directory, or a hub snapshot directory |
| `--device` | `cuda` puts the model on `cuda:0` |

## Models

### openvla/openvla-7b

The model is a Prismatic VLM: DINOv2 ViT-L/14 and SigLIP SO400M/14 read the
frame, an MLP projects their patches into Llama-2-7B, and Llama-2-7B writes
7 action tokens. Each token is one of 256 bins over [-1, 1]. The dataset
statistics turn the bins into robot units.

The table holds the pinned parameters. A changed value changes the action.

| parameter | value | source |
|---|---|---|
| action statistics | `bridge_orig` (WidowX, BridgeData V2) | `UNNORM_KEY` in serve.py; the model card's example |
| prompt | `In: What action should the robot take to {task}?\nOut:`, task lower-cased, token 29871 appended | openvla `vla-scripts/deploy.py`; `predict_action` |
| image | resize 224x224 bicubic (PIL), /255, mean/std per tower | `preprocessor_config.json`: the `tvf_*` fields |
| camera to 8-bit | round to nearest level, ties to even, saturate at 0 and 255 | serve.py |
| decoding | argmax over all 32064 logits, 7 steps, KV cache | `predict_action` with `do_sample=False` |
| attention | sdpa | the model card uses flash_attention_2 |

The normalisation uses the stored, bf16-rounded means (0.484375, not 0.485).
The reference processor does the same.

The action holds x, y, z, roll, pitch, yaw and gripper. `bridge_orig` maps
the first 6 from [-1, 1] to its q01..q99 range. The gripper stays in [-1, 1].
The action tokens are ids 31744 to 31999. A token outside this range clips
to the nearest bin.

The observation is a kot `.npz`: `prompt`, a string, and one camera, float32
`[3, H, W]` with 1.0 at full scale. The server ignores `state` and `noise`.
Any other key is a camera; two cameras stop the request with an error.

`/kot/score` decode builds the prompt from the observation's task. It writes
7 rows, one per action token, and the ids prompt + tokens. A prefill takes
those ids back with the same observation. It writes the teacher-forced
rows: n - 1, or the last `n_rows`. Every row is the float32 log-softmax of
the raw logits over all 32064 columns. The server reads no seed and no
sampling: decoding is greedy. `max_new_tokens` must be 7.

## Check

The identity surface answers with the engine's own facts:

```bash
curl -s http://127.0.0.1:8090/kot/info | python3 -m json.tool
```

Make an observation from a photo. Any RGB photo serves. A frame from the
robot's third-person camera gives a meaningful action:

```bash
mkdir -p /tmp/kot-vla
/root/kotvenv-vla/bin/python /opt/engines/vla-openvla-tf/make_observation.py \
  frame.jpg "put the eggplant in the basket" /tmp/kot-vla/obs001.npz
```

Send it and read the action:

```bash
curl -s http://127.0.0.1:8090/kot/infer \
  -H 'Content-Type: application/json' \
  -d '{"prompt_id": "obs001", "out_dir": "/tmp/kot-vla/out/obs001",
       "attachments": {"observation": "/tmp/kot-vla/obs001.npz"}}' \
  | python3 -m json.tool
/root/kotvenv-vla/bin/python -c \
  'import numpy as np; a = np.load("/tmp/kot-vla/out/obs001/actions.npy"); print(a.shape, a.dtype, a)'
```

Expect `(1, 1, 7) float32`. The server log shows the 7 tokens; each must lie
in 31744..31999.

Make a second observation from a different photo, with the same task:

```bash
/root/kotvenv-vla/bin/python /opt/engines/vla-openvla-tf/make_observation.py \
  other.jpg "put the eggplant in the basket" /tmp/kot-vla/obs002.npz
```

Score both. The two frames must give different log-probabilities:

```bash
for id in obs001 obs002; do
  curl -s http://127.0.0.1:8090/kot/score -H 'Content-Type: application/json' \
    -d "{\"prompt_id\": \"$id\", \"mode\": \"decode\", \"input_ids\": [],
         \"max_new_tokens\": 7, \"out_dir\": \"/tmp/kot-vla/score/$id\",
         \"attachments\": {\"observation\": \"/tmp/kot-vla/$id.npz\"}}"
  echo
done
/root/kotvenv-vla/bin/python -c 'import numpy as np
a, b = (np.load(f"/tmp/kot-vla/score/{i}/log_probs.npy") for i in ("obs001", "obs002"))
print(a.shape, a.dtype, "max |diff|", abs(a - b).max())'
```

Expect `(7, 32064) float32` and a max |diff| above 0. A difference of 0
means the frame does not reach the model, and every action is wrong.

Replay the decode as a prefill over its stored ids. The rows must give the
same tokens as the decode:

```bash
IDS=$(/root/kotvenv-vla/bin/python -c \
  'import numpy as np; print(np.load("/tmp/kot-vla/score/obs001/input_ids.npy").tolist())')
curl -s http://127.0.0.1:8090/kot/score -H 'Content-Type: application/json' \
  -d "{\"prompt_id\": \"obs001\", \"mode\": \"prefill\", \"input_ids\": $IDS,
       \"n_rows\": 7, \"out_dir\": \"/tmp/kot-vla/replay/obs001\",
       \"attachments\": {\"observation\": \"/tmp/kot-vla/obs001.npz\"}}"
echo
/root/kotvenv-vla/bin/python -c 'import numpy as np
d = np.load("/tmp/kot-vla/score/obs001/log_probs.npy")
p = np.load("/tmp/kot-vla/replay/obs001/log_probs.npy")
print("same tokens", (d.argmax(-1) == p.argmax(-1)).all(), "max |diff|", abs(d - p).max())'
```

Expect `same tokens True` and a small max |diff|: the decode reads a KV
cache, the prefill does not, and the kernels differ.

kot's speed run times `/kot/infer`, one request at a time. The server log
gives the prediction time of each request.
