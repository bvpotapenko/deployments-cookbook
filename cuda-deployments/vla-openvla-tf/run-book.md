# Run Book: vla-openvla-tf

This deployment serves openvla/openvla-7b with transformers 5.22.1. One
process serves `GET /info`, `POST /infer` and `POST /score` on
one port. One request runs at a time: one robot, one card. A request names
one observation file. `/infer` predicts one action and writes it as
`actions.npy`, float32, shape `[1, 1, 7]`. `/score` writes the
log-probabilities of the 7 action tokens over the full vocabulary.

The server does not run the checkpoint's remote code. That code needs
transformers 4.40.1. `openvla.py` builds the same network and loads the same
weights; its docstring gives the reason.

## Model parameters

| model | revision | cards | checkpoint dtype | decoding |
|---|---|---|---|---|
| openvla/openvla-7b | `47a0ec7fc4ec123775a391911046cf33cf9ed83f` | 1 | bfloat16 | greedy, 7 tokens |

The 5090 stack uses CPython 3.12, transformers 5.22.1, timm 0.9.16,
torch 2.14.1+cu130 and torchvision 0.29.1+cu130. Greedy decoding uses no
seed.

## Server: NVIDIA A100 (sm80)

### Host environment

```bash
export VLA_HOME="$HOME/serve-venv-vla-cu128"
export VLA_PYTHON="$VLA_HOME/bin/python"
export UV_CACHE_DIR="$HOME/.cache/uv"
export UV_HTTP_TIMEOUT=600
export UV_NO_CACHE=0
mkdir -p "$UV_CACHE_DIR"
uv venv "$VLA_HOME" --python 3.12
uv pip install --python "$VLA_PYTHON" \
  torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$VLA_PYTHON" --no-deps \
  torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$VLA_PYTHON" \
  transformers==5.22.1 timm==0.9.16 numpy pillow
```

### Verify environment

```bash
export CUDA_VISIBLE_DEVICES=0
export SERVE_VLA_PYTHON="$VLA_PYTHON"
"$VLA_PYTHON" -c 'import torch, torchvision, transformers, timm; \
assert torch.cuda.is_available(); \
print(torch.__version__, torch.version.cuda, torchvision.__version__, \
      transformers.__version__, timm.__version__, \
      torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), \
      torch.cuda.is_bf16_supported())'
uv pip check --python "$VLA_PYTHON"
```

### Model root

```bash
export MODEL_ROOT="$HOME/models"
export HF_HOME="$MODEL_ROOT/hub"
export MODEL_REVISION=47a0ec7fc4ec123775a391911046cf33cf9ed83f
mkdir -p "$HF_HOME"
```

### Download

```bash
nohup env \
  HF_HUB_DISABLE_TELEMETRY=1 \
  HF_HUB_DISABLE_XET=1 \
  HF_ENDPOINT=https://hf-mirror.com \
  "$VLA_HOME/bin/hf" download \
  openvla/openvla-7b --revision "$MODEL_REVISION" \
  --cache-dir "$HF_HOME" \
  > "$MODEL_ROOT/openvla-download.log" 2>&1 < /dev/null &
printf 'download pid: %s\n' "$!"
```

```bash
tail -n 30 "$MODEL_ROOT/openvla-download.log"
```

### Verify model

```bash
export MODEL="$HF_HOME/models--openvla--openvla-7b/snapshots/$MODEL_REVISION"
printf 'shards: '
find -L "$MODEL" -maxdepth 1 -name 'model-*-of-00003.safetensors' | wc -l
du -shL "$MODEL"
ls "$MODEL"/config.json "$MODEL"/preprocessor_config.json
```

Expected: three model shards and about 15.1 GB.

### Serve in FP32 on the A100

From this deployment directory on the A100, start the server with the verified model:

```bash
SERVE_DTYPE=float32 ./serve.sh --port 8090 --model "$MODEL" --device cuda
```

In a second terminal, check the model dtype:

```bash
curl -fsS http://127.0.0.1:8090/info | python3 -c 'import json, sys; print(json.load(sys.stdin)["dtype"])'
```

Expected output is `float32`.

## Server: NVIDIA RTX 5090 (sm_120)

### Environment

The launcher uses this path.

```bash
uv venv /root/serve-venv-vla --python 3.12
uv pip install --python /root/serve-venv-vla/bin/python \
  torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
uv pip install --python /root/serve-venv-vla/bin/python --no-deps \
  torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
uv pip install --python /root/serve-venv-vla/bin/python \
  transformers==5.22.1 timm==0.9.16 numpy pillow
```

The torch build must carry sm_120; cu130 does. torchvision 0.29.x pairs with
torch 2.14.x. `--no-deps` keeps the torch build. timm builds the two vision
towers and imports torchvision. Use timm 0.9.x only: timm 1.0 changed `get_intermediate_layers`, and the reference code refuses it.

Verify the stack:

```bash
/root/serve-venv-vla/bin/python -c 'import torch, transformers, timm; \
  print(torch.__version__, transformers.__version__, timm.__version__, \
  torch.cuda.get_device_name(0))'
```

### Layout

```text
/opt/engines/vla-openvla-tf   this directory
/models/hub                   hub cache; the weights are one snapshot in it
```

### Transfer

```bash
ssh -p $PORT user@$HOST 'mkdir -p /opt/engines/vla-openvla-tf'
rsync -a --partial -e "ssh -p $PORT" \
  --exclude __pycache__ --exclude .DS_Store \
  ./vla-openvla-tf/ user@$HOST:/opt/engines/vla-openvla-tf/
ssh -p $PORT user@$HOST 'chmod +x /opt/engines/vla-openvla-tf/*.sh /opt/engines/vla-openvla-tf/*.py'
rsync -a --partial -e "ssh -p $PORT" user@$HOST:/outputs/ ./outputs/
```

Download the weights on the box into the hub cache. The snapshot directory carries the revision; `/info` reports it. On vast, huggingface.co times
out, so the download goes through the mirror endpoint:

```bash
HF_HUB_DISABLE_TELEMETRY=1 \
HF_ENDPOINT=https://hf-mirror.com /root/serve-venv-vla/bin/hf download \
  openvla/openvla-7b --revision 47a0ec7fc4ec123775a391911046cf33cf9ed83f \
  --cache-dir /models/hub
export MODEL=/models/hub/models--openvla--openvla-7b/snapshots/47a0ec7fc4ec123775a391911046cf33cf9ed83f
ls $MODEL
```

Verify: the directory holds three `model-*-of-00003.safetensors` shards, `config.json`, `preprocessor_config.json` and the tokenizer files. The
weights take 15.1 GB.

### Start

```bash
cd /opt/engines/vla-openvla-tf
mkdir -p /outputs
export CUDA_VISIBLE_DEVICES=0
./serve.sh --port 8090 --model $MODEL --device cuda 2>&1 | tee /outputs/vla-openvla-tf.log
```

### Parameters

| variable | effect |
|---|---|
| `SERVE_DTYPE` | Model execution dtype. Accepts `bfloat16` or `float32`. When unset, uses the checkpoint dtype. `GET /info` reports the selected value in `dtype`. |
| `SERVE_VLA_PYTHON` | the interpreter the launcher execs; default `/root/serve-venv-vla/bin/python` |
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
7 action tokens. Each token selects one of 255 bin centres over [-1, 1]. The dataset
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

The observation is an `.npz` file: `prompt`, a string, and one camera, float32
`[3, H, W]` with 1.0 at full scale. The server ignores `state` and `noise`.
Any other key is a camera; two cameras stop the request with an error.

`/score` decode builds the prompt from the observation's task when
`input_ids` is empty. It uses other ids as sent: at least 2 ids, `<s>` first.
It writes 7 rows, one per action token, and the ids prompt + tokens. A prefill
takes those ids back with the same observation. It writes the teacher-forced
rows: n - 1, or the last `n_rows`. Every row is the float32 log-softmax of
the raw logits over all 32064 columns. The server reads no seed and no
sampling: decoding is greedy. `max_new_tokens` must be 7.

## Check

The identity surface answers with the engine's own facts:

```bash
curl -s http://127.0.0.1:8090/info | python3 -m json.tool
```

Make an observation from a photo. Any RGB photo serves. A frame from the
robot's third-person camera gives a meaningful action:

```bash
mkdir -p /tmp/serve-vla
/root/serve-venv-vla/bin/python /opt/engines/vla-openvla-tf/make_observation.py \
  frame.jpg "put the eggplant in the basket" /tmp/serve-vla/obs001.npz
```

Send it and read the action:

```bash
curl -s http://127.0.0.1:8090/infer \
  -H 'Content-Type: application/json' \
  -d '{"prompt_id": "obs001", "out_dir": "/tmp/serve-vla/out/obs001",
       "attachments": {"observation": "/tmp/serve-vla/obs001.npz"}}' \
  | python3 -m json.tool
/root/serve-venv-vla/bin/python -c \
  'import numpy as np; a = np.load("/tmp/serve-vla/out/obs001/actions.npy"); print(a.shape, a.dtype, a)'
```

Expect `(1, 1, 7) float32`. The server log shows the 7 tokens; each must lie
in 31744..31999.

Make a second observation from a different photo, with the same task:

```bash
/root/serve-venv-vla/bin/python /opt/engines/vla-openvla-tf/make_observation.py \
  other.jpg "put the eggplant in the basket" /tmp/serve-vla/obs002.npz
```

Score both. The two frames must give different log-probabilities:

```bash
for id in obs001 obs002; do
  curl -s http://127.0.0.1:8090/score -H 'Content-Type: application/json' \
    -d "{\"prompt_id\": \"$id\", \"mode\": \"decode\", \"input_ids\": [],
         \"max_new_tokens\": 7, \"out_dir\": \"/tmp/serve-vla/score/$id\",
         \"attachments\": {\"observation\": \"/tmp/serve-vla/$id.npz\"}}"
  echo
done
/root/serve-venv-vla/bin/python -c 'import numpy as np
a, b = (np.load(f"/tmp/serve-vla/score/{i}/log_probs.npy") for i in ("obs001", "obs002"))
print(a.shape, a.dtype, "max |diff|", abs(a - b).max())'
```

Expect `(7, 32064) float32` and a max |diff| above 0. A difference of 0
means the frame does not reach the model, and every action is wrong.
