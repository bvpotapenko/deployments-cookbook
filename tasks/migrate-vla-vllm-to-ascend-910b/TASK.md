# Task: migrate the CUDA VLA deployment to Ascend 910B

## Starting point

The agent receives the CUDA reference deployment in
`cuda-deployments/vla-vllm/`. It serves `openvla/openvla-7b` at the KOT
contract and is the source of truth for preprocessing, prompt construction,
token decoding, action unnormalization, and score outputs.

The agent does **not** receive an NPU-ready VLA deployment. It must build the
Ascend deployment from the CUDA reference and the conventions documented by
the other Ascend deployments in this repository.

## Goal

Create an Ascend 910B deployment under `npu-deployments/vla-vllm/` that can be
run by Bastet/KOT with the same model revision and the same observable
contract as the CUDA reference:

- `GET /kot/info`
- `POST /kot/infer` → `actions.npy`, `float32`, shape `[1, 1, 7]`
- `POST /kot/score` decode → seven full-vocabulary log-probability rows
- `POST /kot/score` prefill → teacher-forced rows over caller-supplied ids

The target must preserve the reference's input semantics: one RGB camera in a
KOT observation, the pinned OpenVLA prompt, 224×224 preprocessing, the
`bridge_orig` action statistics, action-token ids 31744–31999, greedy seven
token decoding, and the full vocabulary width reported by the checkpoint.

## Ascend packaging requirements

The returned deployment must be self-contained and runnable on an Ascend
910B host through a Docker image. Follow the existing NPU deployment pattern:

- provide a `Dockerfile` based on a CANN 910B image;
- install the matching CUDA-free `torch`, `torch_npu`, and `torchvision`
  wheels explicitly rather than resolving them at image build time;
- pin the transformers/model stack and document the CPython version;
- provide a launcher that mounts the Ascend driver, device nodes, model, and
  writable KOT output directories;
- keep the server code mounted or copied in a way that matches the launcher;
- make version, device, dtype, model revision, and toolkit facts visible in
  `/kot/info`;
- never require network access when the graded container starts.

The image/driver and wheelhouse layout can be copied conceptually from
`npu-deployments/mllm-vllm/` and `tasks/migrate-ocr-transformers-4-to-5/`.
Those files are packaging references only; they are not VLA implementations.

## Required self-test

The CUDA reference directories contain `self_check.py`. An agent must run the
corresponding self-test against its returned deployment before declaring it
ready:

```bash
python3 self_check.py --deployment /path/to/deployment \
  --model /models/openvla-7b --device npu
```

The self-test creates deterministic synthetic observations and checks:

1. the deployment starts and `/kot/info` answers;
2. `/kot/infer` writes finite `float32 [1, 1, 7]` actions;
3. decode scoring writes finite normalized rows with the configured
   vocabulary width and seven output ids;
4. replaying the stored ids through prefill preserves the ids and row count;
5. changing the image changes the score rows, proving the frame reaches the
   model instead of being silently dropped.

The self-test checks the wire contract, shapes, finiteness, normalization and
data flow. It does not replace CUDA-vs-NPU numerical comparison; Bastet owns
that comparison using the frozen reference captures.

## Acceptance criteria

- `self_check.py` passes on the returned Ascend deployment.
- The same frozen observation can be sent to CUDA and NPU with the same task
  and produces comparable token/log-probability/action artifacts.
- A fresh container can start without host Python packages or network access.
- The run book states the exact image, wheel, model, device, and launch
  requirements, plus the self-test command.
- Changes are limited to this migration task and the two CUDA self-tests.
