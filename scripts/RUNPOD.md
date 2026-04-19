# Running on RunPod

Step-by-step for your first cloud GPU training run. Should take ~15 minutes
the first time.

## 0. One-time prerequisites (on your Mac)

1. Push this repo to GitHub (public is simplest, no auth needed on the pod):
   ```bash
   gh repo create efficientzero-v2 --public --source=. --push
   ```
   (or use the GitHub web UI then `git remote add origin ... && git push -u origin main`)

2. Make a free [Weights & Biases](https://wandb.ai) account. Copy your API
   key from https://wandb.ai/authorize.

3. Make a [RunPod](https://runpod.io) account. Add ~$10 of credits.

## 1. Launch a pod

1. RunPod web UI → **Pods** → **Deploy**.
2. Select **Secure Cloud** (more reliable while you're learning) and a single
   **RTX 3090** (~$0.43/hr) or **RTX 4090** (~$0.69/hr).
3. Template: **RunPod PyTorch 2.x** (whichever is the latest stable).
4. Storage: 30 GB container disk + 30 GB volume is plenty.
5. Click **Deploy On-Demand**.

The pod takes ~30s to boot. When it's ready, click **Connect → SSH**.

## 2. SSH in and set up

```bash
# (paste the SSH command from RunPod's Connect dialog)

cd /workspace
git clone https://github.com/<your-username>/efficientzero-v2.git
cd efficientzero-v2

# put your W&B key in a .env file (or just `export WANDB_API_KEY=...`)
cp .env.example .env
nano .env       # paste your WANDB_API_KEY

bash scripts/setup_remote.sh
```

`setup_remote.sh` installs deps, verifies the GPU, logs into W&B, and runs
the smoke test. If that all succeeds, you're ready to train.

## 3. Start training inside tmux

`tmux` is essential — without it, your training dies the moment your SSH
session disconnects.

```bash
tmux new -s train
make train-atari        # or train-dmc-proprio / train-dmc-vision
# detach: Ctrl-b then d
```

You can now close the SSH window. Training keeps going.

## 4. Monitor

- Open https://wandb.ai/<your-username>/efficientzero-v2 in your browser.
- Watch loss curves, episode returns, eval scores.
- You can do this from your Mac, your phone, anywhere.

## 5. Reattach later

```bash
ssh ...                 # back into the pod
tmux attach -t train    # see live stdout
```

## 6. Pull the trained checkpoint down

From your Mac:
```bash
scp -P <port> root@<pod-ip>:/workspace/efficientzero-v2/checkpoints/<run>/latest.pt ./
```

Or push to W&B Artifacts from inside the trainer — see `utils/logging.py`.

## 7. STOP THE POD when done

In the RunPod web UI: **Pods → ⋯ → Stop**. (Or `runpodctl stop pod <id>`.)

If you stop instead of terminate, your storage persists (~$0.10/GB/month)
and you can resume later. **Terminate** if you don't need the volume.

You will absolutely forget at least once. Set a billing alert.

## Cost reality check

- Smoke test on CPU: free.
- Single Atari-100k run on a 3090: ~$1–3.
- Single DMC task on a 3090: ~$1–5.
- Full reproduction of paper (26 Atari × 3 seeds + 20 DMC × 3 seeds × 2 obs): ~$2k.

Stop after each task while learning.
