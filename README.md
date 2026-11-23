## Mario Reinforcement Learning

Train a Stable-Baselines3 PPO policy to clear Super Mario Bros. World 1-1.

### Setup

Use Python 3.10. The NES emulator dependency builds a native extension and may
require a C++ compiler.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Train and evaluate

The default run collects 64 agent steps, performs one PPO rollout update, and
saves a checkpoint. Training runs headlessly and each agent step repeats a
controller action for up to four emulator frames.

```bash
python train_ppo.py train
python train_ppo.py evaluate runs/ppo/mario_1_1.zip
```

The evaluator uses deterministic actions and reports flag captures and maximum
horizontal position. The 64-step run verifies training and checkpoint loading.

Train in chunks by loading a checkpoint and specifying how many additional
steps to run. Supply a new output path to keep the earlier checkpoint.

```bash
python train_ppo.py train --steps 4096 --resume runs/ppo/mario_1_1.zip --output runs/ppo/mario_1_1_4160.zip
python train_ppo.py evaluate runs/ppo/mario_1_1_4160.zip
```

PPO updates after each complete rollout. With one emulator, the default
rollout size is 512 steps or the requested step count if it is smaller. The
total step budget must be divisible by the rollout size across all emulators.

### Train on one NVIDIA GPU

Copy the project or unpack the cluster bundle on a filesystem shared by the
cluster login and compute nodes. Run the commands below from the directory
containing `train_ppo.py`. The setup targets Linux x86-64 and needs Python
3.10, a C++ compiler for `nes-py`, and access to PyPI and the PyTorch wheel
index. If your site supplies Python or a compiler via environment modules,
load them before setup and submission.

```bash
bash cluster/setup.sh
sbatch --time=00:15:00 cluster/train.slurm --steps 2048
```

The setup installs a CUDA 12.6 build of PyTorch in `.venv-cluster`. Follow
your cluster's driver guidance when choosing a wheel. For example, run
`CUDA_WHEEL=cu128 bash cluster/setup.sh` for a CUDA 12.8 build, or set
`CUDA_WHEEL=cu118` for CUDA 11.8. Use `PYTHON_BIN` if Python 3.10 has a
different executable name. The batch job prints the GPU name and fails if
PyTorch cannot access CUDA. [PyTorch CUDA wheel options](https://pytorch.org/get-started/previous-versions/#v271)

Slurm requests one GPU and six CPU cores. PPO uses the GPU for the neural
network and runs four headless NES emulators on CPU. Add your site's
`--partition`, `--account`, or `--qos` options to `sbatch` when required.
[Slurm submission options](https://slurm.schedmd.com/sbatch.html)

Once the smoke job succeeds, submit a longer run.

```bash
sbatch cluster/train.slurm --steps 1024000
squeue --me
```

Replace `12345` below with the job ID returned by `sbatch`.

```bash
tail -f mario-ppo-12345.log
.venv-cluster/bin/python train_ppo.py evaluate runs/cluster/12345/latest.zip
```

Training logs include measured steps per second. With four emulators and
512 steps per emulator, one rollout is 2,048 agent steps. The batch launcher
refreshes `latest.zip` every 24,576 steps and when training ends. Each
refresh saves the policy and optimizer after the preceding update.

Resume from the latest checkpoint if you hit the time limit. The new job
keeps its output in its own directory, and `--steps` is the additional
budget.

```bash
sbatch cluster/train.slurm --steps 204800 --resume runs/cluster/12345/latest.zip
```

On an already allocated GPU node you can also run the same trainer directly.

```bash
.venv-cluster/bin/python -u train_ppo.py train --device cuda --envs 4 --verbose \
  --steps 1024000 --checkpoint-every 24576 --output runs/gpu/latest.zip
```

In the full repository, the DQN experiments use `train.py`.
