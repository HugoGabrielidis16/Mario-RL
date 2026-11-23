"""Train or evaluate a PPO policy on Super Mario Bros. World 1-1."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory


DEFAULT_CHECKPOINT = Path("runs/ppo/mario_1_1.zip")


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def save_checkpoint(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent, prefix=".checkpoint-") as directory:
        temporary_path = Path(directory) / path.name
        model.save(temporary_path)
        temporary_path.replace(path)
    print(f"Saved checkpoint at {model.num_timesteps} steps to {path}", flush=True)


def train(args, parser):
    rollout_steps = args.rollout_steps or min(512, args.steps // args.envs)
    rollout_size = rollout_steps * args.envs
    if rollout_size < 2 or args.steps % rollout_size:
        parser.error(
            f"--steps must be a multiple of the total rollout size ({rollout_size}). "
            "Each rollout must collect at least 2 steps across all environments."
        )

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    from ppo_environment import make_ppo_env

    device = torch.device(args.device) if args.device != "auto" else None
    if device is not None and device.type == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA is unavailable. Check the GPU allocation, driver, and PyTorch installation.")
        index = device.index if device.index is not None else 0
        if index >= torch.cuda.device_count():
            parser.error(f"{args.device} is outside the allocated CUDA devices.")
        print(f"Using {args.device} ({torch.cuda.get_device_name(index)})", flush=True)
    elif args.device == "cpu":
        torch.set_num_threads(2)

    checkpoint = args.output if args.output.suffix == ".zip" else Path(f"{args.output}.zip")

    class PeriodicCheckpoint(BaseCallback):
        def _on_training_start(self):
            self.last_saved = self.num_timesteps

        def _on_rollout_start(self):
            # The previous rollout has been used for optimizer updates.
            if self.num_timesteps - self.last_saved >= args.checkpoint_every:
                save_checkpoint(self.model, checkpoint)
                self.last_saved = self.num_timesteps

        def _on_step(self):
            return True

    batch_size = min(64, rollout_size)
    env = make_ppo_env(args.envs)
    try:
        if args.resume:
            overrides = {"seed": args.seed} if args.seed is not None else {}
            model = PPO.load(
                args.resume, env=env, device=args.device, n_steps=rollout_steps,
                batch_size=batch_size, verbose=int(args.verbose), **overrides
            )
        else:
            model = PPO(
                "CnnPolicy", env, n_steps=rollout_steps, batch_size=batch_size,
                n_epochs=4, seed=args.seed if args.seed is not None else 0, device=args.device,
                verbose=int(args.verbose),
            )

        model.learn(
            total_timesteps=args.steps,
            reset_num_timesteps=args.resume is None,
            callback=PeriodicCheckpoint() if args.checkpoint_every else None,
            log_interval=10,
        )
        save_checkpoint(model, checkpoint)
        print(f"Trained for {args.steps} steps ({model.num_timesteps} total).")
    finally:
        env.close()


def evaluate(args):
    import torch
    from stable_baselines3 import PPO

    from ppo_environment import make_ppo_env

    if args.device == "cpu":
        torch.set_num_threads(2)
    env = make_ppo_env()
    try:
        model = PPO.load(args.checkpoint, device=args.device)
        obs = env.reset()
        clears = 0
        for episode in range(1, args.episodes + 1):
            max_x = 0
            steps = 0
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, dones, infos = env.step(action)
                info = infos[0]
                steps += 1
                max_x = max(max_x, info["x_pos"])
                if dones[0]:
                    cleared = bool(info["flag_get"])
                    clears += int(cleared)
                    outcome = "clear" if cleared else "death"
                    if info["TimeLimit.truncated"]:
                        outcome = "timeout"
                    print(f"Episode {episode}: {outcome}, max_x={max_x}, steps={steps}")
                    break
        print(f"World 1-1 clears: {clears}/{args.episodes}")
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="train for a bounded number of steps")
    train_parser.add_argument(
        "--steps", type=positive_int, default=64, help="additional agent steps (default: 64)"
    )
    train_parser.add_argument(
        "--rollout-steps", type=positive_int,
        help="rollout steps per environment (default: min(512, steps // envs))",
    )
    train_parser.add_argument(
        "--envs", type=positive_int, default=1, help="parallel emulators (default: 1)"
    )
    train_parser.add_argument(
        "--checkpoint-every", type=positive_int,
        help="refresh the output checkpoint after this many steps at a rollout boundary",
    )
    train_parser.add_argument("--verbose", action="store_true", help="log PPO training metrics")
    train_parser.add_argument("--seed", type=int, help="training seed (default: 0 for new models)")
    train_parser.add_argument("--resume", type=Path, help="continue from a PPO checkpoint")
    train_parser.add_argument(
        "--output", type=Path, default=DEFAULT_CHECKPOINT,
        help=f"checkpoint path (default: {DEFAULT_CHECKPOINT})",
    )

    eval_parser = commands.add_parser("evaluate", help="play World 1-1 with a saved checkpoint")
    eval_parser.add_argument("checkpoint", type=Path)
    eval_parser.add_argument("--episodes", type=positive_int, default=1)

    for command in (train_parser, eval_parser):
        command.add_argument("--device", default="cpu", help="PyTorch device (default: cpu)")

    args = parser.parse_args()
    if args.command == "train":
        train(args, train_parser)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
