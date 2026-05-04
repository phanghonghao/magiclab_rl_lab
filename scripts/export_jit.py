#!/usr/bin/env python3
"""
Universal JIT export script for rsl-rl checkpoints.

Extracts the actor network from an rsl-rl checkpoint, traces it as a JIT model.
Pure PyTorch operation — does NOT launch Isaac Sim.

Supports both:
  - rsl-rl 3.x: checkpoint contains `model_state_dict` (full ActorCritic)
  - rsl-rl 5.x: checkpoint contains `actor_state_dict` + `critic_state_dict`

Usage:
    # From a checkpoint (auto-detect obs/action dims from state_dict):
    python scripts/export_jit.py --checkpoint logs/rsl_rl/<RUN>/model_5000.pt

    # Optionally specify deploy.yaml for metadata:
    python scripts/export_jit.py \
        --checkpoint logs/rsl_rl/<RUN>/model_5000.pt \
        --deploy_cfg logs/rsl_rl/<RUN>/params/deploy.yaml

    # Export ONNX as well:
    python scripts/export_jit.py --checkpoint logs/rsl_rl/<RUN>/model_5000.pt --onnx
"""

import argparse
import os
import sys

import torch
import torch.nn as nn


def parse_args():
    parser = argparse.ArgumentParser(description="Export rsl-rl actor network as JIT/ONNX model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to rsl-rl checkpoint (.pt)")
    parser.add_argument("--deploy_cfg", type=str, default=None, help="Path to deploy.yaml (optional, for metadata)")
    parser.add_argument("--output", type=str, default=None, help="Output path (default: <checkpoint_dir>/exported/policy.pt)")
    parser.add_argument("--onnx", action="store_true", help="Also export as ONNX")
    parser.add_argument("--device", type=str, default="cpu", help="Device for trace (default: cpu)")
    return parser.parse_args()


def extract_actor_state(ckpt: dict) -> dict:
    """Extract actor state_dict from checkpoint, handling rsl-rl 3.x and 5.x formats.

    rsl-rl 3.x: `model_state_dict` with keys like `actor.0.weight`, `actor.0.bias`, ..., `std`
    rsl-rl 5.x: `actor_state_dict` with keys like `0.weight`, `0.bias`, ..., plus `std`
    """
    if "model_state_dict" in ckpt:
        # rsl-rl 3.x: strip "actor." prefix, keep only actor keys
        state = {}
        for k, v in ckpt["model_state_dict"].items():
            if k.startswith("actor."):
                state[k[len("actor."):]] = v
            elif k == "std":
                state[k] = v
        return state
    elif "actor_state_dict" in ckpt:
        # rsl-rl 5.x: already actor-only
        return dict(ckpt["actor_state_dict"])
    else:
        raise ValueError(
            "Checkpoint does not contain 'model_state_dict' (rsl-rl 3.x) "
            "or 'actor_state_dict' (rsl-rl 5.x). "
            f"Available keys: {list(ckpt.keys())}"
        )


def infer_mlp_structure(state: dict) -> tuple[list[int], str]:
    """Infer MLP layer sizes and activation from state_dict keys.

    The actor is an nn.Sequential with pattern:
        Linear, Activation, Linear, Activation, ..., Linear

    Keys are indexed: 0.weight, 0.bias (Linear), 2.weight, 2.bias (Linear), ...

    Returns:
        layer_sizes: [obs_dim, hidden1, hidden2, ..., action_dim]
        activation: "elu" or "relu" (default elu)
    """
    # Find all Linear layer indices (have .weight and .bias)
    linear_indices = sorted(set(
        int(k.split(".")[0]) for k in state.keys()
        if k.endswith(".weight") and "std" not in k
    ))

    layer_sizes = []
    for idx in linear_indices:
        w = state[f"{idx}.weight"]
        in_features = w.shape[1]
        out_features = w.shape[0]
        if not layer_sizes:
            layer_sizes.append(in_features)
        layer_sizes.append(out_features)

    return layer_sizes


def build_actor_mlp(layer_sizes: list[int], activation: str = "elu") -> nn.Sequential:
    """Build an MLP matching the rsl-rl actor architecture.

    Structure: Linear, Act, Linear, Act, ..., Linear (no activation on last layer)
    """
    act_fn = nn.ELU if activation == "elu" else nn.ReLU

    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        # Add activation after every layer except the last
        if i < len(layer_sizes) - 2:
            layers.append(act_fn())

    return nn.Sequential(*layers)


def export_jit(checkpoint_path: str, output_path: str, device: str = "cpu") -> int:
    """Export actor network from checkpoint as JIT model.

    Returns:
        obs_dim: observation dimension for reference
    """
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(f"[INFO] Loaded checkpoint: {checkpoint_path}")
    print(f"[INFO] Checkpoint keys: {list(ckpt.keys())}")
    if "iter" in ckpt:
        print(f"[INFO] Training iteration: {ckpt['iter']}")

    # Extract actor state
    actor_state = extract_actor_state(ckpt)
    print(f"[INFO] Actor state_dict keys ({len(actor_state)}):")
    for k, v in actor_state.items():
        print(f"  {k}: {v.shape}")

    # Infer network structure from state_dict
    layer_sizes = infer_mlp_structure(actor_state)
    obs_dim = layer_sizes[0]
    action_dim = layer_sizes[-1]
    hidden_dims = layer_sizes[1:-1]
    print(f"[INFO] Inferred network: obs_dim={obs_dim}, hidden={hidden_dims}, action_dim={action_dim}")

    # Build actor MLP and load weights
    actor = build_actor_mlp(layer_sizes, activation="elu")

    # Load state (strip "std" key which is not part of Sequential)
    load_state = {k: v for k, v in actor_state.items() if k != "std"}
    actor.load_state_dict(load_state)
    actor.eval()
    actor.to(device)
    print(f"[INFO] Actor network loaded and set to eval mode")

    # Export as JIT
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Use torch.jit.script for reliability (matches IsaacLab's exporter)
    traced = torch.jit.script(actor)
    traced.save(output_path)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[INFO] JIT model saved to: {output_path} ({file_size_mb:.2f} MB)")

    return obs_dim


def export_onnx(checkpoint_path: str, output_path: str, device: str = "cpu"):
    """Export actor network from checkpoint as ONNX model."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    actor_state = extract_actor_state(ckpt)
    layer_sizes = infer_mlp_structure(actor_state)
    obs_dim = layer_sizes[0]

    actor = build_actor_mlp(layer_sizes, activation="elu")
    load_state = {k: v for k, v in actor_state.items() if k != "std"}
    actor.load_state_dict(load_state)
    actor.eval()
    actor.to(device)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    dummy_input = torch.randn(1, obs_dim, device=device)
    torch.onnx.export(
        actor,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=18,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={},
    )
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[INFO] ONNX model saved to: {output_path} ({file_size_mb:.2f} MB)")


def main():
    args = parse_args()

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        ckpt_dir = os.path.dirname(args.checkpoint)
        output_path = os.path.join(ckpt_dir, "exported", "policy.pt")

    # Export JIT
    obs_dim = export_jit(args.checkpoint, output_path, device=args.device)

    # Verify the exported model loads correctly
    loaded = torch.jit.load(output_path, map_location=args.device)
    test_input = torch.randn(1, obs_dim, device=args.device)
    with torch.no_grad():
        test_output = loaded(test_input)
    print(f"[INFO] Verification: input shape={test_input.shape}, output shape={test_output.shape}")
    print(f"[INFO] Sample output: {test_output[0, :4].tolist()}")

    # Export ONNX if requested
    if args.onnx:
        onnx_path = output_path.replace(".pt", ".onnx")
        export_onnx(args.checkpoint, onnx_path, device=args.device)

    print("[INFO] Export complete!")


if __name__ == "__main__":
    main()
