# Isaac Sim Docker Streaming

Use this when Isaac/IsaacLab runs on a remote RTX server and the local machine does not have an NVIDIA GPU.

## Prerequisites

- Server has `NVIDIA Driver`, `nvidia-container-toolkit`, and `Docker`
- Server GPU supports `NVENC` (`A100` does not support Isaac Sim livestream)
- Open these ports on the server firewall/security group:
  - `49100/TCP` for signaling
  - `47998/UDP` for media stream

## Option 1: docker compose

Run on the server from the repo root:

```bash
cd magiclab_rl_lab
export PUBLIC_IP=<server_public_ip_or_reachable_lan_ip>
export SIGNAL_PORT=49100
export STREAM_PORT=47998
docker compose -f deploy/docker-compose.isaac-streaming.yml up
```

## Option 2: docker run helper

```bash
cd magiclab_rl_lab
chmod +x deploy/isaac_sim_streaming_docker.sh
PUBLIC_IP=<server_public_ip_or_reachable_lan_ip> \
  ./deploy/isaac_sim_streaming_docker.sh
```

## Client Connection

- Use the official Isaac Sim WebRTC Streaming Client
- Connect to `PUBLIC_IP:49100`

## Browser Viewer

This repo does not start the browser viewer stack.

- NVIDIA also provides an official browser-based viewer
- Its default URL is `http://PUBLIC_IP:8210/streaming/webrtc-client`
- That path uses NVIDIA's separate compose stack, not the single-container `runheadless.sh` flow in this repo

## Low-Latency Notes

- Prefer low RTT network and wired connection
- Keep training headless and launch a separate low-`num_envs` preview/play session for interaction
- Reduce render load before changing network settings: fewer envs, fewer sensors, lower render resolution
- Set `PUBLIC_IP` to the real externally reachable address when crossing NAT

## Debug

```bash
docker logs isaac-sim-streaming
ss -lntup | grep -E '49100|47998'
```
