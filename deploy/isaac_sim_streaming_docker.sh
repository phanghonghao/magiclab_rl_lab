#!/usr/bin/env bash
set -euo pipefail

# Official Isaac Sim WebRTC streaming ports
SIGNAL_PORT="${SIGNAL_PORT:-49100}"
STREAM_PORT="${STREAM_PORT:-47998}"

# Public IP/FQDN reachable by the client.
# For same-LAN usage, you can set this to the server LAN IP.
PUBLIC_IP="${PUBLIC_IP:-}"

ISAACSIM_IMAGE="${ISAACSIM_IMAGE:-nvcr.io/nvidia/isaac-sim:5.1.0}"
CONTAINER_NAME="${CONTAINER_NAME:-isaac-sim-streaming}"

if [[ -z "${PUBLIC_IP}" ]]; then
  echo "ERROR: PUBLIC_IP is required."
  echo "Example:"
  echo "  PUBLIC_IP=203.0.113.10 ./deploy/isaac_sim_streaming_docker.sh"
  exit 1
fi

echo "[info] starting container: ${CONTAINER_NAME}"
echo "[info] image: ${ISAACSIM_IMAGE}"
echo "[info] public ip: ${PUBLIC_IP}"
echo "[info] signal port: ${SIGNAL_PORT} (tcp)"
echo "[info] stream port: ${STREAM_PORT} (udp)"
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

docker run --name "${CONTAINER_NAME}" --rm -it \
  --gpus all \
  --network host \
  --ipc host \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -e CARB_APP_PATH=/isaac-sim/kit \
  -e PUBLIC_IP="${PUBLIC_IP}" \
  -v "${HOME}/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw" \
  -v "${HOME}/docker/isaac-sim/cache/ov:/root/.cache/ov:rw" \
  -v "${HOME}/docker/isaac-sim/cache/pip:/root/.cache/pip:rw" \
  -v "${HOME}/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw" \
  -v "${HOME}/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw" \
  -v "${HOME}/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw" \
  -v "${HOME}/docker/isaac-sim/data:/root/.local/share/ov/data:rw" \
  -v "${HOME}/docker/isaac-sim/documents:/root/Documents:rw" \
  "${ISAACSIM_IMAGE}" \
  bash -lc "./runheadless.sh \
    --/exts/omni.kit.livestream.app/primaryStream/publicIp=${PUBLIC_IP} \
    --/exts/omni.kit.livestream.app/primaryStream/signalPort=${SIGNAL_PORT} \
    --/exts/omni.kit.livestream.app/primaryStream/streamPort=${STREAM_PORT}"
