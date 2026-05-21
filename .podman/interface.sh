#!/bin/sh

image="${1:-docker.io/taylorabarnes/devenv:latest}"
port="${2:-56610}"

# Check if host has X11 available
if [ -n "$DISPLAY" ] && [ -e "/tmp/.X11-unix/X${DISPLAY#:}" ]; then
  # Use Vulkan backend with host X11 (Intel GPU has good Vulkan support)
  # Force X11 backend for winit (not Wayland)
  X11_ARGS="-e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -e XDG_RUNTIME_DIR=/tmp/runtime -e WINIT_UNIX_BACKEND=x11 --device /dev/dri --group-add keep-groups --security-opt label=disable"
  XVFB_PREFIX="mkdir -p /tmp/runtime && "
  echo "Note: Host X11 display detected. GPU acceleration enabled."
else
  # No host X11 - use Xvfb inside the container
  # Note: GPU apps (Bevy, wgpu) won't work properly with Xvfb due to lack of DRI3 support
  # Force OpenGL software rendering for basic X11 apps
  X11_ARGS="-e DISPLAY=:99 -e WGPU_BACKEND=gl -e WINIT_UNIX_BACKEND=x11 -e LIBGL_ALWAYS_SOFTWARE=1 -e XDG_RUNTIME_DIR=/tmp/runtime --device /dev/dri --group-add keep-groups --security-opt label=disable"
  XVFB_PREFIX="mkdir -p /tmp/runtime && Xvfb :99 -screen 0 1024x768x24 & sleep 1 && "
  echo "Note: No host X11 display detected. Using virtual framebuffer (Xvfb)."
  echo "Warning: GPU-accelerated apps (Bevy, wgpu) require a host X11 display."
  echo ""
fi

echo "Entering an interactive terminal session"
echo ""
podman run --rm --device nvidia.com/gpu=all -it -v "$(pwd):/work" ${X11_ARGS} -v ~/.claude:/root/.claude:cached -v ~/.claude.json:/root/.claude.json ${image} bash -c "${XVFB_PREFIX}exec bash"

