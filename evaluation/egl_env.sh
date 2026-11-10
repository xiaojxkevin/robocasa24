# Source this file before launching Python. Local user-space libraries only.
robocasa24_enable_egl() {
    local repo_root egl_root
    repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
    egl_root="$repo_root/.runtime/nvidia-egl/root"
    if [[ ! -f "$egl_root/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.550.144.03" ]]; then
        echo 'Missing local NVIDIA EGL runtime; see evaluation/README.md' >&2
        return 1
    fi
    if ! [[ "$(cat /proc/driver/nvidia/version)" == *550.144.03* ]]; then
        echo 'Local EGL runtime 550.144.03 must match the loaded kernel driver' >&2
        return 1
    fi
    export LD_LIBRARY_PATH="$egl_root/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export __EGL_VENDOR_LIBRARY_FILENAMES="$egl_root/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    export MUJOCO_GL=egl
    export PYOPENGL_PLATFORM=egl
    export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
}
robocasa24_enable_egl
