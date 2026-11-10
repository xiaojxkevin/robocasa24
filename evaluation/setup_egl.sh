#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
package_name=libnvidia-gl-550_550.144.03-0ubuntu1_amd64.deb
expected_sha=78a99f1e888e313584da355c196f46472b9b1b25eddb31733a7576018ae7f510
if ! [[ "$(cat /proc/driver/nvidia/version)" == *550.144.03* ]]; then
    echo 'This local runtime recipe requires kernel driver 550.144.03' >&2
    exit 1
fi
mkdir -p "$repo_root/.runtime/nvidia-egl"
cd "$repo_root/.runtime/nvidia-egl"
if [[ ! -f "$package_name" ]]; then
    apt-get download libnvidia-gl-550=550.144.03-0ubuntu1
fi
printf '%s  %s\n' "$expected_sha" "$package_name" | sha256sum --check
dpkg-deb -x "$package_name" root
echo "NVIDIA EGL extracted under $PWD/root; source evaluation/egl_env.sh before Python."
