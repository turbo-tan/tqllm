#!/data/data/com.termux/files/usr/bin/bash
# Install the checksum-pinned native Android bundle over SSH. No root needed.
set -euo pipefail

if [[ $(uname -m) != aarch64 || -z ${PREFIX:-} ]] || ! command -v pkg >/dev/null; then
    echo 'Run this installer inside 64-bit ARM Termux on the phone.' >&2
    exit 1
fi
host=${1:?Usage: bash install-termux.sh USER@HOST}
if [[ $host == -* || $host == *[[:space:]]* || $host != *@* ]]; then
    echo 'Use an SSH destination in USER@HOST form.' >&2
    exit 1
fi
readonly model_name=Qwen3.5-4B-Q4_K_M.gguf
readonly model_bytes=2740937888
readonly model_sha=00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4
readonly bundle_name=tqllm-android-s25-ultra-arm64.zip
readonly bundle_sha=e9ee3d05eb56bd55947f3c96f0cedd333c187d95338aef692d3f5b2f2ce4f89a
readonly remote_dir=.cache/tqllm/android-s25
readonly app="$HOME/tqllm-android"

pkg install -y openssh rsync unzip coreutils
mkdir -p "$app/models" "$app/downloads" "$app/releases"
model="$app/models/$model_name"
bundle="$app/downloads/$bundle_name"
verify() { printf '%s  %s\n' "$2" "$1" | sha256sum --check --status; }

# Interrupted model downloads are kept and resumed on the next invocation.
remaining=$model_bytes
if [[ -f $model ]] && verify "$model" "$model_sha"; then
    remaining=0
elif [[ -f $model.partial ]]; then
    partial_bytes=$(stat -c %s "$model.partial")
    if ((partial_bytes > model_bytes)); then
        echo "Oversized partial file: $model.partial. Move it aside and retry." >&2
        exit 1
    fi
    remaining=$((model_bytes - partial_bytes))
fi
free_kb=$(df -Pk "$app" | awk 'END {print $4}')
if [[ ! $free_kb =~ ^[0-9]+$ ]] || ((free_kb * 1024 < remaining + 64 * 1024 * 1024)); then
    echo 'Not enough free app storage. A fresh install needs about 2.9 GB free.' >&2
    exit 1
fi
socket_dir=$(mktemp -d)
cleanup() {
    ssh -S "$socket_dir/connection" -O exit "$host" >/dev/null 2>&1 || true
    rm -rf -- "$socket_dir"
}
trap cleanup EXIT
transport="ssh -o ControlMaster=auto -o ControlPersist=60 -o ControlPath=$socket_dir/connection"
echo "Connecting to $host. Accept its SSH host key only if it is your server."
ssh -o ControlMaster=auto -o ControlPersist=60 -o ControlPath="$socket_dir/connection" "$host" true
if [[ ! -f $bundle ]] || ! verify "$bundle" "$bundle_sha"; then
    rsync -L --partial --append-verify --info=progress2 -e "$transport" \
        "$host:$remote_dir/$bundle_name" "$bundle.partial"
    verify "$bundle.partial" "$bundle_sha" || { echo 'Bundle checksum failed.' >&2; exit 1; }
    mv -f -- "$bundle.partial" "$bundle"
fi
if [[ ! -f $model ]] || ! verify "$model" "$model_sha"; then
    rsync -L --partial --append-verify --info=progress2 -e "$transport" \
        "$host:$remote_dir/$model_name" "$model.partial"
    verify "$model.partial" "$model_sha" || { echo 'Model checksum failed.' >&2; exit 1; }
    mv -f -- "$model.partial" "$model"
fi
release="$app/releases/${bundle_sha:0:16}"
mkdir -p "$release"
unzip -oq "$bundle" -d "$release"
chmod 700 "$release/bin/tqandroid-run" "$release/bin/test-android-native"
echo 'Checking the native executable on this phone...'
"$release/bin/test-android-native"
"$release/bin/tqandroid-run" --help
launcher="$PREFIX/bin/tqandroid"
if [[ -e $launcher ]] && ! grep -q '^# tqllm-native-android-launcher$' "$launcher"; then
    echo "An unrelated command already exists at $launcher; leaving it unchanged." >&2
    exit 1
fi
{
    printf '#!%s/bin/bash\n# tqllm-native-android-launcher\nset -euo pipefail\n' "$PREFIX"
    printf 'binary=%q\nmodel=%q\n' "$release/bin/tqandroid-run" "$model"
    cat <<'LAUNCH'
if (($# == 0)); then
    read -r -p 'You: ' prompt
    [[ -n $prompt ]] || exit 0
    set -- --prompt "$prompt"
elif [[ $1 != --* ]]; then
    set -- --prompt "$*"
fi
exec "$binary" --model "$model" --context 4096 --threads 4 --max-tokens 256 "$@"
LAUNCH
} > "$app/launcher.new"
chmod 700 "$app/launcher.new"
mv -f -- "$app/launcher.new" "$launcher"
echo
printf 'Installed. Run:\n  tqandroid "Say hello in English and Chinese."\n'
printf 'Or type tqandroid and enter a prompt. This first runner handles one request at a time.\n'
