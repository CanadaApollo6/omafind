#!/bin/sh
# Builds omafind from this checkout and installs it for the current user:
#   ~/.local/bin/omafind, models in ~/.local/share/omafind/models, a systemd user service
#   that indexes in the background and, on Omarchy, the Find launcher plugin.
# Options (environment): CUDA=1 builds with the CUDA execution provider (needs cuDNN 9);
# NO_SERVICE=1 skips the service; NO_PLUGIN=1 skips the launcher plugin.
set -eu

cd "$(dirname "$0")/.."
bin_dir="${BIN_DIR:-$HOME/.local/bin}"
config_home="${XDG_CONFIG_HOME:-$HOME/.config}"

features=""
[ "${CUDA:-0}" = 1 ] && features="--features cuda"
echo "==> Building omafind"
# shellcheck disable=SC2086
cargo build --release --locked $features

echo "==> Installing $bin_dir/omafind"
running=0
systemctl --user is-active --quiet omafind.service 2>/dev/null && running=1
[ "$running" = 1 ] && systemctl --user stop omafind.service
install -Dm755 target/release/omafind "$bin_dir/omafind"

conf="$config_home/omafind/config.toml"
if [ ! -f "$conf" ]; then
    echo "==> Writing $conf"
    mkdir -p "$(dirname "$conf")"
    "$bin_dir/omafind" config >"$conf"
fi

echo "==> Fetching models (verified by SHA-256; about 370 MB)"
"$bin_dir/omafind" models fetch

if [ "${NO_SERVICE:-0}" != 1 ]; then
    echo "==> Installing the systemd user service"
    unit_dir="$config_home/systemd/user"
    mkdir -p "$unit_dir"
    sed "s|^ExecStart=.*|ExecStart=$bin_dir/omafind daemon|" dist/systemd/omafind.service >"$unit_dir/omafind.service"
    systemctl --user daemon-reload
    systemctl --user enable --now omafind.service
    [ "$running" = 1 ] && systemctl --user restart omafind.service
fi

if [ "${NO_PLUGIN:-0}" != 1 ] && command -v omarchy >/dev/null 2>&1; then
    echo "==> Installing the Omarchy launcher plugin"
    plugins="$config_home/omarchy/plugins"
    mkdir -p "$plugins"
    rm -rf "$plugins/omafind.search"
    cp -r dist/omarchy/omafind.search "$plugins/"
    omarchy plugin enable omafind.search || true
    cat <<'MSG'

To open Find with a key, add this to ~/.config/hypr/bindings.lua:

    o.bind("SUPER + CTRL + SLASH", "Find files", "omarchy-shell shell toggle omafind.search")

MSG
fi

echo "Done. The first index of your home folder runs in the background; follow it with"
echo "  omafind status"
echo "Roots, exclusions and limits are in $conf (restart the service after editing)."
