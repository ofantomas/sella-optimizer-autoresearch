#!/usr/bin/env bash
set -Eeuo pipefail

CONDA_ENV_NAME="${CONDA_ENV_NAME:-sella-autoresearch}"
CONDA_ENV_FILE="${CONDA_ENV_FILE:-environment.yml}"
RALPH_DIR="${RALPH_DIR:-$HOME/tools/open-ralph-wiggum}"
RALPH_REPO="${RALPH_REPO:-https://github.com/Th0rgal/open-ralph-wiggum}"
INSTALL_CURSOR_AGENT="${INSTALL_CURSOR_AGENT:-1}"
INSTALL_MINIFORGE="${INSTALL_MINIFORGE:-1}"

usage() {
  cat <<'EOF'
Usage: scripts/install_autoresearch_tools.sh

Environment variables:
  CONDA_ENV_NAME          Conda environment name to create/update.
                          Default: sella-autoresearch
  CONDA_ENV_FILE          Conda environment file.
                          Default: environment.yml
  RALPH_DIR               Open Ralph Wiggum checkout path.
                          Default: $HOME/tools/open-ralph-wiggum
  RALPH_REPO              Open Ralph Wiggum git remote.
                          Default: https://github.com/Th0rgal/open-ralph-wiggum
  INSTALL_CURSOR_AGENT    Set to 0 to skip Cursor Agent installation.
                          Default: 1
  INSTALL_MINIFORGE       Set to 0 to require an existing conda install.
                          Default: 1
EOF
}

log() {
  printf '\n==> %s\n' "$*"
}

repo_root() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "$script_dir/.." && pwd
}

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    printf 'Missing required command: %s\n' "$cmd" >&2
    exit 1
  fi
}

ensure_local_bin_on_path() {
  export PATH="$HOME/.local/bin:$HOME/.bun/bin:$PATH"
  mkdir -p "$HOME/.local/bin"
}

load_conda() {
  if command -v conda >/dev/null 2>&1; then
    return 0
  fi

  local candidate
  for candidate in "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
    if [[ -f "$candidate/etc/profile.d/conda.sh" ]]; then
      # shellcheck disable=SC1090
      source "$candidate/etc/profile.d/conda.sh"
      return 0
    fi
  done

  return 1
}

install_miniforge() {
  if [[ "$INSTALL_MINIFORGE" != "1" ]]; then
    printf 'conda was not found and INSTALL_MINIFORGE=0.\n' >&2
    exit 1
  fi

  local arch installer_url tmp_installer install_dir
  case "$(uname -m)" in
    x86_64) arch="x86_64" ;;
    aarch64|arm64) arch="aarch64" ;;
    *)
      printf 'Unsupported architecture for Miniforge installer: %s\n' "$(uname -m)" >&2
      exit 1
      ;;
  esac

  install_dir="$HOME/miniforge3"
  installer_url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${arch}.sh"
  tmp_installer="$(mktemp "${TMPDIR:-/tmp}/miniforge.XXXXXX.sh")"

  log "Installing Miniforge into $install_dir"
  curl -fsSL "$installer_url" -o "$tmp_installer"
  bash "$tmp_installer" -b -p "$install_dir"
  rm -f "$tmp_installer"

  # shellcheck disable=SC1090
  source "$install_dir/etc/profile.d/conda.sh"
}

ensure_conda() {
  if load_conda; then
    log "Using conda: $(command -v conda)"
    return
  fi

  require_cmd curl
  install_miniforge
  load_conda
  log "Using conda: $(command -v conda)"
}

ensure_bun() {
  ensure_local_bin_on_path
  if command -v bun >/dev/null 2>&1; then
    log "Using bun: $(command -v bun)"
    return
  fi

  require_cmd curl
  log "Installing Bun"
  curl -fsSL https://bun.sh/install | bash
  ensure_local_bin_on_path
  command -v bun >/dev/null 2>&1
}

ensure_ralph() {
  require_cmd git
  ensure_bun

  log "Installing Open Ralph Wiggum in $RALPH_DIR"
  mkdir -p "$(dirname "$RALPH_DIR")"
  if [[ -d "$RALPH_DIR/.git" ]]; then
    git -C "$RALPH_DIR" pull --ff-only
  else
    git clone "$RALPH_REPO" "$RALPH_DIR"
  fi

  (
    cd "$RALPH_DIR"
    bun install
    chmod +x ralph.ts
  )

  cat >"$HOME/.local/bin/ralph" <<EOF
#!/usr/bin/env bash
export PATH="\$HOME/.bun/bin:\$HOME/.local/bin:\$PATH"
exec bun "$RALPH_DIR/ralph.ts" "\$@"
EOF
  chmod +x "$HOME/.local/bin/ralph"
  log "Installed ralph wrapper: $HOME/.local/bin/ralph"
}

ensure_cursor_agent() {
  ensure_local_bin_on_path
  if [[ "$INSTALL_CURSOR_AGENT" != "1" ]]; then
    log "Skipping Cursor Agent installation"
    return
  fi

  if command -v agent >/dev/null 2>&1; then
    log "Using Cursor Agent: $(command -v agent)"
    return
  fi

  require_cmd curl
  log "Installing Cursor Agent CLI"
  curl https://cursor.com/install -fsS | bash
  ensure_local_bin_on_path
  command -v agent >/dev/null 2>&1
}

ensure_conda_env() {
  local root env_file
  root="$(repo_root)"
  env_file="$root/$CONDA_ENV_FILE"
  if [[ ! -f "$env_file" ]]; then
    printf 'Conda environment file does not exist: %s\n' "$env_file" >&2
    exit 1
  fi

  ensure_conda
  log "Creating/updating conda env: $CONDA_ENV_NAME"
  if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV_NAME"; then
    conda env update -n "$CONDA_ENV_NAME" -f "$env_file" --prune
  else
    conda env create -n "$CONDA_ENV_NAME" -f "$env_file"
  fi
}

main() {
  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
  fi

  ensure_local_bin_on_path
  require_cmd git
  require_cmd curl

  ensure_bun
  ensure_ralph
  ensure_cursor_agent
  ensure_conda_env

  log "Installation complete"
  printf 'Add this to your shell startup if needed:\n'
  printf '  export PATH="$HOME/.local/bin:$HOME/.bun/bin:$PATH"\n'
  printf 'Activate the environment with:\n'
  printf '  conda activate %s\n' "$CONDA_ENV_NAME"
}

main "$@"
