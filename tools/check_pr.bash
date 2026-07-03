#!/usr/bin/env bash
set -euo pipefail

ACTIONLINT_VERSION="${ACTIONLINT_VERSION:-1.7.12}"
PINACT_VERSION="${PINACT_VERSION:-4.1.0}"
SHELLCHECK_VERSION="${SHELLCHECK_VERSION:-0.8.0}"
PR_CHECK_TOOLS_DIR="${PR_CHECK_TOOLS_DIR:-.cache/pr-check-tools}"

usage() {
  cat <<'EOF'
Usage: bash tools/check_pr.bash [--all|--python|--lint-builders]

Runs the local equivalent of the pull request checks in .github/workflows/test.yml.
The Python checks run on the current OS; GitHub Actions still provides the
Windows/macOS/Linux matrix coverage.

Modes:
  --all             Run Python checks and lint-builders checks. Default.
  --python          Run the test.yml test job commands for the current OS.
  --lint-builders   Run the test.yml lint-builders job commands.
  -h, --help        Show this help.
EOF
}

mode="all"
while (($# > 0)); do
  case "$1" in
    --all)
      mode="all"
      ;;
    --python)
      mode="python"
      ;;
    --lint-builders)
      mode="lint-builders"
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

run_step() {
  local label="$1"
  shift
  printf '\n==> %s\n' "$label"
  "$@"
}

tool_platform() {
  local os
  local arch

  case "$(uname -s)" in
    Linux)
      os="linux"
      ;;
    Darwin)
      os="darwin"
      ;;
    *)
      echo "Unsupported OS for automatic tool download: $(uname -s)" >&2
      exit 127
      ;;
  esac

  case "$(uname -m)" in
    x86_64 | amd64)
      arch="amd64"
      ;;
    aarch64 | arm64)
      arch="arm64"
      ;;
    *)
      echo "Unsupported architecture for automatic tool download: $(uname -m)" >&2
      exit 127
      ;;
  esac

  echo "${os}_${arch}"
}

shellcheck_platform() {
  case "$(uname -s):$(uname -m)" in
    Linux:x86_64 | Linux:amd64)
      echo "linux.x86_64"
      ;;
    Linux:aarch64 | Linux:arm64)
      echo "linux.aarch64"
      ;;
    Darwin:x86_64 | Darwin:amd64)
      echo "darwin.x86_64"
      ;;
    *)
      echo "Unsupported platform for ShellCheck ${SHELLCHECK_VERSION}: $(uname -s) $(uname -m)" >&2
      exit 127
      ;;
  esac
}

ensure_shellcheck() {
  if command -v shellcheck >/dev/null 2>&1; then
    command -v shellcheck
    return
  fi

  local platform
  local shellcheck_dir
  local shellcheck_bin
  local archive
  local url

  platform="$(shellcheck_platform)"
  shellcheck_dir="${PR_CHECK_TOOLS_DIR}/shellcheck-${SHELLCHECK_VERSION}-${platform}"
  shellcheck_bin="${shellcheck_dir}/shellcheck-v${SHELLCHECK_VERSION}/shellcheck"
  archive="${shellcheck_dir}/shellcheck.tar.xz"

  if [[ ! -x "$shellcheck_bin" ]]; then
    mkdir -p "$shellcheck_dir"
    url="https://github.com/koalaman/shellcheck/releases/download/v${SHELLCHECK_VERSION}/shellcheck-v${SHELLCHECK_VERSION}.${platform}.tar.xz"
    echo "Downloading shellcheck ${SHELLCHECK_VERSION} for ${platform}" >&2
    curl -fsSL --retry 3 --retry-delay 5 "$url" -o "$archive"
    tar -xf "$archive" -C "$shellcheck_dir"
    chmod +x "$shellcheck_bin"
  fi

  echo "$shellcheck_bin"
}

ensure_actionlint() {
  if command -v actionlint >/dev/null 2>&1; then
    command -v actionlint
    return
  fi

  local platform
  local actionlint_dir
  local archive
  local url

  platform="$(tool_platform)"
  actionlint_dir="${PR_CHECK_TOOLS_DIR}/actionlint-${ACTIONLINT_VERSION}-${platform}"
  archive="${actionlint_dir}/actionlint.tar.gz"

  if [[ ! -x "${actionlint_dir}/actionlint" ]]; then
    mkdir -p "$actionlint_dir"
    url="https://github.com/rhysd/actionlint/releases/download/v${ACTIONLINT_VERSION}/actionlint_${ACTIONLINT_VERSION}_${platform}.tar.gz"
    echo "Downloading actionlint ${ACTIONLINT_VERSION} for ${platform}" >&2
    curl -fsSL --retry 3 --retry-delay 5 "$url" -o "$archive"
    tar -xzf "$archive" -C "$actionlint_dir" actionlint
    chmod +x "${actionlint_dir}/actionlint"
  fi

  echo "${actionlint_dir}/actionlint"
}

ensure_pinact() {
  if command -v pinact >/dev/null 2>&1; then
    command -v pinact
    return
  fi

  local platform
  local pinact_dir
  local archive
  local url

  platform="$(tool_platform)"
  pinact_dir="${PR_CHECK_TOOLS_DIR}/pinact-${PINACT_VERSION}-${platform}"
  archive="${pinact_dir}/pinact.tar.gz"

  if [[ ! -x "${pinact_dir}/pinact" ]]; then
    mkdir -p "$pinact_dir"
    url="https://github.com/suzuki-shunsuke/pinact/releases/download/v${PINACT_VERSION}/pinact_${platform}.tar.gz"
    echo "Downloading pinact ${PINACT_VERSION} for ${platform}" >&2
    curl -fsSL --retry 3 --retry-delay 5 "$url" -o "$archive"
    tar -xzf "$archive" -C "$pinact_dir" pinact
    chmod +x "${pinact_dir}/pinact"
  fi

  echo "${pinact_dir}/pinact"
}

check_shell_files() {
  local shellcheck_bin
  local shell_files=()

  shellcheck_bin="$(ensure_shellcheck)"
  mapfile -t shell_files < <(git ls-files "*.bash" "*.sh")
  if ((${#shell_files[@]} == 0)); then
    return
  fi

  "$shellcheck_bin" "${shell_files[@]}"
}

check_workflow_files() {
  local actionlint_bin
  local shellcheck_bin

  actionlint_bin="$(ensure_actionlint)"
  shellcheck_bin="$(ensure_shellcheck)"
  PATH="$(dirname "$shellcheck_bin"):$PATH" "$actionlint_bin"
}

check_pinned_actions() {
  local pinact_bin
  local github_token

  pinact_bin="$(ensure_pinact)"
  github_token="${PINACT_GITHUB_TOKEN:-${GITHUB_TOKEN:-${GH_TOKEN:-}}}"

  if [[ -z "$github_token" ]] && command -v gh >/dev/null 2>&1; then
    github_token="$(gh auth token --hostname github.com 2>/dev/null || true)"
  fi

  if [[ -n "$github_token" ]]; then
    PINACT_GITHUB_TOKEN="$github_token" "$pinact_bin" run -check -min-age 7
  else
    "$pinact_bin" run -check -min-age 7
  fi
}

run_python_checks() {
  run_step "Install Python dependencies" uv sync --frozen --group dev
  run_step "Validate uv.lock" uv lock --check
  run_step "Check uv.lock did not change" git diff --exit-code -- uv.lock
  run_step "Check linting" env PYTHONUTF8=1 uv run ruff check
  run_step "Check formatting" env PYTHONUTF8=1 uv run ruff format --check
  run_step "Check typing" uv run task mypy
  run_step "Test codes and coverage" uv run coverage run --omit=test/* -m pytest
  run_step "Check licenses" uv run task check-licenses
  run_step "Test names by checking typo" uv run task typos
}

run_lint_builders_checks() {
  run_step "Check shell files" check_shell_files
  run_step "Check workflow files" check_workflow_files
  run_step "Check pinned GitHub Actions" check_pinned_actions
}

cd "$(git rev-parse --show-toplevel)"

case "$mode" in
  all)
    run_python_checks
    run_lint_builders_checks
    ;;
  python)
    run_python_checks
    ;;
  lint-builders)
    run_lint_builders_checks
    ;;
esac
