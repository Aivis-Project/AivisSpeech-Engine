# 仮想環境を作って実行に必要な依存のみをインストールし、ライセンス一覧を生成する

set -eux

if [[ -z "${OUTPUT_LICENSE_JSON_PATH+x}" ]]; then
  echo 'OUTPUT_LICENSE_JSON_PATH is not defined.'
  exit 1
fi

output_license_json_tmp_path="$(mktemp "${TMPDIR:-/tmp}/dependency_licenses.json.tmp.XXXXXX")"

cleanup_tmp_file() {
  rm -f "$output_license_json_tmp_path"
}

trap cleanup_tmp_file EXIT

uv venv 'licenses_venv'
export VIRTUAL_ENV='licenses_venv'
uv sync --active --group licenses
uv run --active tools/generate_licenses.py > "$output_license_json_tmp_path"
if [[ "$OUTPUT_LICENSE_JSON_PATH" != '/dev/null' ]]; then
  mv -f "$output_license_json_tmp_path" "$OUTPUT_LICENSE_JSON_PATH"
fi

rm -rf "$VIRTUAL_ENV"
unset VIRTUAL_ENV
