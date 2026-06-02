#!/usr/bin/env bash
# Load KEY = value or KEY=value lines from a dotenv file into the environment.
# Usage: source scripts/load_env.sh [path_to_env_file]
# Defaults to $REPO_ROOT/.env when REPO_ROOT is set, else ./.env
load_env_file() {
  local env_file="${1:-${REPO_ROOT:-.}/.env}"
  [[ -f "$env_file" ]] || return 0
  local line key value
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%#*}"
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "$line" ]] && continue
    if [[ "$line" =~ ^([^=[:space:]]+)[[:space:]]*=[[:space:]]*(.*)$ ]]; then
      key="${BASH_REMATCH[1]}"
      value="${BASH_REMATCH[2]}"
      value="${value#"${value%%[![:space:]]*}"}"
      value="${value%"${value##*[![:space:]]}"}"
      # Strip optional surrounding quotes.
      if [[ "$value" =~ ^\"(.*)\"$ ]]; then value="${BASH_REMATCH[1]}"; fi
      if [[ "$value" =~ ^\'(.*)\'$ ]]; then value="${BASH_REMATCH[1]}"; fi
      export "$key=$value"
    fi
  done < "$env_file"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  load_env_file "$1"
fi
