#!/usr/bin/env bash
# Status line for the PyTorch CV Boilerplate project.
# Renders: <dir> | <git branch> | <release> <phase progress> | <model>
# Phase progress is read from docs/dev/v*/backlog.json, the single source of
# truth for phase status (see CLAUDE.md, Document Rules).

set -u

input="$(cat)"

read_json() {
  # $1: python expression over the parsed stdin payload bound as `d`
  printf '%s' "$input" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    d = {}
def g(*keys, default=''):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur
print($1)
" 2>/dev/null
}

model="$(read_json "g('model','display_name', default='?')")"
cwd="$(read_json "g('workspace','current_dir', default='') or g('cwd', default='')")"
[ -z "$cwd" ] && cwd="$PWD"

dir_name="$(basename "$cwd")"

# Git branch and dirty marker.
branch=""
if git -C "$cwd" rev-parse --git-dir >/dev/null 2>&1; then
  branch="$(git -C "$cwd" branch --show-current 2>/dev/null)"
  [ -z "$branch" ] && branch="detached"
  if [ -n "$(git -C "$cwd" status --porcelain 2>/dev/null)" ]; then
    branch="${branch}*"
  fi
fi

# Current release and phase progress from the newest docs/dev/v*/backlog.json.
phase_info=""
backlog="$(ls -1d "$cwd"/docs/dev/v*/backlog.json 2>/dev/null | sort -V | tail -n 1)"
if [ -n "$backlog" ]; then
  release="$(basename "$(dirname "$backlog")")"
  counts="$(python3 -c "
import json, sys
try:
    phases = json.load(open(sys.argv[1])).get('phases', [])
except Exception:
    phases = []
done = sum(1 for p in phases if p.get('status') == 'completed')
active = next((p for p in phases if p.get('status') not in ('completed',)), None)
label = active.get('id', '').replace('phase-', 'P') if active else 'done'
print('%s %s %d/%d' % (sys.argv[2], label, done, len(phases)))
" "$backlog" "$release" 2>/dev/null)"
  phase_info="$counts"
fi

# Conda environment, if the session inherited one.
env_name="${CONDA_DEFAULT_ENV:-}"

dim=$'\033[2m'
cyan=$'\033[36m'
green=$'\033[32m'
yellow=$'\033[33m'
reset=$'\033[0m'

out="${cyan}${dir_name}${reset}"
[ -n "$branch" ] && out="${out} ${dim}|${reset} ${green}${branch}${reset}"
[ -n "$phase_info" ] && out="${out} ${dim}|${reset} ${yellow}${phase_info}${reset}"
[ -n "$env_name" ] && out="${out} ${dim}|${reset} ${dim}${env_name}${reset}"
out="${out} ${dim}|${reset} ${dim}${model}${reset}"

printf '%s' "$out"
