#!/bin/bash
# Build the conversion image with its commit stamped in.
#
# The image has no .git, so the converter cannot discover which build it is at
# run time; it reads DRONESYNTH_GIT_COMMIT, which is baked in here. That stamp
# ends up in every dataset the image converts, so building by hand with a bare
# `docker build` costs the provenance record its commit. Hence this script:
# the documented path computes the args rather than relying on anyone to
# remember them.
#
# A dirty tree is recorded, not refused. Iterating on the converter against
# real data is normal; shipping a dataset whose labels came from uncommitted
# code and not knowing is the problem.
#
#   scripts/build_image.sh                 # -> dronesynth-convert:latest
#   scripts/build_image.sh myname:mytag
set -euo pipefail

cd "$(dirname "$0")/.."

tag=${1:-dronesynth-convert:latest}

commit=$(git rev-parse --short=10 HEAD 2>/dev/null || echo unknown)

# Only what the Dockerfile copies can differ between this commit and the image,
# so those are the paths the dirty flag is about. An edited README does not make
# the running converter disagree with its stamp; an edited src/ does.
IMAGE_PATHS=(src configs pyproject.toml docker)

# Deliberately not `git status --porcelain`. On a Windows checkout the worktree
# holds CRLF while the index holds LF, so run from WSL that reports every file
# in the repo as modified and the flag would be stuck at true -- which is worse
# than useless, since it would discredit stamps that are in fact exact.
if ! git diff --quiet --ignore-cr-at-eol HEAD -- "${IMAGE_PATHS[@]}" 2>/dev/null; then
  dirty=true
elif [[ -n "$(git ls-files --others --exclude-standard -- "${IMAGE_PATHS[@]}" 2>/dev/null)" ]]; then
  # Untracked files under those paths get copied in too.
  dirty=true
else
  dirty=false
fi
if [[ "$commit" == unknown ]]; then
  dirty=unknown
fi

echo "building $tag at commit $commit (dirty=$dirty)"
if [[ "$dirty" == true ]]; then
  echo "  note: working tree is dirty; datasets built by this image will say so"
fi
echo

docker build \
  -f docker/Dockerfile \
  --build-arg "GIT_COMMIT=$commit" \
  --build-arg "GIT_DIRTY=$dirty" \
  -t "$tag" \
  .

echo
echo "built $tag"
echo "stamp: $(docker run --rm --entrypoint printenv "$tag" DRONESYNTH_GIT_COMMIT)"
