#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

install_conda() {
  pushd .ci/docker || return
  # --override-channels excludes the anaconda defaults channel so the
  # solve doesn't have to reconcile defaults' older transitive deps
  # (e.g. zlib=1.2.13, rhash=1.4.3) with conda-forge's newer ones
  # required by cmake=3.31.2 (libzlib>=1.3.1, rhash>=1.4.5). Mixing
  # the two channels intermittently failed with LibMambaUnsatisfiable
  # Error; libmamba on macOS does not implement strict channel
  # priority, so the only deterministic fix is to drop defaults.
  ${CONDA_INSTALL} -c conda-forge --override-channels -y --file conda-env-ci.txt
  popd || return
}

install_conda
