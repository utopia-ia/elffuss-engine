// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 UtopiaIA
//
// GENERATED from the tuning JSON files. Do not edit by hand: change the JSON
// (through the autotune skill) and rebuild with the tuning build script.
export default {
  "default": {
    "bench": {
      "heldoutSha": "",
      "suite": "memory",
      "tuneSha": "",
      "version": 1
    },
    "model": "default",
    "rev": 1,
    "schema": 1,
    "shared": {
      "memory": {
        "write": {
          "dupSim": 0.85,
          "similarMin": 0.5,
          "support": 0.6
        }
      }
    },
    "targets": {
      "memory": {
        "abstain": {
          "minCoverage": 0.34
        },
        "block": {
          "format": "list-dated",
          "rules": "v1",
          "wording": "v1"
        },
        "budget": {
          "coreMax": 160,
          "coreShare": 0.4,
          "max": 1024,
          "maxItems": 16,
          "maxItemsSmall": 4,
          "min": 192,
          "noCtx": 300,
          "share": 0.03,
          "smallCtx": 8192
        },
        "proposer": {
          "perConv": 3,
          "perTurn": 1
        },
        "rank": {
          "b": 0.3,
          "halfLifeDays": 30,
          "k1": 1.2,
          "keyBoost": 2,
          "minScore": 0.25,
          "prevWeight": 0.5,
          "wImp": 0.27,
          "wRec": 0.27,
          "wRel": 1,
          "wTmp": 0.27
        }
      }
    }
  }
};
