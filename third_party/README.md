# Third-party source

StockMixer is referenced as a Git submodule:

| Project | Upstream | Pinned commit |
|---|---|---|
| StockMixer | <https://github.com/SJTU-DMTai/StockMixer> | `cce13598afd3ff33ae317700a85ae08db0554652` |

Run `bash scripts/setup_third_party.sh` after cloning. The script idempotently applies
`patches/stockmixer-variable-lookback.patch`, which derives the strided Conv1d scale length from the configured
lookback instead of using the upstream value hard-coded for 16 days.

The upstream project retains its own copyright and license terms. No upstream datasets are redistributed here.
