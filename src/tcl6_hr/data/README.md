# Generator data

`hr-v1.jsonl` contains the 124 corrected Hadamard-reduced groups HR001-HR124.
It is the sole generator-expression input required by the CPU runtime.
`cc-v1.jsonl` contains the 40 connected cumulative-correlation terms for
reference; normal evaluation does not repeat the symbolic CC-to-HR derivation.

The JSON schemas describe these records. `manifest.json` is preserved original
derivation metadata: its historical file references describe the source
project, not extra installation requirements. Historical 125-group datasets,
spreadsheet sources, and compatibility aliases are deliberately absent.

`source-provenance.json` records hashes of source files/data and the definitions
retained in the internal engine. The CPU package retains the existing temporal
formulas; public solver glue and standalone lower orders are new files.

Unit conventions and lambda powers are specified in `docs/conventions.md` in
the source distribution and in the public API docstrings.
