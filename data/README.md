# Hierarchical dataset format

The graph `.pt` file contains a list of PyG `Data` objects. Each object requires:

- `index`: scalar sample identifier matching the CSV `index`.
- `x`: integer graph-node vocabulary identifiers.
- `ast_edge_index`: AST edges with shape `[2, edge_count]`.
- `cfg_edge_index`: CFG edges with shape `[2, edge_count]`.

The CSV requires:

- `index`, `func_before`, `llm_explanation`, `root_label`.
- `level_1` through `level_5`, encoded as contiguous class indices per level.
- Optional `level_mask_1` through `level_mask_5`.
- Optional `cwe_id`, `pair_id`, and `is_fixed`.

Safe samples use `root_label=0`; their hierarchy labels are ignored. Vulnerable samples use
`root_label=1`. Missing hierarchy levels should be `-100` with a zero level mask.
