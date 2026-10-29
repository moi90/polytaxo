# PolyTaxo

<p align="center" width="100%">
    <img width="15%" src="docs/transparent-128.png" alt="PolyTaxo">
</p>

PolyTaxo is a versatile hierarchical classification system designed to handle
complex, polyhierarchical descriptions. It supports creating, manipulating, and
querying taxonomies whose nodes can belong to multiple categories, making it
useful for overlapping classifications, multi-label tagging, and ontologies.

## Features

- **Flexible hierarchical structure:** Create taxonomies with hierarchical and
  polyhierarchical relationships.
- **Descriptor-based classification:** Define, negate, and query structures
  using descriptors.
- **Conflict resolution:** Handle conflicting descriptors with customizable
  strategies (`replace`, `skip`, `raise`).
- **Expression parsing:** Parse and apply complex expressions to match or
  modify descriptions.
- **Probability-based classification:** Generate descriptions from
  probabilities over taxonomy nodes.
- **Linear hierarchy parsing:** Convert linear hierarchies into PolyTaxo
  descriptions; virtual nodes can break compound names into simpler concepts.

## Use case

PolyTaxo was created to categorize zooplankton objects in marine biology. For
example, a copepod can be described as `Copepoda sex:female view:lateral`,
indicating a female copepod viewed laterally. This polyhierarchical system
supports complex queries and advanced machine-learning tasks in the MAZE-IPP
project.

The [package README](lib/README.md) contains usage examples. The shared
documentation site is in [`docs/`](docs/).

## Contributing

Contributions are welcome! Please submit issues or pull requests on the
[GitHub repository](https://github.com/moi90/polytaxo).

## License

This project is licensed under the MIT License. See the `LICENSE` file for
details.

For more information, see the [original blog post](https://www.nerdluecht.de/index.php/2023/10/23/polytaxo-using-a-polyhierarchical-taxonomy-to-describe-zooplankton-objects/).
