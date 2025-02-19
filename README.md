# PolyTaxo

<p align="center" width="100%">
    <img width="15%" src="docs/transparent-128.png"> 
</p>


## Overview

The `PolyTaxo` library is a versatile hierarchical classification system designed to handle complex, polyhierarchical descriptions.
It allows for the creation, manipulation, and querying of a taxonomy where nodes can belong to multiple categories, supporting both hierarchical and non-hierarchical structures.
This library is particularly useful for organizing data with overlapping categories, such as multi-label classification, tagging systems, and ontologies.

## Features

- **Flexible Hierarchical Structure:** Create and manipulate taxonomies with both hierarchical and polyhierarchical relationships.
- **Descriptor-Based Classification:** Use descriptors to define, negate, and query hierarchical structures.
- **Conflict Resolution:** Handle conflicting descriptors with customizable conflict resolution strategies (`replace`, `skip`, `raise`).
- **Expression Parsing:** Parse and apply complex expressions to match or modify descriptions.
- **Probability-Based Classification:** Generate descriptions based on probability distributions over nodes.
- **Parsing of Linear Hierarchies:** Parse linear hierarchies to obtain PolyTaxo descriptions.
  **Virtual Nodes** define complex descriptions so that compound names that don't exist in the primary hierarchy can be broken down into simpler concepts.

## Usage Examples

### Creating a Taxonomy

Here’s how to create a taxonomy for zooplankton classification:

```python
from polytaxo import PolyTaxonomy

taxonomy_dict = {
    "Copepoda": {
        "tags": {
            "view": {"lateral": {}, "dorsal": {}, "ventral": {}},
            "sex": {"male": {}, "female": {}},
            "stage": {"CI": {}, "CII": {}, "CIII": {}, "CIV": {}, "CV": {}},
        },
        "children": {
            "Calanus": {"children": {"Calanus finmarchicus": {}, "Calanus glacialis": {}}},
            "Metridia": {"children": {"Metridia longa": {}}},
        }
    }
}

taxonomy = PolyTaxonomy.from_dict(taxonomy_dict)
```

### Describing Objects

Using the created taxonomy, you can describe specific objects:

```python
description = taxonomy.parse_description("Copepoda Calanus view:lateral sex:female")
print(description)  # Output: /Copepoda/Calanus view:lateral sex:female
```

### Parsing Complex Expressions

You can parse and apply complex expressions:

```python
expression = taxonomy.parse_expression("Copepoda/Calanus -view:dorsal")
modified_description = expression.apply(description)
```

### Integrating Probabilities

Generate descriptions based on node probability scores:

```python
probabilities = {0: 0.85, 1: 0.15, 2: 0.92}
description = taxonomy.parse_probabilities(probabilities)
```

### Visualizing Taxonomy

Format and print the taxonomy:

```python
taxonomy.print_tree(virtuals=True)
```

## Use Case: Zooplankton Classification

The `PolyTaxonomy` library was created to categorize zooplankton objects in marine biology.
For example, a copepod can be described as `Copepoda sex:female view:lateral`, indicating a female copepod viewed laterally.
This polyhierarchical system allows for complex queries and facilitats advanced machine learning tasks in the context of the MAZE-IPP project.

## Contributing

Contributions are welcome! Please submit issues or pull requests on the [GitHub repository](https://github.com/moi90/polytaxo).

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

For more information, refer to the [original blog post](https://www.nerdluecht.de/index.php/2023/10/23/polytaxo-using-a-polyhierarchical-taxonomy-to-describe-zooplankton-objects/).
