# PolyTaxo

# `polytaxo` package

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
