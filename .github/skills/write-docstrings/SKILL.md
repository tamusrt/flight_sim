---
name: write-docstrings
description: "Write Python docstrings in the project's Google style. Use when adding or editing any function, method, class, or module."
---

# Write Docstrings

Every function, method, class, and module in this project carries a docstring.
Each one is either a **single-line summary** or a **full Google-style block** —
nothing in between, and no other format.

## One-liners

Use a one-liner when the signature already explains the arguments and the return
value. Most classes, dataclasses, and tests fall here.

Do not restate in a subclass what its base class docstring already says. A reader
who wants to know what `UnitChecked` does will read `UnitChecked`; repeating the
explanation in every subclass duplicates one fact across many files, and the
copies drift apart.

```python
def zero_vector(units: str) -> Vector:
    """Return a zero-valued three-dimensional vector quantity."""
```

## Google style

Use the full block as soon as an argument, return value, or exception needs
explanation. Document every parameter, and give each one its type in
parentheses. Omit a section when it does not apply — a function that raises
nothing has no `Raises:` section.

```python
def scalar(magnitude: float, units: str) -> Scalar:
    """Build a scalar quantity from a magnitude and a unit string.

    Extended description, when the summary line is not enough on its own.

    Args:
        magnitude (float): Numeric value of the measurement.
        units (str): Pint unit expression, such as "kg" or "m/s**2".

    Returns:
        Scalar: The magnitude tagged with the given units.

    Raises:
        DimensionalityError: If the units are not a recognised expression.
    """
```

The full set of sections, in order:

```
{{summaryPlaceholder}}

{{extendedSummaryPlaceholder}}

Args:
    {{var}} ({{typePlaceholder}}): {{descriptionPlaceholder}}
    {{var}} ({{typePlaceholder}}, optional): {{descriptionPlaceholder}}. Defaults to {{default}}.

Raises:
    {{type}}: {{descriptionPlaceholder}}

Returns:
    {{typePlaceholder}}: {{descriptionPlaceholder}}

Yields:
    {{typePlaceholder}}: {{descriptionPlaceholder}}
```

## Never describe the code's history

A docstring describes what the code does **now**. It must never contrast the
current code with a previous version or a rejected alternative. Ban phrases like
"deliberately not a `RocketState`", "reusing X here would...", "this replaces...",
"unlike the old version", "now uses...".

Someone reading the docstring is looking at the code as it exists; a contrast
with code they will never see only makes them reconstruct it. Design history
belongs in the commit message and the pull request, where it is dated and
attributed.

```python
# Wrong — describes a rejected alternative.
"""Rate of change of a RocketState.

Deliberately not a RocketState: reusing RocketState here would store an
acceleration in a field declared as a velocity.
"""

# Right — states the positive fact.
"""Rate of change of a RocketState with respect to time.

Each field is the time derivative of the RocketState field it integrates
into: velocity integrates into position, acceleration into velocity.
"""
```

## Summary lines

Write the summary in the imperative ("Return the ...", "Advance the ...") and
keep it on one line, ending with a period.
