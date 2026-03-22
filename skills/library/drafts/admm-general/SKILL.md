---
name: admm-general
description: Reusable guidance for admm-style workflows.
category: knowledge_general
agent-scope: General
user-invocable: false
disable-model-invocation: false
---

# General pattern: admm

## When to use
- Plan relies on `admm` or a closely related method.

## Guidance
- **Forward Model:**
- $$\mathbf{f} = \mathbf{A}\mathbf{x} + \mathbf{b} + \mathbf{n}$$
- - $\mathbf{f} \in \mathbb{R}^{128 \times 128}$: observed fluorescence microscopy image
- - $\mathbf{A}$: convolution operator with the Point Spread Function (PSF), i.e., $\mathbf{A}\mathbf{x} = \text{PSF} * \mathbf{x}$

## Constraints
- Only apply when the forward model and data assumptions match.
- Keep as draft until multiple successful trajectories corroborate.
