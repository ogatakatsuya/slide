---
marp: true
image: https://ogatakatsuya.github.io/slide/seminar_03_06_2025/slide.png
title: GCNs
description: 中島研究室のセミナー発表資料
math: katex
---
<!-- paginate: true -->

# GCNs

### Semi Supervised Classification with Graph Convolutional Networks

MIM Lab 
Katsuya Ogata

---

<!-- _header: Agenda -->

1. Basic Knowledge of GNN
1. Introduction
1. Fast Approximate Convolutions on Graphs
1. Semi-Supervised Node Classification
1. Related Work
1. Experiments
1. Results
1. Discussion

---

<!-- _header: Basic Knowledge of GNN -->

## Basic Elements of a Graph

- **Node**  
  A vertex in the graph, e.g., a user in a social network or an atom in a molecule.

- **Edge**  
  A connection between two nodes, e.g., friendship between users or bonds between atoms.

- **Adjacency Matrix**  
  A matrix representing the connectivity of nodes. If node $i$ and node $j$ are connected, $A_{ij} = 1$; otherwise, $0$.

---

<!-- _header: Basic Knowledge of GNN -->

## Graph Spectral Theory

- Study the properties of graphs by analyzing the **eigenvalues** and **eigenvectors** of matrices associated with the graph.

- Think of these eigenvalues as a kind of "fingerprint" or "signature" of the graph. They reveal crucial structural and global properties, such as:

  - Connectivity (how well the graph is connected)
  - Bipartiteness (if the graph can be divided into two independent sets)
  - The presence of certain motifs or communities

---

<!-- _header: Basic Knowledge of GNN -->

## Graph Laplacian

- **Definition**  
  $L := D - A$

  Where:  
  - $D$ = Degree matrix (diagonal, $D_{ii}$ = degree of node $i$)  
  - $A$ = Adjacency matrix

- **Properties**  
  - Symmetric (if undirected graph)  
  - Captures the difference between a node and its neighbors

---

<!-- _header: Basic Knowledge of GNN -->

## Normalized Laplacian

- **Definition**
  $\tilde{L} := D^{-1/2} L D^{-1/2} = I - D^{-1/2} A D^{-1/2} = I -\tilde{A}${.empty}

  Where:  
  - $\tilde{A} = D^{-1/2} A D^{-1/2}$ is the normalized adjacency

- **Why normalize?**  
  - Removes scale differences due to node degrees  
  - Makes it easier to compare graphs with different structures

---

<!-- _header: Basic Knowledge of GNN -->

## Graph Fourier Transform

- **Forward transform**  
  $$ F(x) = U^T x $$

- **Inverse transform**  
  $$ F^{-1}(x) = U x $$

Where:
- $U$: matrix of eigenvectors (graph Fourier basis)
- $x$: graph signal

For more details, refer to:  
[Shuman, D.I., Narang, S.K., Frossard, P., Ortega, A., Vandergheynst, P., 2013]

---

<!-- _header: Basic Knowledge of GNN -->

## Spectral Convolution

- **Definition (using Fourier domain):**  
  $$ g * x = F^{-1}(F(g) \odot F(x)) = U (U^T g \odot U^T x) $$

Where:
- $\odot$: element-wise multiplication
---

<!-- _header: Basic Knowledge of GNN -->

## Practical Filtering

- Direct use of $U^T g$ is often impractical.

- Instead, we typically use a **learnable diagonal matrix** $g_w$:
  $$ g_w * x = U g_w U^T x $$

This simplifies the filter design and makes learning scalable.

---

<!-- _header: Basic Knowledge of GNN -->

## Summary

1. Transform the graph signal \( x \) into the spectral domain
1. Apply a filter in the spectral domain
1. Return to the original space

==![w:800](./image/graph_fourier.png)=={.image}

---

<!-- _header: Basic Knowledge of GNN -->

## What is Semi-Supervised Learning?

- **Semi-Supervised Learning**  
  A learning setup where only a small portion of data points have labels, and the rest are unlabeled.

- Goal:  
  Use both labeled and unlabeled data to improve model performance.

---

<!-- _header: Introduction -->

## Loss Functions of GNN

$$
L = L_0 + \lambda L_{\text{reg}}
$$

where:  
- $L_0$ : supervised loss over the labeled part of the graph  
- $\lambda$ : weighting factor controlling the strength of the regularization  
- $L_{reg}$: graph regularization term
---

<!-- _header: Introduction -->

## Regularization Term

$$
L_{\text{reg}} = \sum_{i, j} A_{ij} \| f(X_i) - f(X_j) \|^2 = f(X)^\top \Delta f(X)
$$

where:  
- $f(\cdot)$: neural network-like differentiable function  
- $X$: matrix of node feature vectors $X_i$  
- $A$: adjacency matrix  
- $\Delta = D - A$: unnormalized graph Laplacian

---

<!-- _header: Introduction -->

## Homophily Hypothesis

- **Homophily** refers to the tendency of connected nodes to share similar attributes or labels.  
- In graph learning, it is assumed that:
  
  **"Connected nodes are likely to belong to the same class."**

---

<!-- _header: Introduction -->

## Proposed Methods

$$
loss = L_0
$$

$$
output = f(X, A)
$$

### Where:

- $X \in \mathbb{R}^{N \times D}$: Node feature matrix,  
  where $N$ = number of nodes, $D$ = number of features  
- $A \in \mathbb{R}^{N \times N}$: Adjacency matrix  
- $f(\cdot)$: Neural network mapping features and graph structure  
- $L_0$: Supervised loss on labeled nodes

---

<!-- _header: Fast Approximate Convolutions on Graphs -->

## Spectral GNNs: Polynomial Filter Approximation

- **Sopectral Graph Convolution**: $g_w * x = U g_w U^T x$
- These models often use filters defined as a function of eigenvalues:  
  $g_w := g(\Lambda)$

- **Direct eigendecomposition is computationally expensive.**  
  To address this, the filter is expressed as a $K$-th order polynomial:
  $$
  g(\Lambda) = \sum_{k=0}^K w_k (I - \Lambda)^k
  $$

- The convolution can then be rewritten as:
  $$
  U g(\Lambda) U^T X = \sum_{k=0}^K w_k \tilde{A}^k X
  $$
  where $\tilde{A}$ is the normalized adjacency matrix.

- **Key Points:**
  - Eigenvectors and eigenvalues disappear from the final expression.
  - The operation is now a sum of powers of the normalized adjacency matrix.
  - This form is closely related to random walks and personalized PageRank.
  - Many related models have been studied based on this framework.


---

<!-- _header: Semi-Supervised Node Classification -->

---

<!-- _header: Related Work -->

---

<!-- _header: Experiments -->

---

<!-- _header: Results -->

---

<!-- _header: Discussion -->

---

<!-- _header: Conclusion -->