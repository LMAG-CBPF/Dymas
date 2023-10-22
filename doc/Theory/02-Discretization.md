# Discretization

The dynamical operator is a continuous operator. 
In order to apply numerical algorithms to it, a finite matrix representation is initially required. 
To achieve this, a basis of functions $\{b_i\}$, where each $b_i: \mathbb{R}^3 \rightarrow \mathbb{C}^3$, must be carefully selected. 
Subsequently, the elements $D_{ij}$ of the matrix representation of the dynamical operator can be computed as follows:

$$D_{ij} = \langle b_i, D \,b_j\rangle$$

## References

## Authors and License

Authors: D.E. Gonzalez-Chavez, G. P. Zamudio

This work in licensed under the Creative Commons Attribution 4.0 International license ([CC BY 4.0](http://creativecommons.org/licenses/by/4.0/.))