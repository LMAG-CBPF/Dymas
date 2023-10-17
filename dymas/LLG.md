# LLG.py Documentation

## Indices convention

$i,j,k$ or $l,m,n$ are related to positons $x,y,z$  (not used anymore)  
$x,y$ are related to positions as a list (flatten $ijk$ indices with or without a mask)  
$a, b, c, d, e$ are related to vector component in 3d space  
$u,v$ are realted to vector component in the plane locally orthogonal to $m$

## Effective field operators

Returns the magnetization derivative $\frac{dm}{dt} [x,a]$ when operated on an effective field $H_\mathrm{eff} [x,b]$.

$$\frac{dm}{dt} = (T+D) H_\mathrm{eff}$$

This operators are used to minimize $(D_\mathrm{LLG})$ or evolve ($T+D$) the magnetization 

### torque_operator_LL

Toque operator $T_\mathrm{LL}[x,a,b]$ in the Landau Lifshitz (LL) equation

$$T_\mathrm{LL} = -\gamma \frac{1}{1+\alpha^2} \, m \times$$

```python
def torque_operator_LL(alpha: npt.NDArray,
                       gamma: npt.NDArray,
                       m: npt.NDArray) -> npt.NDArray:
```

### torque_operator_LLG

Toque operator $T_\mathrm{LLG}[x,a,b]$ in the Landau Lifshitz Gilbert (LLG) equation

$$T_\mathrm{LLG} = -\gamma \, m \times$$

```python
def torque_operator_LLG(alpha: npt.NDArray,
                        gamma: npt.NDArray,
                        m: npt.NDArray) -> npt.NDArray:
```

### damping_operator_LL

Damping operator $D_\mathrm{LL}[x,a,b]$ in the Landau Lifshitz (LL) equation

$$D_\mathrm{LL} = \gamma\frac{\alpha}{1+\alpha^2}
 m \times \square \times m$$

```python
def damping_operator_LLG(alpha: npt.NDArray,
                         gamma: npt.NDArray,
                         m: npt.NDArray) -> npt.NDArray:
```

### damping_operator_LLG

Damping operator $D_{\mathrm{LLG}}[x,a,b]$ in the Landau Lifshitz Gilbert (LLG) equation

$$D_\mathrm{LLG} =
D_\mathrm{LL} + T_\mathrm{LL} - T_\mathrm{LLG} $$
$$D_\mathrm{LLG} = D_\mathrm{LL} - \alpha^2 T_\mathrm{LL} $$

```python
def damping_operator_LLG(alpha: npt.NDArray,
                         gamma: npt.NDArray,
                         m: npt.NDArray) -> npt.NDArray:
```

