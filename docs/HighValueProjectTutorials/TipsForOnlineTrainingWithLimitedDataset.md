# Tips For Online Training With Limited Dataset

## Avoid using Z-Score And Min-Max Normalization

Z-Score and Min-Max normalization have this issue where the distribution changes when you use different datasets to train the same models.

As such, we would recommend to scale the values relative to other values when possible. Below, we will show you how it can be done.

### Difference-Ratio Scaling

For two values \(v_1\) and \(v_2\):

\[
x_\text{scaled} = \frac{v_1 - v_2}{v_1 + v_2}
\]

Add a small \(\epsilon\) to prevent division by zero:

\[
x_\text{scaled} = \frac{v_1 - v_2}{v_1 + v_2 + \epsilon}
\]

For vectors \(\mathbf{v} = [v_1, v_2, \dots, v_n]\) and \(\mathbf{u} = [u_1, u_2, \dots, u_n]\):

\[
\mathbf{x}_\text{scaled} = 
\frac{\mathbf{v} - \mathbf{u}}{\mathbf{v} + \mathbf{u} + \epsilon} =
\begin{bmatrix}
\frac{v_1 - u_1}{v_1 + u_1 + \epsilon} \\
\frac{v_2 - u_2}{v_2 + u_2 + \epsilon} \\
\vdots \\
\frac{v_n - u_n}{v_n + u_n + \epsilon}
\end{bmatrix}
\]
