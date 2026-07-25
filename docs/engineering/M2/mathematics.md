# Mathematical Formulations in Chatii (M2)

## 1. Linear Transformation
$$Y = X W^T + b$$
where $X \in \mathbb{R}^{B \times T \times D_{in}}$, $W \in \mathbb{R}^{D_{out} \times D_{in}}$, and $b \in \mathbb{R}^{D_{out}}$.

---

## 2. Normalization Layers

### Layer Normalization (LayerNorm)
$$\text{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta$$
where $\mu = \frac{1}{d}\sum_{i=1}^d x_i$ and $\sigma^2 = \frac{1}{d}\sum_{i=1}^d (x_i - \mu)^2$.

### Root Mean Square Normalization (RMSNorm)
$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}} \odot \gamma$$

---

## 3. Activation Functions

### Gaussian Error Linear Unit (GELU)
$$\text{GELU}(x) = 0.5 \cdot x \cdot \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}\left(x + 0.044715 x^3\right)\right)\right)$$

### Sigmoid Linear Unit (SiLU / Swish-1)
$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

---

## 4. Rotary Position Embedding (RoPE)
Given 2D vector $(x_1, x_2)$ at position $m$ with angle $\theta$:
$$\begin{pmatrix} x_1' \\ x_2' \end{pmatrix} = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}$$
Expressed using complex numbers:
$$\mathbf{x}_{\text{complex}}' = \mathbf{x}_{\text{complex}} \cdot e^{i m \theta}$$
