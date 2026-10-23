<p align="center">
  <img src="https://raw.githubusercontent.com/nyosef1108/multilayer-perceptron-from-scratch/main/assets/logo.svg" alt="Multilayer Perceptron from scratch" width="520"/>
</p>

<p align="center">
  <a href="https://github.com/nyosef1108/multilayer-perceptron-from-scratch"><img alt="GitHub Repo" src="https://img.shields.io/badge/GitHub-Repository-blue?logo=github"></a>
  <a href="https://opensource.org/licenses/MIT"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
  <a href="https://www.python.org/downloads/"><img alt="Python 3.9+" src="https://img.shields.io/badge/python-3.9+-blue.svg"></a>
  <a href="https://numpy.org/"><img alt="NumPy" src="https://img.shields.io/badge/dependency-NumPy-blueviolet.svg"></a>
</p>

# Multilayer Perceptron (MLP) from Scratch 🧠

A lightweight, educational implementation of a feedforward neural network built entirely from scratch with **NumPy**.

It implements the core mathematics of deep learning, including **backpropagation** and **Stochastic Gradient Descent (SGD)** with data shuffling, without heavy frameworks like PyTorch or TensorFlow.

<table align="center">
  <tr>
    <td align="center" width="33%">
      <img src="https://raw.githubusercontent.com/nyosef1108/multilayer-perceptron-from-scratch/main/assets/demo1_moons.png" alt="Moons Dataset Decision Boundary" width="100%"/>
      <br/><sub>Moons</sub>
    </td>
    <td align="center" width="33%">
      <img src="https://raw.githubusercontent.com/nyosef1108/multilayer-perceptron-from-scratch/main/assets/demo2_spiral.png" alt="6-Arm Spiral Decision Boundary" width="100%"/>
      <br/><sub>Spiral</sub>
    </td>
    <td align="center" width="33%">
      <img src="https://raw.githubusercontent.com/nyosef1108/multilayer-perceptron-from-scratch/main/assets/demo3_sin_cos.png" alt="Continuous Cosine vs Interrupted Sine Boundary" width="100%"/>
      <br/><sub>Cosine vs. Sine</sub>
    </td>
  </tr>
</table>

---

## ✨ Key Features

* **🎯 Pure NumPy:** the core engine depends only on `numpy`. The demos additionally use `matplotlib` and `scikit-learn`.
* **🔄 Stochastic Gradient Descent (SGD):** data is shuffled every epoch for better convergence.
* **🏗️ Flexible Architecture:** define any number of hidden layers and neurons.
* **📉 Matrix-Based Backpropagation:** built on the four fundamental equations of backpropagation.
* **🎓 Educational Design:** clean, commented code for learning how neural networks work under the hood.

---

## 📦 Installation

Install the package from PyPI:

```bash
pip install multilayer-perceptron-from-scratch
```

To run the demo notebooks locally, install the plotting dependencies as well:

```bash
pip install matplotlib scikit-learn jupyter
```

---

## 🚀 Quick Start (XOR Example)

```python
import numpy as np
from multilayer_perceptron import Network

np.random.seed(0)  # reproducible weight initialization

# 1. Define the architecture: [Input (2), Hidden (20), Output (1)]
net = Network([2, 20, 1])

# 2. Prepare the training data (XOR)
# Format: list of tuples (input_vector, target_vector)
training_data = [
    (np.array([[0], [0]]), np.array([[0]])),
    (np.array([[0], [1]]), np.array([[1]])),
    (np.array([[1], [0]]), np.array([[1]])),
    (np.array([[1], [1]]), np.array([[0]]))
]

# 3. Train with SGD
# Parameters: data, epochs, learning_rate
print("Starting Training...")
net.SGD(training_data, epochs=3000, learning_rate=0.2)

# 4. Test the model
# feedforward returns a tuple; the first element is the network output
print("\nResults:")
for x, y in training_data:
    output, _, _ = net.feedforward(x)
    prediction = 1 if output >= 0.5 else 0
    print(f"Input: {x.flatten()} | Target: {int(y[0][0])} | Predicted: {prediction}")
```

---

## 🧮 How It Works

This section explains, step by step, what the network does and why. You do not need prior knowledge of machine learning. If you can follow multiplication and addition, you can follow everything here.

### 1. The big picture

A neural network is a function that turns numbers into other numbers. You give it **inputs** (for example the coordinates of a point), and it gives back an **output** (for example "this point belongs to class 1").

At the start the network knows nothing: its internal settings, called **weights** and **biases**, are random, so its answers are random too. **Training** is the process of fixing that, and it works like a student practicing with an answer key:

1. **Guess.** The network sees an example and makes a prediction. This is the *forward pass*.
2. **Check.** We compare the prediction with the correct answer and measure how wrong it was. This is the *cost*.
3. **Find the blame.** We work out which weights contributed most to the mistake. This is the *backward pass*, also called *backpropagation*.
4. **Correct.** We nudge every weight slightly in the direction that would have made the mistake smaller. This is the *SGD update*.

Repeat these four steps thousands of times over many examples, and the network gradually becomes accurate.

### 2. Vocabulary: what every symbol means

| Symbol | Name | Meaning in plain words |
|---|---|---|
| $x$ | input | The numbers we feed in, for example a point $(1, 0)$ |
| $y$ | target | The correct answer we want the network to produce |
| $l$ | layer index | Which layer we are talking about: $1$ is the first hidden layer, $2$ the next, and so on |
| $L$ | last layer | The index of the output layer |
| $a^{(l)}$ | activation | The list of numbers that layer $l$ outputs. By convention $a^{(0)} = x$ |
| $z^{(l)}$ | weighted input | What layer $l$ computes before the activation function is applied |
| $W^{(l)}$ | weights | A table of numbers. Entry $(j, k)$ says how strongly neuron $k$ of the previous layer influences neuron $j$ of this layer |
| $b^{(l)}$ | biases | One number per neuron. It shifts the neuron's result up or down, like a built-in threshold |
| $\sigma$ | activation function | A fixed curve that bends the result, which lets the network learn non-straight patterns |
| $C$ | cost | One number that says how wrong the network is. Zero means perfect |
| $\delta^{(l)}$ | error of layer $l$ | How sensitive the cost is to the weighted inputs of layer $l$. Think of it as "how much each neuron is to blame" |
| $\nabla_a C$ | gradient of the cost | How the cost changes when the output changes |
| $\dfrac{\partial C}{\partial W}$ | partial derivative | How much the cost changes if we change one weight by a tiny amount |
| $\odot$ | element-wise product | Multiply two lists entry by entry: $(a, b) \odot (c, d) = (ac, bd)$ |
| $\top$ | transpose | Flip a table so its rows become columns |
| $\eta$ | learning rate | The size of each correction step |
| epoch | | One full pass over all the training examples |

### 3. The forward pass: making a prediction

Each layer applies the same two lines:

$$
\begin{aligned}
z^{(l)} &= W^{(l)}\,a^{(l-1)} + b^{(l)} \\
a^{(l)} &= \sigma\!\left(z^{(l)}\right)
\end{aligned}
$$

**First line, the weighted sum.** Every neuron in layer $l$ looks at all outputs of the previous layer, multiplies each one by a weight, adds them up, and adds its bias. A large positive weight means "this input matters and pushes me up". A negative weight means "this input pushes me down". A weight near zero means "I ignore this input".

For one neuron $j$ this reads:

$$
z_j^{(l)} = \sum_k W_{jk}^{(l)}\, a_k^{(l-1)} + b_j^{(l)}
$$

The matrix notation $W^{(l)} a^{(l-1)}$ simply does this for all neurons at once, which is also why NumPy can compute it quickly.

**Second line, the activation function.** The weighted sum alone can only draw straight lines. Without the second line, stacking many layers would still collapse into one big straight-line rule, no matter how deep the network is. The activation function $\sigma$ bends the result and removes that limit. This is what allows the network to separate curved shapes such as the two moons or the spiral arms in the demos.

In the worked example below we use the **sigmoid** function:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

It squeezes any number into the range between 0 and 1: very negative values become close to 0, very positive values become close to 1, and $\sigma(0) = 0.5$. That makes its output easy to read as a probability, and it explains the rule used in the code, `prediction = 1 if output >= 0.5 else 0`.

**Shapes.** If layer $l-1$ has $n$ neurons and layer $l$ has $m$ neurons, then $W^{(l)}$ has $m$ rows and $n$ columns, and $b^{(l)}$ and $z^{(l)}$ each have $m$ entries. For `Network([2, 20, 1])` this means $W^{(1)}$ is $20 \times 2$ and $W^{(2)}$ is $1 \times 20$.

### 4. The cost: measuring how wrong we are

After the forward pass we have a prediction $a^{(L)}$ and the correct answer $y$. The cost turns the gap between them into a single number. In the worked example below we use the squared error:

$$
C = \frac{1}{2}\left(a^{(L)} - y\right)^2
$$

If the prediction equals the target, the cost is $0$. The further apart they are, the larger the cost. The squaring makes all mistakes positive, and the $\tfrac{1}{2}$ is only there to keep later formulas tidy. Training means **making this number as small as possible**.

### 5. The backward pass: finding who is to blame

We want to know, for every single weight, "if I increase this weight a little, does the cost go up or down, and by how much?" That quantity is the partial derivative $\frac{\partial C}{\partial W}$. A direct computation for each weight separately would be very slow. Backpropagation computes all of them in one sweep from the output layer back to the input, reusing results along the way. The tool behind it is the chain rule from calculus: the effect of an early weight on the cost is the product of the effects of every step between them.

There are four equations.

$$
\delta^{(L)} = \nabla_{a} C \odot \sigma'\!\left(z^{(L)}\right)
\qquad \text{(1) error at the output layer}
$$

* $\nabla_a C$ answers: "how does the cost change if the output changes?" For the squared error it equals $a^{(L)} - y$, the signed gap between prediction and target.
* $\sigma'(z^{(L)})$ answers: "how much does the output change if the weighted input changes?" For the sigmoid, $\sigma'(z) = \sigma(z)\,(1 - \sigma(z))$. It is largest in the middle of the curve and close to zero when a neuron is saturated near 0 or 1.
* Multiplying them gives $\delta^{(L)}$: how sensitive the cost is to the output layer's weighted input. It is large when the prediction is far off *and* the neuron is able to react.

$$
\delta^{(l)} = \left(\left(W^{(l+1)}\right)^{\!\top} \delta^{(l+1)}\right) \odot \sigma'\!\left(z^{(l)}\right)
\qquad \text{(2) error passed back one layer}
$$

* $\delta^{(l+1)}$ is the error of the layer ahead of us, which we already know.
* $\left(W^{(l+1)}\right)^{\top}$ sends that error backward along the same connections the signal used going forward. A neuron that fed strongly into a wrong output receives a large share of the blame.
* $\odot\, \sigma'(z^{(l)})$ scales the result by how responsive this layer's neurons are. Repeat this equation layer by layer, moving toward the input, until every layer has its $\delta$.

$$
\frac{\partial C}{\partial b^{(l)}} = \delta^{(l)}
\qquad \text{(3) gradient for the biases}
$$

A bias adds directly to $z^{(l)}$, so changing the bias changes $z^{(l)}$ one-for-one. The sensitivity of the cost to the bias is therefore exactly the layer's error.

$$
\frac{\partial C}{\partial W^{(l)}} = \delta^{(l)} \left(a^{(l-1)}\right)^{\!\top}
\qquad \text{(4) gradient for the weights}
$$

A weight multiplies one input, so its effect depends on two things: how much the neuron it feeds is to blame ($\delta^{(l)}$), and how large the input passing through it was ($a^{(l-1)}$). If the input was zero, the weight had no influence, so its gradient is zero. You will see exactly this in the worked example.

### 6. The SGD update: taking a step downhill

Imagine standing on a hilly landscape in thick fog, where the height at your position is the cost. You cannot see the valley, but you can feel the slope under your feet. The sensible strategy is to take a small step in the downhill direction and repeat. The gradient is the slope, and each update is one step:

$$
\begin{aligned}
W^{(l)} &\leftarrow W^{(l)} - \eta\,\frac{\partial C}{\partial W^{(l)}} \\
b^{(l)} &\leftarrow b^{(l)} - \eta\,\frac{\partial C}{\partial b^{(l)}}
\end{aligned}
$$

* The arrow $\leftarrow$ means "replace the old value with the new one".
* The minus sign is what makes the step go *downhill*. If the gradient is positive (raising the weight raises the cost), we lower the weight, and vice versa.
* $\eta$, the **learning rate**, sets the step length. Too small and training is painfully slow. Too large and you overshoot the valley and the cost bounces around or grows.

**Why "stochastic"?** Instead of computing the slope from the entire dataset before every single step, SGD updates the weights using small portions of the data. This makes each step cheap, and the randomness helps the network avoid getting stuck. At the start of every epoch the training data is **shuffled**, so the network never sees the examples in the same order twice.

### 7. A complete worked example, by hand

Let us run one full training step on a tiny network with all numbers visible. You can check every line with a calculator.

**Setup**

* Architecture `[2, 2, 1]`: two inputs, two hidden neurons, one output neuron.
* Activation: sigmoid. Cost: squared error. Learning rate: $\eta = 0.5$.
* Training example: input $x = (1, 0)$ with target $y = 1$ (this is one row of XOR).
* Starting weights (chosen by hand so the numbers are easy to follow):

$$
W^{(1)} = \begin{pmatrix} 0.5 & 0.2 \\ -0.3 & 0.4 \end{pmatrix},\quad
b^{(1)} = \begin{pmatrix} 0.1 \\ -0.1 \end{pmatrix},\quad
W^{(2)} = \begin{pmatrix} 0.6 & -0.5 \end{pmatrix},\quad
b^{(2)} = \begin{pmatrix} 0.2 \end{pmatrix}
$$

Values below are rounded to four decimal places.

**Step 1: forward pass, hidden layer**

$$
z^{(1)} = W^{(1)} x + b^{(1)}
= \begin{pmatrix} 0.5\cdot 1 + 0.2\cdot 0 + 0.1 \\ -0.3\cdot 1 + 0.4\cdot 0 - 0.1 \end{pmatrix}
= \begin{pmatrix} 0.6 \\ -0.4 \end{pmatrix}
$$

$$
a^{(1)} = \sigma\!\left(z^{(1)}\right) = \begin{pmatrix} \sigma(0.6) \\ \sigma(-0.4) \end{pmatrix}
= \begin{pmatrix} 0.6457 \\ 0.4013 \end{pmatrix}
$$

**Step 2: forward pass, output layer**

$$
z^{(2)} = 0.6\cdot 0.6457 + (-0.5)\cdot 0.4013 + 0.2 = 0.3867
$$

$$
a^{(2)} = \sigma(0.3867) = 0.5955
$$

The network predicts **0.5955**, but the target is **1**.

**Step 3: the cost**

$$
C = \tfrac{1}{2}\,(0.5955 - 1)^2 = \tfrac{1}{2}\,(0.4045)^2 = 0.0818
$$

**Step 4: backward pass, output layer (equation 1)**

$$
\nabla_a C = a^{(2)} - y = 0.5955 - 1 = -0.4045
$$

$$
\sigma'\!\left(z^{(2)}\right) = 0.5955\,(1 - 0.5955) = 0.2409
$$

$$
\delta^{(2)} = -0.4045 \cdot 0.2409 = -0.0974
$$

The negative sign says: "the output is too low, increasing it would reduce the cost".

**Step 5: backward pass, hidden layer (equation 2)**

First send the error back along the connections:

$$
\left(W^{(2)}\right)^{\!\top} \delta^{(2)}
= \begin{pmatrix} 0.6 \\ -0.5 \end{pmatrix} \cdot (-0.0974)
= \begin{pmatrix} -0.0585 \\ 0.0487 \end{pmatrix}
$$

Then scale by the slope of the sigmoid at each hidden neuron:

$$
\sigma'\!\left(z^{(1)}\right) = \begin{pmatrix} 0.6457\,(1-0.6457) \\ 0.4013\,(1-0.4013) \end{pmatrix}
= \begin{pmatrix} 0.2288 \\ 0.2403 \end{pmatrix}
$$

$$
\delta^{(1)} = \begin{pmatrix} -0.0585 \cdot 0.2288 \\ 0.0487 \cdot 0.2403 \end{pmatrix}
= \begin{pmatrix} -0.0134 \\ 0.0117 \end{pmatrix}
$$

**Step 6: gradients (equations 3 and 4)**

$$
\frac{\partial C}{\partial b^{(2)}} = \delta^{(2)} = -0.0974
\qquad
\frac{\partial C}{\partial W^{(2)}} = \delta^{(2)} \left(a^{(1)}\right)^{\!\top} = \begin{pmatrix} -0.0629 & -0.0391 \end{pmatrix}
$$

$$
\frac{\partial C}{\partial b^{(1)}} = \delta^{(1)} = \begin{pmatrix} -0.0134 \\ 0.0117 \end{pmatrix}
\qquad
\frac{\partial C}{\partial W^{(1)}} = \delta^{(1)} x^{\top} = \begin{pmatrix} -0.0134 & 0 \\ 0.0117 & 0 \end{pmatrix}
$$

Notice that the **second column of $\frac{\partial C}{\partial W^{(1)}}$ is zero**. That column belongs to the second input, and that input was $0$, so those weights played no part in this prediction and receive no correction.

**Step 7: SGD update with $\eta = 0.5$**

$$
W^{(2)} \leftarrow \begin{pmatrix} 0.6 + 0.5\cdot 0.0629 & -0.5 + 0.5\cdot 0.0391 \end{pmatrix} = \begin{pmatrix} 0.6315 & -0.4804 \end{pmatrix}
$$

$$
b^{(2)} \leftarrow 0.2 + 0.5\cdot 0.0974 = 0.2487
$$

$$
W^{(1)} \leftarrow \begin{pmatrix} 0.5067 & 0.2 \\ -0.3059 & 0.4 \end{pmatrix}
\qquad
b^{(1)} \leftarrow \begin{pmatrix} 0.1067 \\ -0.1059 \end{pmatrix}
$$

**Step 8: did it help?** Run the forward pass again with the new weights, using the same input:

| | Before the update | After the update |
|---|---|---|
| Output $a^{(2)}$ | 0.5955 | **0.6146** |
| Cost $C$ | 0.0818 | **0.0742** |

The output moved toward the target $1$, and the cost dropped. That is one learning step. Training with SGD repeats this thousands of times across all the examples, and each repetition makes the network a little less wrong.

### 8. Example run: XOR

XOR is a classic test because no single straight line can separate its classes (the outputs are $1$ only when the two inputs differ), so the network needs a hidden layer. The Quick Start code above trains a `[2, 20, 1]` network on it for 3000 epochs. Once training has converged, the final lines look like this:

```text
Starting Training...

Results:
Input: [0 0] | Target: 0 | Predicted: 0
Input: [0 1] | Target: 1 | Predicted: 1
Input: [1 0] | Target: 1 | Predicted: 1
Input: [1 1] | Target: 0 | Predicted: 0
```

Every prediction matches its target. Training uses random starting weights, so on an unlucky start a small network can occasionally get stuck. If that happens, change the seed or train for more epochs.

### 9. Common questions

**Why not start with all weights equal to zero?** Then every neuron in a layer computes exactly the same thing and receives exactly the same correction, so they never become different from each other. Random starting weights break this symmetry.

**What happens if the learning rate is too large?** The steps overshoot the valley. The cost may oscillate or even grow. If the cost curve in a demo looks jagged or rising, lower $\eta$.

**What happens if it is too small?** Training works but is slow, and may need many more epochs.

**How many epochs are enough?** Watch the loss curve and the accuracy in the demos. When they stop improving, more epochs will not help. The demos use 600 to 3000 epochs depending on how hard the problem is.

**Why do deeper networks help the spiral demo?** A single hidden layer combines straight-line boundaries once. Additional layers combine those combinations, which lets the network bend its decision boundary around the twisting arms.

---

## 📓 Interactive Jupyter Notebooks

The repository includes ready-to-run notebooks with non-linear classification tasks. Each demo tracks **loss** and **accuracy**, and ends with a decision-boundary plot made with `matplotlib`. Accuracy values below are measured on the training data.

### 🌕 DEMO 1: Moons Dataset Boundary

<p align="center">
  <img src="https://raw.githubusercontent.com/nyosef1108/multilayer-perceptron-from-scratch/main/assets/demo1_moons.png" alt="Moons Dataset Decision Boundary" width="600"/>
</p>

* **Description:** a classic non-linear benchmark. A hidden layer of 16 neurons separates two interleaving half-circles.
* **Key Components:** `sklearn.datasets.make_moons`, `matplotlib.pyplot` contour plotting.
* **Performance:** reaches **100% accuracy** in fewer than 1000 epochs.
* **🔗 Run in Cloud:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nyosef1108/multilayer-perceptron-from-scratch/blob/main/DEMO1.ipynb)

---

### 🌀 DEMO 2: Wide-Gap 6-Arm Spiral

<p align="center">
  <img src="https://raw.githubusercontent.com/nyosef1108/multilayer-perceptron-from-scratch/main/assets/demo2_spiral.png" alt="6-Arm Spiral Decision Boundary" width="550"/>
</p>

* **Description:** a geometric distribution that tests how well the network captures sharp, twisting transitions. A deeper architecture (`[2, 10, 10, 1]`) and a higher learning rate (η = 0.2) let the model follow the spiral arms.
* **Key Components:** a custom synthetic spiral generator with a wide center gap.
* **Performance:** separates all 6 arms and reaches **99.8%+ accuracy** by epoch 2000.
* **🔗 Run in Cloud:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nyosef1108/multilayer-perceptron-from-scratch/blob/main/DEMO2.ipynb)

---

### 📊 DEMO 3: Continuous Cosine vs. Interrupted Sine

<p align="center">
  <img src="https://raw.githubusercontent.com/nyosef1108/multilayer-perceptron-from-scratch/main/assets/demo3_sin_cos.png" alt="Continuous Cosine vs Interrupted Sine Boundary" width="650"/>
</p>

* **Description:** a phase and amplitude classification task. The network separates a continuous blue cosine wave from a red sine wave that is interrupted wherever the two curves come close, so the classes never overlap.
* **Architecture:** `[2, 32, 32, 1]`, trained for 600 epochs with η = 0.08.
* **Key Components:** a custom hybrid wave generator, `matplotlib.lines.Line2D` legends.
* **Performance:** reaches **98.6% accuracy**.
* **🔗 Run in Cloud:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nyosef1108/multilayer-perceptron-from-scratch/blob/main/DEMO3.ipynb)

---

## 📜 License

Distributed under the **MIT License**. See the [LICENSE](https://github.com/nyosef1108/multilayer-perceptron-from-scratch/blob/main/LICENSE) file for more information.

---

**Created with ❤️ by Nehorai Yosef**
