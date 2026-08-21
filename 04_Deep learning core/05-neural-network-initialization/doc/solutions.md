# Solutions — 05 Neural Network Initialization

1. The loss would never decrease (or would only barely, in a very degenerate way). With every weight at exactly 0, every neuron in a given layer computes the exact same output (0, then whatever the bias/activation produces) for any input, and — critically — every neuron receives the exact same gradient during backprop, since the symmetry is perfect. All neurons in a layer update identically forever, so the network behaves as if each layer had only 1 unique neuron, drastically limiting what it can represent, no matter how long you train.

2. Results vary slightly by random seed, but expect the `naive_small` Tanh network's activation std to shrink by roughly 2-4+ orders of magnitude from depth 0 to depth 14 (e.g. from ~0.5 down to ~0.001 or smaller) — a dramatic, visually obvious collapse on the log-scale plot.

3. At `NUM_LAYERS=4`, the effect is usually far less dramatic — activation std for `naive_small`/`naive_large` may drift noticeably but won't collapse/explode nearly as catastrophically as at 15 layers, since there are far fewer multiplicative compounding steps. This demonstrates that initialization scale matters increasingly as depth increases — a shallow, 2-4 layer network can often tolerate a mediocre initialization scheme, while a genuinely deep network (which is most of what "deep learning" means today) cannot.

4. Using Xavier initialization on a ReLU network typically shows activation std still shrinking somewhat with depth, though less catastrophically than `naive_small` — because Xavier's variance target `2/(fan_in+fan_out)` doesn't account for ReLU zeroing out (on average) half of its inputs, so the *effective* variance passed forward through each ReLU layer ends up somewhat too small compared to what He's `2/fan_in` target correctly compensates for. Over many layers this mismatch compounds, even though it's much milder than the `naive_small` case shown in the theory.

5. Example clipping snippet:
```python
def clip_grad(grad, max_norm=1.0):
    norm = np.linalg.norm(grad)
    if norm > max_norm:
        grad = grad * (max_norm / norm)
    return grad
```
Applying this inside the backward loop (clipping `grad` before each `grad @ weights[i].T` step) would have the biggest effect at the deepest/output-adjacent layers under `naive_large`, precisely where the unclipped gradient magnitude is largest before it even has a chance to compound further on its way back to earlier layers.

6. Biases don't cause the same symmetry problem weights do, because even if all biases start at 0, each neuron in a layer still receives *different* weighted inputs (assuming the weights themselves are randomly initialized) — the weights alone are enough to break symmetry between neurons. Biases only shift the activation function's input, and a zero starting bias is simply a neutral, unbiased starting point; there's no risk of neurons collapsing into identical behavior purely because their biases match, as long as their weights differ.
