# Vision-Language Models

*A rigorous but accessible guide to the concepts underlying the Vision-Language Bootcamp notebooks — CLIP, zero-shot classification, and instruction-tuned VLMs (Qwen2.5-VL) for captioning and detection.*

---

## How to use this document

Parts **1–9** are the original theory companion. **Parts 10–16** — they go deeper into the math, cover architecture families you'll be asked to compare against, add evaluation methodology.



---

## Part 1 — The Idea That Makes Everything Else Possible: Embeddings

### 1.1 The core problem

A computer cannot compare "a photo of a dog" to the word "dog" directly — one is a grid of pixel intensities, the other is a string of characters. Before any comparison can happen, both need to be converted into the *same kind of object*. That object is a **vector**: a fixed-length list of numbers.

An embedding is a learned function that maps an input (an image, a sentence, a word) to a point in a high-dimensional numeric space — typically 256 to 1024 numbers — such that **inputs with similar meaning end up at nearby points**, and unrelated inputs end up far apart.

Think of it like a very elaborate coordinate system. On a 2D map, "close together" means "geographically near." In embedding space, "close together" means "semantically similar." The number of dimensions (512 in the CLIP notebooks) just gives the model more axes along which to encode distinctions — color, shape, category, context, style, and thousands of other features nobody explicitly labeled.

**Critical point:** nobody hand-designs these coordinates. The model *learns* where to place each input by being trained on millions of examples, adjusting its internal parameters until the geometry of the space reflects real-world semantic relationships.

### 1.2 Why vectors, specifically?

Vectors support two operations that turn out to be exactly what's needed for measuring meaning:

- **Distance** — how far apart are two points? (Euclidean distance)
- **Direction / angle** — do two vectors point the same way, regardless of length? (cosine similarity — see Part 3)

Both operations are cheap to compute even for very high-dimensional vectors, which is why embeddings scale to billions of comparisons in production search and recommendation systems.

### 1.3 Where do embeddings come from?

They are the *byproduct* of training a neural network to do some task. In CLIP's case, the task is: "given an image and a caption, decide whether they belong together." To get good at that task, the network is forced to develop internal representations (embeddings) that capture meaning — because there's no other way to solve the task well. This is the central insight of representation learning: **you don't train a model to produce good embeddings directly; you train it to solve a task that is only solvable by producing good embeddings as a side effect.**

---

## Part 2 — Turning Raw Inputs Into Numbers: Tokenization and Patching

Before any embedding can be computed, images and text need to be broken into discrete pieces the model can consume.

### 2.1 Text: tokenization

A **tokenizer** breaks a sentence into sub-word units (tokens) and maps each to an integer ID from a fixed vocabulary. "unbelievable" might become `["un", "believ", "able"]`. Sub-word tokenization (used by CLIP's tokenizer and Qwen's) strikes a balance: common words stay as single tokens for efficiency, while rare or novel words are decomposed into familiar pieces so the model is never completely stuck on an unseen word.

`padding=True` in the notebooks exists because sentences have different lengths, but neural networks process inputs in fixed-size batches — shorter sequences are padded with a special token so every sequence in a batch has equal length, and an accompanying attention mask tells the model to ignore the padding.

### 2.2 Images: patching

A Vision Transformer (**ViT**) — the image encoder inside CLIP — does not look at an image pixel by pixel. Instead it slices the image into a grid of fixed-size squares (patches), commonly 32×32 pixels (hence "patch32" in `clip-vit-base-patch32`). Each patch is flattened into a vector and linearly projected into the model's embedding dimension, exactly the way a word token is projected into an embedding. From the model's point of view, **an image is just a sequence of "visual tokens,"** structurally identical to a sequence of word tokens. This unification is precisely what allows a single Transformer architecture to process both modalities.

A learned **position embedding** is added to each patch so the model knows *where* in the image the patch came from — otherwise a Transformer, which processes tokens without inherent order, would treat a shuffled image identically to the original.

---

## Part 3 — The Transformer and Self-Attention (in plain terms)

Both CLIP's image and text encoders, and the Qwen2.5-VL backbone, are built from Transformer blocks. You don't need to derive the math to use these models, but understanding the mechanism demystifies everything downstream.

### 3.1 The problem attention solves

Consider the sentence "The animal didn't cross the street because *it* was too tired." To understand what "it" refers to, a model needs to relate a word to *other words anywhere else in the sequence*, not just its immediate neighbors. **Self-attention** lets every token look at every other token and decide, dynamically, how much to weight each one when updating its own representation.

### 3.2 The mechanism, briefly

Each token produces three vectors: a **Query** (what am I looking for?), a **Key** (what do I contain?), and a **Value** (what information do I offer?). A token's updated representation is a weighted sum of every other token's Value vector, where the weights come from comparing that token's Query against every other token's Key (via dot product, then softmax-normalized). Tokens whose Keys strongly match the Query contribute more.

Stack many of these attention layers (with additional feed-forward layers in between), and the network builds up increasingly abstract representations — early layers might capture edges or syntax, later layers capture objects or semantics.

**Why this matters for the bootcamp:** the image encoder uses self-attention *between image patches* (so the model can relate a paw in one patch to a tail in another), and Qwen2.5-VL's language model uses self-attention *across both image and text tokens together*, which is exactly what allows it to answer "what color is the object on the left?" — attending jointly across modalities.

---

## Part 4 — CLIP: Contrastive Language-Image Pretraining

### 4.1 The architecture

CLIP consists of two independent encoders:

- A **text encoder** (a Transformer) that maps a caption to a vector.
- An **image encoder** (a ViT, or in some variants a ResNet) that maps an image to a vector.

Both output vectors of the same dimensionality (512 for `clip-vit-base-patch32`) and are projected into a **shared embedding space** — the crucial design choice that makes cross-modal comparison possible at all. Critically, the two encoders never directly interact with each other's internals; they only meet at the end, when their output vectors are compared.

### 4.2 The training objective: contrastive learning

CLIP was trained on ~400 million (image, caption) pairs scraped from the internet. For each training batch of *N* pairs, CLIP computes the cosine similarity between *every* image and *every* caption in the batch, producing an *N × N* similarity matrix. The diagonal of this matrix holds the *true* pairs (image *i* with its own caption *i*); every off-diagonal entry is a *negative* — a mismatched pair.

The training objective (a symmetric cross-entropy loss, closely related to **InfoNCE**) pushes the model to:

- **Maximize** similarity along the diagonal (true pairs pulled together in embedding space).
- **Minimize** similarity everywhere else (mismatched pairs pushed apart).

This is why it's called *contrastive*: the model doesn't just learn "what a dog looks like" in isolation — it learns by *contrasting* the correct caption against many incorrect ones, batch after batch, at massive scale. A **learnable temperature parameter** controls how sharply the similarity scores are scaled before the loss is applied — too flat and the model can't confidently distinguish close negatives, too sharp and training becomes unstable.

### 4.3 Cosine similarity, formally

Given two vectors **a** and **b**, cosine similarity is:

$$\cos(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}}{\lVert \mathbf{a} \rVert \, \lVert \mathbf{b} \rVert}$$

The numerator (dot product) measures overall alignment; the denominator normalizes by each vector's length, so the result depends **only on direction, not magnitude**. This matters because embedding *magnitude* often just reflects how "generic" or "confident" a representation is, which is not the signal we want — we want to know if two things point the *same way* in meaning-space. Cosine similarity ranges from −1 (opposite) through 0 (unrelated/orthogonal) to +1 (identical direction).

**Why not Euclidean distance instead?** Euclidean distance is sensitive to vector length, which can vary for reasons unrelated to semantics (e.g., how "typical" an example is). Cosine similarity sidesteps this, which is why nearly every embedding-based system (search, recommendation, retrieval) defaults to it.

### 4.4 Why this makes zero-shot classification possible

Once you have an image encoder and a text encoder in a shared space, "classification" stops being a fixed, closed operation and becomes an open-ended *retrieval* problem:

1. Encode the image once → one vector.
2. Encode each candidate label as a sentence, typically wrapped in a template like `"a photo of a {label}"` (this templating measurably improves accuracy over a bare word, because it matches the style of captions CLIP was trained on — a phenomenon called **prompt engineering**, discussed further in Part 6).
3. Compute cosine similarity between the image vector and every label vector.
4. The label with the highest similarity is the prediction.

No classification head was ever trained for "bird" or "car" — CLIP was never shown labeled examples of these categories in the traditional supervised-learning sense. It works because "bird" and "car" already exist as *directions* in the same space the image lives in, learned as a side effect of the contrastive objective. This is the literal meaning of **zero-shot**: zero labeled training examples for the specific classes being predicted at inference time. You can add or remove candidate classes at will, with no retraining — the entire "model" for a new classification task is just a new list of text strings.

### 4.5 What CLIP is *not*

It's worth being precise about CLIP's limits, because they motivate Part 5:

- CLIP produces a **similarity score**, not a caption. It cannot generate free-form text describing an image.
- It has no notion of spatial reasoning beyond what's implicitly encoded in the patch grid — it cannot natively point to *where* an object is (no bounding boxes).
- It struggles with counting, fine-grained attribute binding ("the red cup, not the blue one"), and compositional reasoning ("a cat chasing a dog" vs. "a dog chasing a cat") — because the contrastive objective rewards *overall* similarity, not compositional structure.

---

## Part 5 — From Embeddings to Reasoning: Instruction-Tuned VLMs (Qwen2.5-VL)

### 5.1 Why CLIP alone can't caption or detect

Captioning and detection are **generative** tasks — the model must produce novel sequences of text (a sentence, or a structured list of bounding boxes) rather than pick the best match from a fixed candidate list. This requires an entirely different capability: a **language model decoder** that generates tokens one at a time, conditioned on everything that came before — including, now, the image.

### 5.2 The architecture pattern

Modern instruction-tuned VLMs like Qwen2.5-VL follow a now-standard three-part pattern:

1. **Vision encoder** — a ViT (conceptually similar to CLIP's, though trained and adapted specifically for this pipeline) that turns the image into a sequence of visual tokens. Qwen2.5-VL notably uses a **dynamic/native resolution** encoder — rather than always resizing to one fixed size, it can process images at (within limits) their native resolution and aspect ratio, producing more or fewer visual tokens depending on image complexity, which materially improves performance on tasks like detection where fine spatial detail matters.
2. **A connector/projector** — a small module (often just one or two layers) that maps the vision encoder's output into the same embedding dimensionality the language model expects, so visual tokens and text tokens can be freely mixed in one sequence.
3. **A large language model (LLM) decoder** — the same type of autoregressive Transformer that powers text-only chat models, except its input sequence now interleaves visual tokens and text tokens. Self-attention (Part 3) lets every text token attend to every visual token and vice versa, which is the mechanism that allows a question like "what color is the car?" to be answered by attending specifically to the car's patches.

This is fundamentally an **early/mid-fusion** design — vision and language don't stay in separate towers meeting only at the end (as CLIP does); they're fused inside the same Transformer stack and processed jointly through many layers, which is what unlocks compositional, multi-step reasoning about an image rather than a single similarity score.

### 5.3 Autoregressive generation

The LLM decoder generates output one token at a time: at each step, it predicts a probability distribution over the entire vocabulary for "what comes next," samples (or greedily picks) a token, appends it to the sequence, and repeats — now conditioning on its own previous output as well as the original image and prompt. This is why VLM output is a *sequence*, not a fixed-size prediction: a caption can be one sentence or five, and a detection result can list one object or twenty, because generation simply continues until the model produces an end-of-sequence signal.

### 5.4 Instruction tuning and the chat template

A raw, pretrained language model just predicts "plausible next text" — it has no inherent concept of following an instruction or playing the role of a helpful assistant. **Instruction tuning** is an additional training stage (typically supervised fine-tuning on curated instruction/response pairs, often followed by reinforcement-learning-based alignment) that teaches the model to interpret input as a structured conversation with distinct roles:

- **System** — sets the assistant's behavior/persona/constraints for the whole conversation (e.g., "You are an object detector. Output must be valid JSON.").
- **User** — the human's instruction or question, which may include images.
- **Assistant** — the model's response, which the model itself is trained to generate in this exact voice.

A **chat template** is simply the exact string formatting (special tokens, delimiters, role markers) the model was trained to expect. `processor.apply_chat_template(...)` in the notebooks isn't cosmetic — feeding the model text in a different format than it saw during instruction tuning measurably degrades output quality, because the model has learned very specific statistical patterns around those role boundaries.

---

## Part 6 — Prompt Engineering as an Interface, Not a Trick

Once a model can follow instructions, *how you phrase the instruction* becomes a legitimate lever on behavior — this is prompt engineering, and it applies differently at each stage of the bootcamp:

- **For CLIP (Part 4.4):** prompt templates like `"a photo of a {label}"` narrow the gap between how a class name is phrased and how CLIP's training captions were phrased, improving similarity-score reliability.
- **For Qwen2.5-VL detection:** the system prompt in the notebooks explicitly constrains the *output format* — `{'bbox_2d': [x1, y1, x2, y2], 'label': 'class'}` — turning an open-ended generative task into something machine-parseable. This works because instruction-tuned models are demonstrably good at following explicit formatting constraints when they're stated clearly and are consistent with patterns seen during training (JSON is extremely common in instruction-tuning data).
- **For captioning:** the phrasing and specificity of the user prompt (e.g., "describe this image in one sentence" vs. "list every object and its position") directly shapes the generation, since the prompt is part of the conditioning context the decoder attends to at every generation step.

The common thread: prompting works because it is *steering a conditional probability distribution*, not because the model is "understanding intent" in some mystical sense. Every word in the prompt shifts the distribution over likely next tokens.

---

## Part 7 — Object Detection as Language Generation

This is the least intuitive idea in the bootcamp, so it's worth stating plainly: **Qwen2.5-VL performs object detection without any specialized detection architecture** (no anchor boxes, no region proposal network, no non-max suppression module — the machinery classical detectors like Faster R-CNN or YOLO rely on). Instead, detection is reframed entirely as *text generation*: the model is asked to produce a string that happens to be valid JSON encoding coordinates and labels.

This works because:

1. The vision encoder retains enough spatial information (via patch position embeddings and, in Qwen2.5-VL, native-resolution processing) that spatial questions are answerable from the visual tokens.
2. Large-scale instruction tuning has specifically included examples that teach the model the *convention* of expressing spatial location as `[x1, y1, x2, y2]` coordinate text.
3. Because it's fundamentally a language model, it generalizes to object categories and phrasings never explicitly seen in detection-labeled training data — the same zero-shot generalization property that made CLIP powerful, now extended to a structured generative task. This is why the notebooks can detect arbitrary categories (e.g., "elephant") without any detection-specific fine-tuning on that class.

The trade-off: because output is generated token-by-token as text, it is **not guaranteed to be well-formed** (hence the notebooks' JSON-repair utility functions patching malformed output) and coordinate precision is generally lower than a purpose-built detector, since the model has no explicit geometric loss function — it is only ever optimized to predict plausible next tokens.

---

## Part 8 — Putting It Together: Why the Bootcamp Is Sequenced This Way

| Notebook | Core mechanism | What it teaches |
|---|---|---|
| 1. CLIP intro | Shared embedding space + cosine similarity | The foundational idea: meaning as geometry |
| 2. CLIP classification | Contrastive pretraining → zero-shot retrieval | Similarity scores can *become* a classifier with no retraining |
| 3. Qwen captioning | Fused vision-language Transformer + autoregressive decoding | Generation requires fusion, not just comparison |
| 4. Qwen detection | Instruction-following + structured-output prompting | Spatial reasoning can be expressed as constrained text generation |

Each notebook is a strictly larger capability building on the same two primitives established in Part 1–3: **turn everything into vectors, and let attention relate those vectors to each other.** Everything else — contrastive loss, chat templates, JSON-formatted detection — is a different way of *using* those two primitives for a specific task.

---

## Part 9 — Known Limitations and Failure Modes (worth knowing before you trust output)

- **Hallucination:** generative VLMs can describe objects, attributes, or relationships that are not actually present in the image — the decoder is optimized for plausible text, not verified truth.
- **Training data bias:** both CLIP (400M scraped image-caption pairs) and Qwen2.5-VL's instruction data reflect the biases, gaps, and errors of their source data — performance is uneven across cultures, object categories, and image styles underrepresented in training.
- **Prompt sensitivity:** small rewordings can shift output meaningfully, especially for CLIP's class-template phrasing and Qwen's system-prompt formatting constraints.
- **Coordinate imprecision:** as discussed in Part 7, generated bounding boxes have no geometric guarantee of accuracy, unlike purpose-trained detectors evaluated with IoU-based losses.
- **Compute cost:** the fused, generative architecture in Part 5 is substantially more expensive per inference than CLIP's dual-encoder similarity lookup, which is why CLIP remains the default choice for large-scale retrieval, while VLMs like Qwen2.5-VL are reserved for tasks that genuinely require reasoning or generation.

---

## Part 10 — The Full Attention Equation, and Why Each Piece Is There

Part 3 gave you attention in prose. A committee may ask you to write or interpret the actual formula, so here it is, term by term.

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- **Q, K, V** are matrices, not single vectors — one row per token. If your sequence has *n* tokens and each Query/Key has dimension *d_k*, then *Q* and *K* are *n × d_k*. *V* is *n × d_v* (the value dimension can differ from the key dimension, though in practice it's usually the same).
- **QKᵀ** computes, in one matrix multiplication, the dot product of *every* Query against *every* Key — an *n × n* matrix of raw compatibility scores. Row *i*, column *j* answers: "how relevant is token *j* to token *i*'s question?"
- **Dividing by √d_k** is the *scaled* in "scaled dot-product attention." Without it, as *d_k* grows, dot products grow in magnitude (they're a sum of *d_k* terms), which pushes the softmax into a regime where its gradient is nearly zero (saturated) — training stalls. Scaling by √d_k keeps the variance of the dot products roughly constant regardless of dimensionality, so softmax stays in a well-behaved range.
- **softmax** turns each row of raw scores into a probability distribution (non-negative, sums to 1) — this is what makes the operation a *weighted average* of Values rather than an unbounded sum.
- **Multiplying by V** produces the actual output: for each token, a weighted blend of every other token's Value vector, weighted by how much attention that token paid to each other token.

### 10.1 Multi-head attention

A single attention operation forces the model to compress *all* relational information (syntax, coreference, position, semantics...) into one set of weights. **Multi-head attention** runs several independent attention operations in parallel, each with its own learned Q/K/V projection matrices, on lower-dimensional slices of the embedding (e.g., 8 heads of dimension 64 instead of 1 head of dimension 512). Each head is free to specialize — one might learn to track adjacent-word syntax, another long-range coreference, another (in the vision case) spatial adjacency between patches. The heads' outputs are concatenated and linearly projected back to the model dimension. This is why ViT and Transformer configs report a "number of heads" hyperparameter — it's not more attention, it's *more independent relational subspaces computed at the same cost*.

### 10.2 Residual connections and layer normalization

Every attention block and feed-forward block in a Transformer is wrapped as:

$$x_{\text{out}} = x_{\text{in}} + \text{Sublayer}(\text{LayerNorm}(x_{\text{in}}))$$

- **Residual (skip) connection** (the `x_in +`) lets gradients flow directly backward through many stacked layers without vanishing, and lets each layer learn a *refinement* of its input rather than having to reconstruct it from scratch. This is what makes it feasible to stack dozens of Transformer blocks.
- **LayerNorm** rescales each token's activations to have consistent mean/variance before they hit the next sublayer, which stabilizes training — without it, activation magnitudes can drift or explode as they pass through many layers.

Committees sometimes probe here because it's easy to *use* Transformers without ever having to explain why they're trainable at depth; residuals + normalization are the actual answer.

### 10.3 The feed-forward sublayer

Interleaved with attention layers is a position-wise feed-forward network (FFN) — typically two linear layers with a nonlinearity (GELU in most modern Transformers) between them, applied identically and independently to every token's vector. Attention *mixes information across tokens*; the FFN then *processes each token's mixed representation individually*, typically expanding to a much higher intermediate dimension (e.g., 4× the model dimension) before projecting back down. Roughly: attention decides *what to look at*, the FFN decides *what to do with what was seen*.

---

## Part 11 — Positional Information, Formally

A Transformer's attention operation is inherently **permutation-invariant** — swap the order of the input tokens and, absent positional information, the set of attention outputs is identical (only their arrangement changes). This is a real problem: "dog bites man" and "man bites dog" would be indistinguishable. Positional encoding is the fix.

### 11.1 Sinusoidal (fixed) positional encoding

The original Transformer paper used a fixed, non-learned scheme:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \qquad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

Each dimension pair oscillates at a different frequency, so every position gets a unique fingerprint, and — usefully — the encoding for position *pos + k* can be expressed as a linear function of the encoding at *pos*, which theoretically helps the model generalize to relative positions and to sequence lengths not seen during training.

### 11.2 Learned positional embeddings (what ViT and CLIP actually use)

Rather than a fixed formula, a learned position embedding is simply a trainable vector per position index, added to the corresponding token/patch embedding, and updated by gradient descent like any other parameter. This is what's referenced in Part 2.2 for ViT patches. It's simpler and empirically competitive with sinusoidal encodings for vision, but it has a real limitation: it's tied to a fixed grid size (a fixed number of patches), which is exactly the constraint Qwen2.5-VL's dynamic-resolution encoder had to solve around — commonly via **2D rotary position embeddings (RoPE)**, which encode relative patch position through a rotation applied to Query/Key vectors rather than through an added embedding, making them naturally adaptable to variable numbers of patches and variable image aspect ratios.

### 11.3 Why this matters 

If asked "how does the model know where a patch/word is," the complete answer has three parts: (1) attention itself is order-blind, so (2) position information must be injected explicitly, and (3) the *specific* mechanism (learned vs. sinusoidal vs. rotary) is a design choice with different generalization trade-offs — rotary/relative schemes generalize better to variable-length or variable-resolution input, which is precisely why Qwen2.5-VL needs something more flexible than CLIP's fixed learned grid.

---

## Part 12 — The Contrastive Loss, Written Out

Part 4.2 described InfoNCE in prose. The formal form, for image *i* matched with its caption in a batch of *N* pairs:

$$\mathcal{L}_{i \to t} = -\log \frac{\exp(\text{sim}(I_i, T_i)/\tau)}{\sum_{j=1}^{N} \exp(\text{sim}(I_i, T_j)/\tau)}$$

- **sim(·,·)** is cosine similarity (Part 4.3).
- **τ (tau)** is the learnable temperature from Part 4.2 — note it appears as a *divisor* here, so a smaller τ makes the softmax sharper (peakier), amplifying the gap between the true pair and the negatives; too small and gradients become unstable, too large and the loss can't discriminate close negatives.
- The denominator sums similarity over **all N candidate captions in the batch**, including the correct one — this is a softmax over "which caption matches this image," and the loss is the negative log-probability assigned to the correct match. This is exactly the same cross-entropy structure used in ordinary classification; the "classes" here are just "which of the other items in this batch is the true pair," which is why it's called *in-batch* negative sampling.

CLIP's actual loss is **symmetric**: the same expression is computed in the other direction (caption *i* → most similar image, $\mathcal{L}_{t \to i}$) and the two are averaged. This matters because $\mathcal{L}_{i \to t}$ alone only shapes the image encoder's incentive to be discriminative against other captions in the batch; the symmetric term does the equivalent for the text encoder, so both encoders are pushed toward a mutually consistent shared space rather than one encoder doing all the work.

### 12.1 Why batch size matters so much for contrastive learning

Because negatives are drawn from *other items in the same batch*, a larger batch means more, and typically harder, negatives per training step — this is why CLIP was originally trained with very large batches (32,768), and why naively reproducing CLIP-style training with small batches on a single GPU underperforms: the model simply doesn't see enough contrastive pressure per step to learn a well-separated space. Methods like MoCo (a **momentum encoder** with a large external memory bank of negatives) and memory-efficient loss implementations (e.g., SigLIP's per-pair sigmoid loss instead of a batch-softmax) exist specifically to decouple negative-sample count from GPU memory/batch-size constraints — a good answer if asked "what would you change to train this on limited compute."

---

## Part 13 — Architecture Family Comparison (a defense favorite: "why not use X instead?")

| Model | Fusion point | Objective | Can generate text? | Can localize objects? | Typical use |
|---|---|---|---|---|---|
| **CLIP** | Late (encoders meet only at similarity score) | Contrastive (InfoNCE) | No | No (without add-ons) | Zero-shot classification, retrieval |
| **BLIP-2** | Mid (a lightweight Q-Former bridges a *frozen* vision encoder and a *frozen* LLM) | Contrastive + generative, staged | Yes | Limited | Captioning, VQA, efficient fine-tuning |
| **LLaVA** | Early/mid (a simple linear or MLP projector feeds visual tokens directly into an LLM's input sequence, both typically fine-tuned/instruction-tuned together) | Instruction-tuning on generated image-instruction data | Yes | Limited (no native grounding) | General visual chat/VQA |
| **Qwen2.5-VL** | Early/mid, with native-resolution ViT and window attention for efficiency | Instruction tuning incl. explicit grounding/detection data | Yes | Yes (bounding boxes as generated text) | Captioning, VQA, detection, document/UI understanding |
| **Faster R-CNN / YOLO (classical detectors)** | N/A — vision-only, no language | Supervised, geometric losses (IoU/regression + classification) | No | Yes, natively, with geometric guarantees | Fixed-category, high-precision detection |

Key defensible points this table supports:

- **CLIP vs. Qwen2.5-VL is not "old vs. new," it's "different objective for a different task."** CLIP's late fusion is *correct* for retrieval-at-scale (you can pre-compute and index millions of image embeddings once, independent of any query text), which a fused architecture cannot do efficiently — every fused-model query requires a full forward pass conditioned on that specific text. If asked "why not just use Qwen2.5-VL for everything," compute cost and lack of a pre-computable index are the answer (see Part 9's cost note).
- **BLIP-2's frozen-encoder strategy** is worth knowing about specifically because it's the clearest illustration of the point in Part 5.2's "connector" — it demonstrates that a small trainable bridge module, with both large components (vision encoder, LLM) kept frozen, can be sufficient to align modalities, which is far cheaper than full fine-tuning.
- **Classical detectors vs. Qwen2.5-VL-as-detector** is the Part 7 trade-off restated: geometric loss functions (IoU, smooth-L1 on box coordinates) give classical detectors a training signal directly optimized for localization accuracy; a language-modeling loss (next-token cross-entropy) never directly penalizes "how far off was this coordinate," only "was this token likely" — which is the root cause of Part 9's coordinate-imprecision limitation.

---

## Part 14 — Evaluation Methodology (how you'd defend your *results*, not just your *architecture*)

A thesis defense will often probe whether you understand *why* you used a particular metric, not just whether you reported one.

### 14.1 Zero-shot classification

- **Top-1 / Top-k accuracy** — did the correct label have the single highest similarity score (top-1), or was it among the *k* highest (top-k)? Top-5 is commonly reported because it's more forgiving of near-miss confusions between visually/semantically similar classes (e.g., "leopard" vs. "jaguar").
- Report accuracy **per prompt template**, not just once — Part 6 established that CLIP is prompt-sensitive; a single-number accuracy without stating the template(s) used is not a reproducible or complete result. Prompt-ensembling (averaging text embeddings across several templates, e.g. "a photo of a {}", "a blurry photo of a {}") is a standard technique to reduce this sensitivity and is worth mentioning if your pipeline used or could have used it.

### 14.2 Captioning

- **BLEU / ROUGE** measure n-gram overlap with reference captions — cheap to compute but penalize semantically correct captions that are phrased differently from the reference, and don't penalize hallucinated details that happen to share n-grams with the reference.
- **CIDEr** weights n-gram matches by TF-IDF, so it rewards descriptive terms specific to the image over generic ones ("a", "the", "photo") — generally considered a better fit for captioning than raw BLEU.
- **CLIPScore** (image–candidate-caption cosine similarity, using CLIP itself) is a *reference-free* metric — it doesn't need a human-written ground-truth caption at all, only the image, which is valuable when references are scarce, but it inherits CLIP's own blind spots (Part 4.5): it won't reliably penalize counting errors or compositional mistakes since CLIP itself struggles with those.
- **Human/LLM-judged factuality checks** are increasingly used to specifically catch hallucination (Part 9), which n-gram metrics cannot detect by construction — a caption can hallucinate an object and still score well on BLEU/CIDEr if the sentence structure otherwise overlaps with the reference.

### 14.3 Detection

- **IoU (Intersection over Union)** — the overlap between a predicted box and the ground-truth box, divided by their union; the standard geometric correctness measure.
- **mAP (mean Average Precision)** — averages precision-recall performance across IoU thresholds and object classes; the standard aggregate metric for classical detectors (COCO mAP@[.5:.95]).
- **Why mAP is a harder bar for a generative detector like Qwen2.5-VL:** mAP assumes well-formed, confidence-scored predictions and is sensitive to exactly the failure mode named in Part 7 — malformed or slightly-off-format JSON output simply can't be scored at all without a repair/parsing step, and generated coordinates lack the confidence calibration that IoU-threshold sweeps assume. If your thesis benchmarks Qwen2.5-VL against a classical detector on mAP, be ready to state this as a known methodological caveat, not an oversight.

---

## Part 15 — Adapting a Pretrained VLM Without Full Retraining

Worth knowing even if your thesis only used the models zero-shot, since "how would you improve this" is a near-universal defense question.

- **Prompt engineering** (Part 6) — zero training, purely inference-time; the ceiling is limited by what the pretrained model already knows how to do.
- **Linear probing** — freeze the encoder entirely, train only a small linear classifier on top of the frozen embeddings for a specific downstream task. Cheap, fast, and a standard way to *measure* how good frozen embeddings already are for a task without confounding the result with fine-tuning effects.
- **Full fine-tuning** — update all model weights on task-specific data. Highest ceiling, highest cost, and the real risk of **catastrophic forgetting** — the model can lose general capabilities while overfitting to the fine-tuning distribution.
- **Parameter-efficient fine-tuning (PEFT), e.g. LoRA** — instead of updating the full weight matrices, LoRA freezes the pretrained weights and learns a small low-rank decomposition ($\Delta W = BA$, with $B$ and $A$ far smaller than $W$) added at inference time. This gets much of the benefit of fine-tuning at a small fraction of the trainable-parameter count and memory cost, and is the standard answer to "how would you adapt Qwen2.5-VL to a specialized domain (e.g., medical imaging, satellite imagery) without retraining it from scratch."
- **In-context / few-shot examples** — for instruction-tuned generative models, providing a few worked examples in the prompt itself (no weight updates at all) can shift output format and behavior, exploiting the same conditional-generation mechanism described in Part 6.

---

## Part 16 — Type

Grouped by question *type*, since committees tend to reuse question patterns even when the specific model changes.

### "Explain the mechanism" questions

**Q: Why does dividing by √d_k in attention matter — what breaks without it?**
A: Dot products between Q and K grow in expected magnitude as dimensionality increases (they're a sum of *d_k* independent-ish terms). Large-magnitude inputs to softmax push it into a saturated regime where the output is nearly one-hot and the gradient with respect to the inputs is near zero — the model can't learn from that step. Scaling by √d_k keeps the pre-softmax variance roughly constant across different key dimensions, keeping training stable regardless of model size.

**Q: Why is CLIP's loss symmetric (image→text *and* text→image)?**
A: The image→text term alone only trains the image encoder to be discriminative relative to the batch's captions; it doesn't directly push the text encoder to be discriminative relative to the batch's images. Averaging both directions ensures both encoders are optimized jointly toward a shared, mutually consistent space rather than one dominating.

**Q: How does a fused VLM answer a question like "what color is the object on the left"?**
A: Because vision and text tokens are fused inside the same Transformer stack (Part 5.2), self-attention lets the text tokens ("left", "color") attend directly to the visual tokens corresponding to spatial regions, weighted by relevance — this joint attention across modalities is exactly what a late-fusion model like CLIP cannot do, since CLIP's two encoders never see each other's internal representations.

### "Justify the design choice" questions

**Q: Why cosine similarity and not Euclidean distance?**
A: Cosine similarity is invariant to vector magnitude, which mostly reflects representation "confidence"/typicality rather than semantic content; Euclidean distance conflates magnitude and direction. Since the signal we want is "do these two vectors point the same way in meaning-space," cosine similarity isolates exactly that.

**Q: Why use in-batch negatives instead of, say, hand-picked hard negatives?**
A: In-batch negatives are free — every other item already in the batch is automatically a negative, requiring no extra data curation, and at large batch sizes this yields plenty of negatives, some of which are naturally hard (semantically close mismatches). The trade-off (Part 12.1) is that negative diversity/difficulty is capped by batch size, which is why huge batches (or external memory banks, as in MoCo) matter for contrastive pretraining quality.

**Q: Why does Qwen2.5-VL express detection as JSON text instead of using a detection head?**
A: It keeps the model a single unified architecture — one Transformer, one training objective (next-token prediction) — that can be extended to any structured output task purely by changing the instruction-tuning data, without adding task-specific architectural components. The cost is no geometric loss and no guarantee of well-formed output (Part 7), which is the honest trade-off to state.

**Q: Why is the CLIP text encoder a Transformer and not, say, an LSTM/RNN?**
A: Self-attention lets every token attend to every other token in a single layer regardless of distance, whereas RNNs process sequentially and must propagate information step by step, which both limits parallelization during training and makes long-range dependencies harder to learn (vanishing gradient over many steps). Transformers also parallelize across the full sequence during training, which is essential at the scale CLIP was trained (400M pairs).

### "What if this assumption breaks" questions

**Q: What happens if a candidate class label is semantically ambiguous or the test image is out-of-distribution relative to CLIP's training data?**
A: Zero-shot performance degrades — CLIP has no explicit uncertainty mechanism; it will still return the highest-cosine-similarity label even when all candidates are a poor match, giving no signal that the prediction is unreliable. This is a known limitation worth naming explicitly if your thesis reports failure cases.

**Q: What happens to Qwen2.5-VL's detection output on an image far outside its instruction-tuning distribution (e.g., a highly unusual domain like microscopy)?**
A: Because detection is learned purely as a text-generation convention rather than grounded in a geometric loss, performance is not guaranteed to degrade gracefully — the model may still produce well-formed JSON with plausible-looking but inaccurate coordinates (a form of hallucination, Part 9), rather than visibly failing, which is more dangerous than an obvious failure in a deployed system.

**Q: If you doubled the number of ViT patches (finer patch resolution) without changing anything else, what would you expect?**
A: More visual tokens → finer spatial granularity (potentially better fine-detail/small-object performance) but quadratically more attention computation (self-attention cost scales with the square of sequence length), and a mismatch with any learned, fixed-size positional embeddings trained at the original patch count unless the model uses a resolution-flexible positional scheme (Part 11.2's RoPE point) or the position embeddings are interpolated.

### "Compare and contrast" questions

**Q: Late fusion (CLIP) vs. early/mid fusion (Qwen2.5-VL) — when would you pick each?**
A: Late fusion is the right choice when you need to pre-compute and index embeddings once and query them cheaply at scale (retrieval, zero-shot classification over large candidate sets) — you never need to know the query in advance. Early/mid fusion is necessary whenever the task requires *reasoning that depends on the specific interaction* between the image and the text (VQA, detection, captioning with fine compositional detail), because that interaction has to happen inside shared attention layers, which by construction cannot be pre-computed independent of the query.

**Q: How is instruction tuning different from the contrastive pretraining CLIP uses?**
A: Contrastive pretraining (Part 4.2) has no notion of "instructions" or conversational roles at all — it's a similarity-matching objective over unstructured (image, caption) pairs. Instruction tuning (Part 5.4) is a supervised (and often RL-refined) stage on top of an already-pretrained generative language model, teaching it to map structured role-tagged input (system/user/assistant) to helpful, on-format responses. They solve different problems: contrastive pretraining builds a *representation space*; instruction tuning builds *controllable generative behavior* on top of a model that already has broad language (and here, vision) competence from pretraining.

---

## Glossary

- **Embedding** — a learned numeric vector representation of an input, positioned so that semantic similarity corresponds to geometric proximity.
- **ViT (Vision Transformer)** — a Transformer applied to an image that has been divided into fixed-size patches, each treated as a token.
- **Contrastive learning** — a training approach that pulls matching pairs together and pushes mismatched pairs apart in embedding space.
- **Cosine similarity** — a measure of the angle between two vectors, independent of their magnitude; the standard similarity metric for embeddings.
- **Zero-shot** — performing a task at inference time without any labeled training examples for that specific task/class.
- **Self-attention** — the mechanism by which each token in a sequence computes a weighted combination of all other tokens to update its own representation.
- **Multi-head attention** — running several attention operations in parallel on lower-dimensional projections, letting different heads specialize in different relational patterns.
- **Residual connection** — adding a sublayer's input back to its output, enabling stable gradient flow through deep stacks of layers.
- **Layer normalization** — rescaling each token's activations to stabilize training across stacked layers.
- **Positional encoding** — information injected into token embeddings so an inherently order-blind attention mechanism can represent sequence/spatial position (fixed sinusoidal, learned, or relative/rotary variants).
- **Autoregressive generation** — producing output one token at a time, each new token conditioned on all previous tokens.
- **Instruction tuning** — a training stage that teaches a pretrained language model to follow structured, role-based instructions (system/user/assistant).
- **Chat template** — the exact text formatting convention (delimiters, role tags) a given instruction-tuned model expects as input.
- **Fusion (early/mid vs. late)** — where in the network vision and language information are combined; CLIP fuses late (only at the final similarity comparison), while Qwen2.5-VL fuses early/mid (inside shared Transformer layers).
- **Prompt engineering** — deliberately phrasing input to steer a model's output distribution toward a desired behavior or format.
- **LoRA (Low-Rank Adaptation)** — a parameter-efficient fine-tuning method that learns a small low-rank weight update instead of modifying the full pretrained weight matrices.
- **mAP / IoU** — standard geometric evaluation metrics for object detection; IoU measures box overlap, mAP aggregates precision-recall across classes/thresholds.
- **CIDEr / CLIPScore** — captioning evaluation metrics; CIDEr is reference-based (TF-IDF-weighted n-gram overlap), CLIPScore is reference-free (image–caption embedding similarity).

---

*This companion is meant to be read alongside, not instead of, the four bootcamp notebooks. Once the theory here feels solid, every line of code in the notebooks should read as an implementation detail of a concept you already understand — not as a magic incantation to copy. For a defense specifically: Parts 10–12 arm you against "explain the math" questions, Part 13 against "why not a different architecture," Part 14 against "how did you evaluate this," Part 15 against "how would you improve/adapt this," and Part 16 is a rehearsal set — read it once, then try answering each question in your own words with the document closed.*
