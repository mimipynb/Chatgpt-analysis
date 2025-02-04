# Exploratory analysis on GPT Model

Directory of experiments and drafted works on transformers architecture, specifically GPT. Note that no particular objective in mind when fiddling with these networks but only to note interesting trends or patterns within its latent space. Most of the work was done to assist in building my agents.

**Background:** One of the features that sub-categories the Transformer architecture is how the Attention Block is built. By dividing Transformer architectures by the algebraic operations within the Multi-head Attention Network builds, it narrows down to two distinct types of Transformers:

1. CNN Network: The algebraic operations leverages the Convolutional networks and these are core architectures of GPT models.
2. Linear Network: Single or Multi-layer perceptron are used with either RELU or Tanh activation in the Attention Blocks. These Transformer types are common in BERT/Roberta and Llama models.

References to the experimented models' card on Hugging face:

- [DialGPT](https://huggingface.co/microsoft/DialoGPT-small)
- [GPT2](https://huggingface.co/openai-community/gpt2)
