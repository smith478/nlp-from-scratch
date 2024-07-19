# GPT From Scratch

Here we will give instructions on building GPT-2 from scratch.

## Step 1: Load GPT-2 (124M) Model
Starting with the source code from [OpenAI](https://github.com/openai/gpt-2), we [see](https://github.com/openai/gpt-2/blob/master/src/model.py) that it is implemented in Tensorflow. However we will be using PyTorch instead. Therefore we will instead use the [implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py) from Hugging Face. This can be found in the `explore.ipynb` notebook. This is important as a check to ensure that as we go along everything is working as expected.

Note that in contrast to the original transformer architecture, GPT-2 is a decoder only model and has no encoder or cross-attention components.

## Step 2: Implement the GPT-2 nn.Module
We begin by creating the main components or building blocks of the GPT-2 model. This can be found in `train_gpt2.py`. The main components are:
* `wte` - token embeddin
* `wpe` - position embedding
* `h` - hidden state block that contains the multi-headed attention, feed forward layers, and residual connections
* `ln_f` - layer normalization
* `lm_head` - the output of the model that predicts the next token, i.e. the probability distribution over the vocabulary.
