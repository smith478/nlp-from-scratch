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

The hidden state block largely follows the main attention block found in the attention is all you need paper with some modifications on the order of operations and most importantly updating the residual or skip connection stream to allow back propogation to more easily send gradients deeper through the network.

The self attention mechanism is essentially the same as in the attention is all you need paper. For each token in the context window we have 3 outputs (i.e. vectors), the query Q, the key K, and the value V. The element-wise product between the queries and keys gives the amount of attention or importance is given between the two. The output is run through a softmax so that the values sum to one. Finally this output attention amount vector is multiplied by the values vector. Note that GPT-2 uses multi-headed attention wherein each attention head can be run and calculated in parallel. 

## Step 3: Loading model weights from Hugging Face
The naming conventions on each of the layers of the model used in `train_gpt2.py` match those of the hugging face implementation which will allow us to load in those model weights. 

The `from_pretrained` class method within the `GPT` class will load in the fitted weights from hugging face.
