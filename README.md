# nlp-from-scratch
Building popular model from scratch to get a better understanding of the underlying tools and techniques.

## GPT
This will follow along with Andrej Karpathy's [Let's reproduce GPT-2](https://www.youtube.com/watch?v=l8pRSuU81PU&ab_channel=AndrejKarpathy) and associated github [repo](https://github.com/karpathy/build-nanogpt?tab=readme-ov-file).

## BERT
Here are some resources we will be using:
- [github resource](https://coaxsoft.com/blog/building-bert-with-pytorch-from-scratch)
- [databricks mosaic](https://mosaicbert.github.io/)
- [databricks github](https://github.com/mosaicml/examples/tree/main/examples/benchmarks/bert)
- [notebook](https://github.com/antonio-f/BERT_from_scratch)
- [youtube](https://www.youtube.com/watch?v=v5cyVwAXR1I&ab_channel=UygarKurt)
- [Medium](https://medium.com/data-and-beyond/complete-guide-to-building-bert-model-from-sratch-3e6562228891)
- [Towards data science](https://towardsdatascience.com/how-to-train-a-bert-model-from-scratch-72cfce554fc6)

## Additional resources
- [Lightning-AI](https://github.com/Lightning-AI)
    - [litgpt](https://github.com/Lightning-AI/litgpt)
    - [pytorch-lightning](https://github.com/Lightning-AI/pytorch-lightning)
- [LLMs from scratch](https://github.com/rasbt/LLMs-from-scratch)

## Getting started

### Conda/pip (CPU)

This option will work if you want to run everything from CPU.

First create a Python 3.11 environment with conda

```bash
conda create -n nlp-from-scratch python==3.11
```

Next use pip to install all of the library dependencies

```bash
pip install -r requirements_cpu.txt
```

### Docker (GPU)

This option works well to handle GPU related dependencies with Docker.

The docker image can be built using `./Dockerfile`. You can build it using the following command, run from the root directory

```bash
docker build . -f Dockerfile --rm -t llm-finetuning:latest
```

### Run docker container

First navigate to this repo on your local machine. Then run the container:

```bash
docker run --gpus all --name nlp-from-scratch -it --rm -p 8888:8888 -p 8501:8501 -p 8000:8000 --entrypoint /bin/bash -w /nlp-from-scratch -v $(pwd):/nlp-from-scratch -v ~/huggingface_models:/root/huggingface_models llm-finetuning:latest
```

### Run jupyter from the container
Inside the Container:
```bash
jupyter lab --ip 0.0.0.0 --no-browser --allow-root --NotebookApp.token=''
```

Host machine access this url:
```bash
localhost:8888/<YOUR TREE HERE>
```

## Data

To test out a number of different methods and techniques we will use the [Reuters dataset](https://kdd.ics.uci.edu/databases/reuters21578/reuters21578.html). This will be in the `data/` directory. The Reuters dataset contains articles along with a list of topics assocated with each article. We will compare a few different modeling techniques to identify topics using the free text of the article. In particular we will compare:
- Fine tune LLM as a multi-class, multi-label classification problem
- Use retrieval augmented generation (RAG)
- Train LLM from scratch on Reuters data
    - Next token prediction like GPT
    - Masked language predication like BERT
- PEFT - Use LORA (or similar variant) to train efficiently