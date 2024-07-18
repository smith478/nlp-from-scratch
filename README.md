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

## Getting started

### Docker

The docker image can be built using `./Dockerfile`. You can build it using the following command, run from the root directory

```bash
docker build . -f Dockerfile --rm -t llm-finetuning:latest
```

### Run docker container

First navigate to this repo on your local machine. Then run the container:

```bash
docker run --gpus all --name nlp-from-scratch -it --rm -p 8888:8888 -p 8501:8501 -p 8000:8000 --entrypoint /bin/bash -w /nlp-from-scratch -v $(pwd):/nlp-from-scratch llm-finetuning:latest
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