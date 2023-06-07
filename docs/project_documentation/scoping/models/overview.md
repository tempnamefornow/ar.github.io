# Large Language Models


This article to keep tabs on any open source GPT-like models which can be used, either in full or to be fine tunes



## API Based



## Executable


### tatsu-lab/stanford_alpaca

- [Github Repo](https://github.com/tatsu-lab/stanford_alpaca)
- [Paper](https://arxiv.org/abs/2302.13971)
- [Blog](https://crfm.stanford.edu/2023/03/13/alpaca.html)
- Contains Weights: No

Stanford Alpaca: An Instruction-following LLaMA Model. The current Alpaca model is fine-tuned from a 7B LLaMA model [1] on 52K instruction-following data generated by the techniques in the Self-Instruct [2] paper, with some modifications that we discuss in the next section. In a preliminary human evaluation, we found that the Alpaca 7B model behaves similarly to the text-davinci-003 model on the Self-Instruct instruction-following evaluation suite [2].






### bigscience/bloomz-p3

- [Link](https://huggingface.co/bigscience/bloomz-p3)
- [Github Repo](https://github.com/bigscience-workshop/xmtf)
- [Paper](https://arxiv.org/abs/2211.01786)
- Contains Weights: No

> We present BLOOMZ & mT0, a family of models capable of following human instructions in dozens of languages zero-shot. We finetune BLOOM & mT5 pretrained multilingual language models on our crosslingual task mixture (xP3) and find the resulting models capable of crosslingual generalization to unseen tasks & languages.


### nomic-ai/gpt4all

- [Technical Report](https://s3.amazonaws.com/static.nomic.ai/gpt4all/2023_GPT4All_Technical_Report.pdf)
- [Github Repo](https://github.com/nomic-ai/gpt4all)
- [Training Log](https://github.com/nomic-ai/gpt4all/blob/main/TRAINING_LOG.md)
- Contains weights: Yes

This is a free large language model with ~800k GPT-3.5-Turbo Generations based on LLaMa
> We introduce LLaMA, a collection of foundation language models ranging from 7B to 65B parameters. We train our models on trillions of tokens, and show that it is possible to train state-of-the-art models using publicly available datasets exclusively, without resorting to proprietary and inaccessible datasets. In particular, LLaMA-13B outperforms GPT-3 (175B) on most benchmarks, and LLaMA-65B is competitive with the best models, Chinchilla-70B and PaLM-540B. We release all our models to the research community.