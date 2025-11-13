# Lapa LLM

## Introducing Lapa LLM v0.1.2 — the most efficient Ukrainian open-source language model



> Demo page: https://huggingface.co/spaces/lapa-llm/lapa  
> Link to Lapa Models: https://huggingface.co/collections/lapa-llm/lapa-v012-release  
> Training code in this folder (we're polishing final configs): [./training/README.md](training/README.md)

## News:
> `13.11.2025` - All pretraining data and models for filtering are published here: https://huggingface.co/collections/lapa-llm/lapa-v012-pretraining  
> `03.11.2025` - All instruction datasets are uploaded here in HuggingFace collection: https://huggingface.co/collections/lapa-llm/lapa-v012-release  
> TBD: Our Reasoning experiments are published here: https://huggingface.co/collections/lapa-llm/lapa-v012-experimental, checkpoints will be published by 09.11.2025



Today, we proudly present Lapa LLM — a cutting-edge open large language model based on Gemma-3-12B with a focus on Ukrainian language processing. The project is the result of many months of work by a team of Ukrainian researchers in artificial intelligence from the Ukrainian Catholic University, AGH University of Krakow, Igor Sikorsky Kyiv Polytechnic Institute, and Lviv Polytechnic, who united to create the best model for Ukrainian language processing.

The model is named in honor of [Valentyn Lapa](https://de.wikipedia.org/wiki/Walentyn_Lapa), who together with [Oleksiy Ivakhnenko](https://uk.wikipedia.org/wiki/%D0%86%D0%B2%D0%B0%D1%85%D0%BD%D0%B5%D0%BD%D0%BA%D0%BE_%D0%9E%D0%BB%D0%B5%D0%BA%D1%81%D1%96%D0%B9_%D0%93%D1%80%D0%B8%D0%B3%D0%BE%D1%80%D0%BE%D0%B2%D0%B8%D1%87) created the Group Method of Data Handling, which is a predecessor to Deep Learning [(source)](https://people.idsia.ch/~juergen/DeepLearning2July2014.pdf).

The project's goal is to create the best model for Ukrainian language processing with open datasets for pretraining and instruction tuning.

### Key Achievements

**Best tokenizer for the Ukrainian language**

Thanks to a SOTA method for tokenizer adaptation developed by [Mykola Haltiuk](https://www.linkedin.com/in/mykola-haltiuk/) as part of this project, it was possible to replace 80,000 tokens out of 250,000 with Ukrainian ones without loss of model quality, thus making Lapa LLM the fastest model for working with the Ukrainian language. Compared to the original Gemma 3, for working with Ukrainian, the model requires 1.5 times fewer tokens, thus performing three times fewer computations to achieve better results.

**Most efficient instruction-tuned model on the market**

Our instruction version of the model in some benchmark categories is only slightly behind the current leader — [MamayLM](https://huggingface.co/spaces/INSAIT-Institute/mamaylm-v1-blog). The team is actively working on new datasets to further improve benchmark scores, which we plan to surpass in the v1.0 model.

### Benchmark Results

- Best English-to-Ukrainian translator with a result of 33 BLEU on FLORES and vice versa, which allows for natural and cost-effective translation of new NLP datasets into Ukrainian
- One of the best models for image processing in Ukrainian in its size class, as measured on the MMZNO benchmark
- One of the best models for Summarization and Q&A, which means excellent performance for RAG
- Tests on propaganda and disinformation questions show the effectiveness of the filtering approach at the pretraining stage and during instruction fine-tuning

Model measurements and comparisons will be published as part of the Ukrainian LLM Leaderboard project; subscribe to the Telegram channel for further news.

**Leader in pretraining results**

Lapa LLM demonstrates the best performance in pretraining benchmarks for Ukrainian language processing, which opens opportunities for use by other researchers to adapt for their own tasks.

The model was trained on data evaluated by various quality assessment models - evaluation of propaganda and disinformation presence, readability, grammar assessment, etc. In the final stages of training, the model was trained on high-quality materials provided for commercial use by the Open Data division of Harvard Library.

**Maximum openness and transparency**

Unlike most available models, Lapa LLM is a maximally open project:
- The model is available for commercial use
- Approximately 25 datasets for model training have been published
- Methods for filtering and processing data are disclosed, including for detecting disinformation and propaganda
- Open source code for the model
- Documentation of the training process is available

This openness allows for the development of the Ukrainian NLP community and helps businesses obtain a tool for the most efficient Ukrainian language processing in terms of both computation and results.

### Application Possibilities

Lapa LLM opens wide possibilities for:
- Processing sensitive documents without transferring data to external servers
- Working with Ukrainian texts taking into account cultural and historical context without code-switching to Russian or other languages
- Building RAG systems and chatbots that write in proper Ukrainian
- Developing specialized solutions through the ability to fine-tune for specific tasks
- Machine translation with the best translation quality from English to Ukrainian and vice versa among all models, including API providers

### Next Steps

- Complete development of the reasoning model
- We are collecting community feedback on the model's performance, so we look forward to receiving it on GitHub or HuggingFace!
- Collecting additional datasets for image processing in Ukrainian
- Collecting additional datasets for instruction following and programming

### Acknowledgment to Sponsors

The creation of Lapa LLM was made possible thanks to the support of our partners and sponsors, primarily the startup **Comand.AI**, which provided computational resources for training the model. We also want to thank the company **ELEKS**, which supported this project through a grant dedicated to the memory of Oleksiy Skrypnyk, and the startup **HuggingFace**, which provided a free corporate subscription to the team for storing models and datasets.

### Links:

Try the model: https://huggingface.co/spaces/lapa-llm/lapa  
Code: https://github.com/lapa-llm/lapa-llm

Subscribe to the Telegram channel for further news about the project: https://t.me/pehade_blog

### Team
