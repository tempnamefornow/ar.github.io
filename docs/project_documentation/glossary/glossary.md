# Glossary

???+ danger "Note"
    This purpose of this page is keep definitions of domain specific terminology.


### LoRA

LoRA Fine-tuning refers to a technique for improving the performance of a pre-trained language model by fine-tuning it on a specific task or domain. LoRA stands for "Language-oriented Recurrent Attention", which is a neural architecture that has been used for pre-training language models.

Fine-tuning a pre-trained language model involves further training the model on a specific task or dataset to adapt it to that task or dataset. This is done by adding a task-specific layer on top of the pre-trained model and training the entire model end-to-end on the task-specific data.

Fine-tuning a language model on a specific task or domain can improve its performance on that task or domain. For example, if you have a pre-trained language model that has been trained on general text data, you can fine-tune it on a specific text classification task, such as sentiment analysis or named entity recognition, to improve its performance on that task.

LoRA Fine-tuning is a specific implementation of this technique that uses the LoRA architecture for pre-training language models and fine-tuning them on specific tasks or domains.

### Self Instruct

Self-Instruct, a framework for improving the instruction-following capabilities of pretrained language models by bootstrapping off its own generations. Our pipeline generates instruction, input, and output samples from a language model, then prunes them before using them to finetune the original model. Applying our method to vanilla GPT3, we demonstrate a 33% absolute improvement over the original model on Super-NaturalInstructions, on par with the performance of InstructGPT_001, which is trained with private user data and human annotations. For further evaluation, we curate a set of expert-written instructions for novel tasks, and show through human evaluation that tuning GPT3 with Self-Instruct outperforms using existing public instruction datasets by a large margin, leaving only a 5% absolute gap behind InstructGPT_001. Self-Instruct provides an almost annotation-free method for aligning pre-trained language models with instructions, and we release our large synthetic dataset to facilitate future studies on instruction tuning.

### Zero-shot

In the context of machine learning, zero-shot refers to a type of learning where a model is trained to perform a task without being explicitly trained on that task. Instead, the model is trained on related tasks and then can perform the new task without additional training data. This is achieved by encoding the task description or instructions as part of the input to the model, allowing it to generalize to new tasks that share similar characteristics. Zero-shot learning is often used in natural language processing tasks, such as language translation, where the model can translate between two languages it has never seen before by relying on its understanding of the structure and semantics of languages it has been trained on.