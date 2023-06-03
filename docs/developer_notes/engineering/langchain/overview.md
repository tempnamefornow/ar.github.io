# Langchain Overview

## Introduction

LangChain is a framework for developing applications powered by language models. Two principles: **Data aware and agentic**. 


- **Components**: LangChain offers modular abstractions and implementations for working with language models. They are designed to be user-friendly, whether you use the full LangChain framework or not.

- **Use-Case Specific Chains:** Chains combine these components to achieve specific use cases. They provide a user-friendly starting point and can be customized.



## Concepts

### Schema

??? tip "ChatMessages"
    **SystemChatMessage:** A chat message representing information that should be instructions to the AI system.

    **HumanChatMessage:** A chat message representing information coming from a human interacting with the AI system.

    **AIChatMessage**: A chat message representing information coming from the AI system.


??? tip "Examples"
    The documentation is explaining the concept of examples within the LangChain framework, which is an application framework for Language Model (LM) development.

    In this context, examples refer to input/output pairs that represent inputs to a function or LM and their corresponding expected outputs. These examples are used for both training and evaluating models.

    The documentation highlights that examples can be utilized for both models and chains within the LangChain framework. Models refer to individual language models, while chains represent a sequence of components or models working together.

    Examples for a model are used specifically for fine-tuning the model. Fine-tuning involves adjusting the model's parameters using the provided inputs and expected outputs, thereby improving its performance.

    On the other hand, examples for a chain are used to evaluate the end-to-end chain. This means assessing the overall performance of the entire sequence of components or models working together. Additionally, examples for a chain may even be used to train a model that can replace the entire chain, simplifying the process and potentially enhancing efficiency.

    Overall, the documentation clarifies the purpose and usage of examples within the LangChain framework, emphasizing their role in training, evaluating, fine-tuning models, and assessing end-to-end chains.


??? tip "Documents"
    The documentation explains the concept of a "document" within the LangChain framework.

    In this context, a document refers to a piece of unstructured data. It can be any form of textual information, such as a webpage, a paragraph, or a document in a natural language format.

    A document consists of two main components: page_content and metadata.

    - **Page_content:** This refers to the actual content of the data within the document. It represents the textual information that the document contains. For example, in the case of a webpage document, the page_content would be the HTML or plain text of the webpage.

    - **Metadata:** Metadata refers to auxiliary pieces of information that describe attributes of the data within the document. It provides additional context or details about the document. This can include information such as the author, creation date, title, tags, or any other relevant attributes associated with the document. Metadata helps provide a broader understanding of the document and can be used for categorization, retrieval, or other analysis purposes.

    In summary, a document in the LangChain framework is an unstructured piece of data that consists of the page_content, representing the actual textual content, and metadata, providing auxiliary information about the attributes of the data.