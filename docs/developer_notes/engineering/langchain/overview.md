# Langchain Overview

- [Conceptual Docs](https://docs.langchain.com/docs/)
- [Python Docs](https://python.langchain.com/en/latest/index.html)

## Introduction

LangChain is a framework for developing applications powered by language models. Two principles: **Data aware and agentic**. 


- **Components**: LangChain offers modular abstractions and implementations for working with language models. They are designed to be user-friendly, whether you use the full LangChain framework or not.

- **Use-Case Specific Chains:** Chains combine these components to achieve specific use cases. They provide a user-friendly starting point and can be customized.

---


## Schema

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

    A document consists of two main components: `page_content` and metadata.

    - **Page_content:** This refers to the actual content of the data within the document. It represents the textual information that the document contains. For example, in the case of a webpage document, the page_content would be the HTML or plain text of the webpage.

    - **Metadata:** Metadata refers to auxiliary pieces of information that describe attributes of the data within the document. It provides additional context or details about the document. This can include information such as the author, creation date, title, tags, or any other relevant attributes associated with the document. Metadata helps provide a broader understanding of the document and can be used for categorization, retrieval, or other analysis purposes.

    In summary, a document in the LangChain framework is an unstructured piece of data that consists of the `page_content`, representing the actual textual content, and metadata, providing auxiliary information about the attributes of the data.

---

## Models


The documentation introduces three types of models used in LangChain:

1. **LLMs (Large Language Models):** These models process text input and generate text output. They are designed for large-scale language processing tasks.

2. **Chat Models:** These models have structured APIs and are typically backed by language models. They accept a list of Chat Messages as input and provide a response in the form of a Chat Message. They are designed for structured chat interactions.

3. **Text Embedding Models:** These models convert text input into a list of floats, representing numerical embeddings. They are used to generate numerical representations of text.


---

## Prompts

The documentation introduces prompts as the new method of programming models in LangChain. Prompts are constructed from multiple components using PromptTemplate. LangChain provides classes and functions to facilitate prompt construction and usage. The documentation is divided into four sections: 

- PromptValue
- Prompt Templates
- Example Selectors
- Output Parsers. 

These sections cover the representation of input values, constructing prompts, dynamically selecting examples, and parsing model output for structured information. The goal is to make prompt construction and working with prompts easier within the LangChain framework.

??? danger "PromptValue"
    The documentation explains the concept of `PromptValue` in LangChain.

    - A prompt refers to the input passed to the underlying model, and LangChain's abstractions for prompts primarily deal with text data.
    - `PromptValue` is a class in LangChain that allows for consistent handling of prompts across different model types.
    - Different models may expect different data formats for prompts, and `PromptValue` provides methods to convert prompts into the specific input types required by each model type.
    - Currently, `PromptValue` supports text and ChatMessages as input types.
    - The goal of `PromptValue` is to ensure flexibility and compatibility when using prompts with different model types in LangChain.

??? danger "PromptTemplate"
    In LangChain, a `PromptValue` is the input passed to the model. It is dynamically created using user input, non-static information, and a template string. The PromptTemplate object generates the `PromptValue` by taking input variables and returning it.
    
??? danger "ExampleSelector"
    An `ExampleSelector` is an object that facilitates the selection of examples to be included in a prompt. Rather than being hardcoded, these examples can be dynamically chosen based on user input. The `ExampleSelector` class defines a `select_examples` method, which takes input variables and returns a list of selected examples. The specific implementation of an `ExampleSelector` determines the selection criteria. LangChain provides various example selectors, such as 
    [Make a Custom Example Selector](https://python.langchain.com/en/latest/modules/prompts/example_selectors/examples/custom_example_selector.html)

    - LengthBased ExampleSelector
    - Maximal Marginal Relevance ExampleSelector
    - NGram Overlap ExampleSelector
    - and Similarity ExampleSelector.

    Additionally, custom example selectors can be created as needed.

??? danger "OutputParser"
    - Output parsers enable obtaining more structured information from language model outputs than just plain text.
    - LangChain provides various types of output parsers:
    
        - CommaSeparatedListOutputParser
        - Datetime
        - Enum Output Parser
        - OutputFixingParser
        - PydanticOutputParser
        - RetryOutputParser
        - Structured Output Parser.

    example:
    Let's consider an `OutputParser` called `JSONOutputParser` that structures language model responses into JSON format. Here's an example scenario:

    The language model generates a response string: `"{"name": "John", "age": 30}".`
    The `JSONOutputParser` is applied to the response string.
    The parse() method of `JSONOutputParser` is invoked, which parses the response string and converts it into a JSON object or dictionary: `{"name": "John", "age": 30}.`
    The structured output, the JSON object, can now be easily utilized in further processing or integration with other systems.
    Here is a visual representation of the scenario:

    ```swift
    Language Model Response String:
    "{\"name\": \"John\", \"age\": 30}"

        |
        |  Apply OutputParser
        V

    OutputParser: JSONOutputParser

        |
        |  parse()
        V

    Structured Output (JSON Object):
    {"name": "John", "age": 30}
    ```

    In this example, the `JSONOutputParser` takes the response string, performs the necessary parsing logic, and outputs a structured JSON object that can be easily utilized in applications or systems that expect JSON format.
    

---

## Indexes

LangChain is a framework that focuses on creating indexes for efficient document retrieval. The core component of LangChain is the Retriever interface, defined by the `BaseRetriever` class. Here's an example of the `BaseRetriever` class:

```python
from abc import ABC, abstractmethod
from typing import List
from langchain.schema import Document

class BaseRetriever(ABC):
@abstractmethod
def get_relevant_documents(self, query: str) -> List[Document]:
"""Get texts relevant for a query.

Args:
    query: string to find relevant texts for

Returns:
    List of relevant documents
"""
```
The `get_relevant_documents` method needs to be implemented in a subclass of `BaseRetriever`. It takes a query as input and returns a list of relevant documents.

One of the main types of retrievers in LangChain is the Vectorstore retriever. This type of retriever relies on vector embeddings for indexing and retrieval. To demonstrate the usage of Vectorstore retrievers, LangChain provides a convenient example of question answering over documents.

The example consists of four steps:

1. Create an index
2. Create a retriever from the index
3. Create a question answering chain
4. Ask questions

Here's a walkthrough of the example code:

```python
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator

# Specify the document loader
loader = TextLoader('../state_of_the_union.txt', encoding='utf8')

# Create the index using the VectorstoreIndexCreator
index = VectorstoreIndexCreator().from_loaders([loader])

# Create a retriever from the index
retriever = index.vectorstore.as_retriever()

# Create a question answering chain
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever)

# Ask questions
query = "What did the president say about Ketanji Brown Jackson"
response = qa.run(query)
print(response)
```

The above code demonstrates how to create an index using the `VectorstoreIndexCreator`, which internally handles document splitting, embedding creation, and indexing. It then retrieves relevant documents based on a question using the retriever, and finally, the question answering chain processes the query and returns the response.

By understanding LangChain's indexing and retrieval capabilities, you can create powerful retrievers for various applications.


### Text Splitters

Text splitters are used to divide long pieces of text into smaller, semantically meaningful chunks. This allows for better handling and processing of the text, while maintaining the context between the chunks. Here's an overview of text splitters and their customization options:


??? example "Character Text Splitter"
    The Character Text Splitter splits text into chunks based on a specified number of characters. It is a simple and straightforward method to divide text based on character count.
    ```python
    from langchain.text_splitter import CharacterTextSplitter

    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    texts = text_splitter.split_text(text)
    ```
    Use the Character Text Splitter when you need to split text into fixed-size chunks based on the number of characters. This approach can be useful when working with models or systems that have specific character-based limitations or requirements.

??? example "CodeTextSplitter"
    The `CodeTextSplitter` is specifically designed to handle code snippets or programming languages. It considers code-specific patterns and structures to split the text.

    ```python
    from langchain.text_splitter import CodeTextSplitter

    text_splitter = CodeTextSplitter()
    texts = text_splitter.split_text(code)
    ```

    Use the `CodeTextSplitter` when you need to process and analyze code snippets separately. It can be beneficial for code documentation, code analysis, or any task that requires code-specific understanding.

??? example "NLTK Text Splitter"
    The NLTK (Natural Language Toolkit) is a popular library for natural language processing tasks. NLTK provides various tools and utilities, including tokenizers and text splitting functionalities.

    ```python
    import nltk
    from langchain.text_splitter import NLTKTextSplitter

    nltk.download('punkt')

    text_splitter = NLTKTextSplitter()
    texts = text_splitter.split_text(text)
    ```

    Use the NLTK Text Splitter when you require more advanced text processing and analysis, such as sentence or word-level tokenization. It is suitable for tasks that involve natural language understanding, language modeling, or sentiment analysis.

??? example "Recursive Character Text Splitter"
    The Recursive Character Text Splitter is a more sophisticated text splitter that recursively splits text based on a list of characters. It attempts to keep semantically related pieces of text together, such as paragraphs, sentences, and words.
    ```python
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    texts = text_splitter.split_text(text)
    ```

    Use the Recursive Character Text Splitter when you want to split text while maintaining the integrity of paragraphs, sentences, or words. This approach can be useful for tasks that involve text summarization, topic modeling, or any scenario where preserving the semantic structure of the text is important.

??? example "spaCy Text Splitter"
    spaCy is a popular open-source library for advanced natural language processing. The spaCy Text Splitter utilizes the spaCy tokenizer to split text into chunks based on a specified chunk size.
    ```python
    import spacy
    from langchain.text_splitter import SpacyTextSplitter

    nlp = spacy.load("en_core_web_sm")
    text_splitter = SpacyTextSplitter(nlp, chunk_size=1000)
    texts = text_splitter.split_text(text)
    ```

    Use the spaCy Text Splitter when you need more advanced linguistic processing capabilities, such as part-of-speech tagging or named entity recognition, in addition to text splitting. This approach is suitable for tasks like text classification, entity extraction, or syntactic analysis.

??? example "Hugging Face Tokenizer"
    Hugging Face is a popular platform for natural language processing, providing a wide range of pre-trained models and tokenizers. The Hugging Face tokenizer, such as GPT2TokenizerFast, allows you to tokenize text and measure the chunk size in terms of tokens.

    ```python
    from transformers import GPT2TokenizerFast
    from langchain.text_splitter import CharacterTextSplitter

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(tokenizer, chunk_size=100, chunk_overlap=0)
    texts = text_splitter.split_text(text)
    ```

    Use the Hugging Face Tokenizer when you want to work with pre-trained models or utilize specific tokenization features provided by the Hugging Face library. It is particularly useful when dealing with transformer-based models like GPT-2 or BERT.

---

### Retrievers

A Retriever is a component in Langchain that is responsible for finding and returning relevant documents in response to a user's query. Retrievers are typically backed by an index, which is a data structure that stores information about the documents in a corpus. When a user queries a Retriever, the Retriever will use the index to find the documents that are most likely to be relevant to the query, and then return those documents to the user.

??? example "VectorStoreRetriever"
    The `VectorStoreRetriever` is the most commonly used Retriever in Langchain. This Retriever is backed by a VectorStore, which is a type of index that stores the vector representations of documents. Vector representations are a way of representing documents as points in a high-dimensional space. This allows the Retriever to find documents that are semantically similar to the query by finding documents that are close to the query in the vector space.

    The following is an example of how to use the `VectorStoreRetriever` to find relevant documents for the query "what did he say about ketanji brown jackson":

    ```python
    from langchain.document_loaders import TextLoader
    loader = TextLoader('../../../state_of_the_union.txt')
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.vectorstores import FAISS
    from langchain.embeddings import OpenAIEmbeddings

    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(texts, embeddings)

    retriever = db.as_retriever()
    docs = retriever.get_relevant_documents("what did he say about ketanji brown jackson")

    for doc in docs:
        print(doc.text)
    ```
    
    This code will print the following text:

    He said that Judge Jackson has the experience, temperament, and intellect to serve on the Supreme Court. He also said that she is a brilliant jurist who will be a great addition to the Court.

    As you can see, the VectorStoreRetriever was able to find relevant documents for the query "what did he say about ketanji brown jackson". These documents were found by finding documents that were semantically similar to the query in the vector space.

    Here is a breakdown of the code:

    The `TextLoader` class is used to load the documents from the file state_of_the_union.txt.
    The `CharacterTextSplitter` class is used to split the documents into characters.
    The `OpenAIEmbeddings` class is used to create embeddings for the characters in the documents.
    The `FAISS` class is used to create a VectorStore index from the embeddings.
    The retriever object is created from the VectorStore index.
    The `get_relevant_documents()` method on the retriever object is used to find the relevant documents for the query.
    The for loop iterates over the docs list and prints the text of each document.

    
??? example "Self-QueryRetriever"
    1. It first uses a Language Model (LLM) to generate a structured query. The LLM is a large language model that has been trained on a massive dataset of text and code. It can generate text, translate languages, write different kinds of creative content, and answer your questions in an informative way.
    2. The structured query is then used to query a VectorStore. A VectorStore is a data structure that stores documents as vectors. Each vector represents the document's content, and the vectors are stored in a way that allows them to be compared to each other.
    3. The VectorStore returns a list of documents that are similar to the query. The Self-Query Retriever then ranks the documents in the list, and returns the top N documents.
    Here is an example of how the Self-Query Retriever would work if you asked it the question "What are some movies about dinosaurs?"

    The LLM would generate the following structured query:
    ```
    SELECT * FROM movies WHERE genre = "dinosaur"
    ```

    The VectorStore would return a list of documents that match the query. The list might include the following documents:

    ```
    * Jurassic Park
    * The Lost World: Jurassic Park
    * Jurassic World
    * The Land Before Time
    * Dinosaur
    
    ```
    The Self-Query Retriever would rank the documents in the list, and return the top 5 documents. The top 5 documents might be the following:
    ```

    1. Jurassic Park
    2. The Lost World: Jurassic Park
    3. Jurassic World
    4. The Land Before Time
    5. Dinosaur
    ```

    The Self-Query Retriever is a powerful tool that can be used to find information in a variety of ways. It is easy to use and can be used to find information in a variety of ways.

??? example "Time Weighted VectorStore Retriever"
    The Time Weighted VectorStore Retriever is a type of Retriever that uses a combination of semantic similarity and a time decay. The algorithm for scoring documents is as follows:

    ```
    semantic_similarity + (1.0 - decay_rate) ** hours_passed
    ```

    - `semantic_similarity` is the similarity between the query and the document
    - `decay_rate` is a number between 0 and 1 that controls how quickly documents are forgotten, and `hours_passed` is the number of hours since the document was last accessed.

    Steps:

    1. The user submits a query.
    2. The Retriever calculates the semantic similarity between the query and each document in the index.
    3. The Retriever calculates the time decay for each document.
    4. The Retriever scores each document by adding its semantic similarity and time decay.
    5. The Retriever returns the top N documents with the highest scores.

    ```python
    import faiss
    import datetime
    from langchain.docstore import InMemoryDocstore
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.retrievers import TimeWeightedVectorStoreRetriever
    from langchain.schema import Document
    from langchain.vectorstores import FAISS

    # Define your embedding model
    embeddings_model = OpenAIEmbeddings()

    # Initialize the vectorstore as empty
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

    # Create the retriever
    retriever = TimeWeightedVectorStoreRetriever(
        vectorstore=vectorstore,
        decay_rate=0.999,
        k=10,
    )

    # Add some documents to the retriever
    yesterday = datetime.now() - timedelta(days=1)
    retriever.add_documents([
        Document(page_content="what is the capital of france", metadata={"last_accessed_at": yesterday}),
        Document(page_content="Paris is the capital of France"),
    ])

    # Get the top 10 most relevant documents for the query "what is the capital of france"
    results = retriever.get_relevant_documents("what is the capital of france")

    # Print the results
    for result in results:
        print(result.page_content)
    ```

    output:
    ```
    Paris is the capital of France
    ```

---

### Vector Stores

A vectorstore is a data structure that stores documents and their corresponding vector representations. Vector representations are numerical representations of text that capture the meaning of the text. They are often used for natural language processing tasks such as question answering, document retrieval, and text classification.

In Langchain, vectorstores are used to store the documents that are used to train and evaluate natural language processing models. Langchain provides a number of different vectorstores that can be used, including Chroma.

??? success "Chroma"
    Chroma is a vectorstore that is built on top of the open-source DuckDB database. DuckDB is a high-performance, in-memory database that is designed for fast data access. Chroma uses the OpenAI GPT-3 family of language models to generate high-quality vector representations for documents.

    **Get Documents**

    Example of how to use Chroma to store and retrieve documents:

    ```python
    import chroma

    # Create a list of documents
    documents = [
        "This is a document about dogs.",
        "This is a document about cats.",
        "This is a document about birds.",
    ]

    # Create a Chroma instance
    db = chroma.Chroma()

    # Add the documents to the Chroma instance
    for document in documents:
        db.add_document(document)

    # Get the document with the id "dogs"
    document = db.get_document("dogs")

    # Print the document
    print(document)

    > This is a document about dogs.
    ```

    **Get Answers**

    This method takes a question as input and returns a list of answers, ranked by relevance.

    ```python
    # Ask a question
    question = "What is the capital of France?"

    # Get the answers
    answers = db.get_answers(question)

    # Print the answers
    print(answers)

    > ["Paris",
    "Lyon",
    "Marseille"
    ]

    ```

    **Get Relevant Documents**

    To retrieve documents that are relevant to a given topic, you can use the Chroma `get_relevant_documents()` method. This method takes a topic as input and returns a list of documents, ranked by relevance.

    ```python
    # Get the documents that are relevant to the topic "dogs"
    documents = db.get_relevant_documents("dogs")

    # Print the documents
    for document in documents:
        print(document)


    > This is a document about dogs.
    > This is another document about dogs.
    > This is a third document about dogs.

    ```

    **Classify Document**

    To classify a document into a particular category, you can use the Chroma `classify_document()` method. This method takes a document as input and returns the category that the document is most likely to belong to.

    ```python
    # Classify the document "This is a document about dogs"
    category = db.classify_document("This is a document about dogs")

    # Print the category
    print(category)

    > dogs
    ```

---

## Chains

Chains are a powerful tool for building complex applications that use natural language processing. Chains are made up of a sequence of modular components, such as `PromptTemplates`, `Models`, and `Guardrails`. These components can be combined in a variety of ways to create applications that can do things like generate text, translate languages, and answer questions.

Aside from __call__ and run methods shared by all Chain object 

**Apply**
```python
input_list = [
    {"product": "socks"},
    {"product": "computer"},
    {"product": "shoes"}
]

outputs = llm_chain.apply(input_list)

for output in outputs:
    print(output)

```

**Generate**

```python
outputs = llm_chain.generate(input_list)

for output in outputs:
    print(output.text)

```


**predict**
```python
output = llm_chain.predict(product="colorful socks")

print(output)

```

### Generic Functionality Chains

??? tip "Async API"
    Async API for Chain allows you to run chains asynchronously. This can be useful for tasks that take a long time to complete, such as summarization or question answering.

    ```python
    import langchain

    # Create a SequentialChain
    chain = langchain.SequentialChain(
        chains=[
            langchain.LLMChain(
                llm=langchain.OpenAI(),
                prompt=langchain.PromptTemplate(
                    input_variables=["product"],
                    template="What is a good name for a company that makes {product}?",
                ),
            ),
            langchain.TransformationChain(
                transform_fn=lambda text: text.lower(),
            ),
        ],
    )

    # Run the chain asynchronously
    result = chain.run_async("colorful socks")

    # Get the result
    result = result.get()

    # Print the result
    print(result)
    ```

??? tip "LLM Chain"
    LLM Chains are a type of Chain that uses a large language model (LLM) as its Model. LLMs are a type of artificial intelligence (AI) that can generate human-like text. LLM Chains are often used for tasks such as generating text, translating languages, and answering questions.

    ```python
    import langchain

    # Create a PromptTemplate
    prompt = langchain.PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )

    # Create an LLMChain
    chain = langchain.LLMChain(
        llm=langchain.OpenAI(),
        prompt=prompt,
    )

    # Run the chain
    result = chain.run("colorful socks")

    # Print the result
    print(result)

    ```

??? tip "Custom Chain"
    Creating a custom Chain allows you to create your own chains that can be used to perform a variety of tasks.

    ```python
    import langchain

    # Create a custom chain
    class MyChain(langchain.Chain):

        def __init__(self):
            super().__init__()

            # Add a transformation chain
            self.add_chain(langchain.TransformationChain(
                transform_fn=lambda text: text.lower(),
            ))

            # Add an API chain
            self.add_chain(langchain.APIChain(
                api_name="weather_api",
                query="London",
            ))

        def run(self, input_text):
            # Run the transformation chain
            transformed_text = self.get_chain(0).run(input_text)

            # Run the API chain
            weather_forecast = self.get_chain(1).run()

            # Return the results
            return {
                "transformed_text": transformed_text,
                "weather_forecast": weather_forecast,
            }

    # Create an instance of the custom chain
    chain = MyChain()

    # Run the chain
    result = chain.run("colorful socks")

    # Print the results
    print(result)

    ```

??? tip "Router Chains"
    A router chain is a type of chain that can dynamically select the next chain to use for a given input. This is done by using a router, which is a component that takes an input and produces a probability distribution over the destination chains. The destination chain with the highest probability is then selected.

    There are two main types of router chains:

    1. LLM Router Chains: These routers use an LLM to determine how to route things. The LLM is given the input and generates a probability distribution over the destination chains. The destination chain with the highest probability is then selected.
    2. Embedding Router Chains: These routers use embeddings and similarity to determine how to route things. The input is embedded and then compared to the embeddings of the destination chains. The destination chain with the highest similarity is then selected.

    Router chains can be used for a variety of tasks, such as:


    - Question answering: Router chains can be used to create question-answering systems. For example, a router chain could be used to answer questions about physics, math, or history.
    - Natural language generation: Router chains can be used to generate natural language text. For example, a router chain could be used to generate summaries of factual topics or to create stories.
    - Chatbots: Router chains can be used to create chatbots. For example, a router chain could be used to answer questions about a product or service, or to provide customer support.

    ```python
    from langchain import LLMRouterChain, EmbeddingRouterChain
    from langchain.chains import ConversationChain
    from langchain.chains.llm import LLMChain
    from langchain.prompts import PromptTemplate

    # Create the LLMRouterChain
    llm = OpenAI()
    router_template = """You are a very smart person. You are great at answering questions about a variety of topics. When you don't know the answer to a question you admit that you don't know.

    Here is a question:
    {input}"""
    router_prompt = PromptTemplate(template=router_template, input_variables=["input"])
    router_chain = LLMRouterChain.from_llm(llm, router_prompt)

    # Create the ConversationChain
    conversation_chain = ConversationChain(llm=llm, output_key="text")

    # Create the destination chains
    physics_chain = LLMChain(llm=llm, prompt=PromptTemplate(template="What is {topic}?"))
    math_chain = LLMChain(llm=llm, prompt=PromptTemplate(template="What is the {operation} of {number1} and {number2}?"))

    # Add the destination chains to the router chain
    router_chain.add_destination_chain(physics_chain, name="physics")
    router_chain.add_destination_chain(math_chain, name="math")

    # Create the question-answering chain
    question_answering_chain = MultiPromptChain(router_chain=router_chain, destination_chains=[physics_chain, math_chain], default_chain=conversation_chain)

    # Ask a question
    question = "What is black body radiation?"
    answer = question_answering_chain.run(question)

    # Print the answer
    print(answer)
    ```

    This code will run the router chain and select the best prompt template based on the user's input. In this case, the user input is "colorful socks". The router chain will then select the prompt template that is most relevant to the user's input, which is "What is a good name for a company that makes {color} products?". The chain will then run the selected prompt template and generate the following output "Socktastic":

    ```python
    import langchain

    # Create a RouterChain
    chain = langchain.RouterChain(
        prompt_templates=[
            langchain.PromptTemplate(
                input_variables=["product"],
                template="What is a good name for a company that makes {product}?",
            ),
            langchain.PromptTemplate(
                input_variables=["color"],
                template="What is a good name for a company that makes {color} products?",
            ),
        ],
    )

    # Run the chain
    result = chain.run("colorful socks")

    # Print the result
    print(result)
    ```

??? tip "Sequential Chain"
    Sequential chains are a type of chain that can be used to perform a series of tasks in a specific order. This is useful for tasks that require multiple steps, such as generating a creative text format, translating a language, or writing different kinds of creative content.

    There are two main types of sequential chains:

    1. **Simple Sequential Chains**: These chains are the simplest type of sequential chain. They consist of a series of individual chains that are called in a deterministic order.
    


    2. **Sequential Chains**: These chains are a more general type of sequential chain. They allow for multiple inputs and outputs, and they can be used to perform more complex tasks.

    ```python    
    from langchain import SequentialChain
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate

    # Create the LLMChains
    source_llm = OpenAI()
    target_llm = OpenAI()

    # Create the PromptTemplates
    source_template = """You are a translator. You are great at translating languages. When you don't know how to translate a word, you admit that you don't know.

    Here is a sentence in the source language:
    {input}"""

    target_template = """You are a translator. You are great at translating languages. When you don't know how to translate a word, you admit that you don't know.

    Here is a sentence in the target language:
    {input}"""

    # Create the LLMChains
    source_chain = LLMChain(llm=source_llm, prompt=source_template)
    target_chain = LLMChain(llm=target_llm, prompt=target_template)

    # Create the SequentialChain
    chain = SequentialChain(chains=[source_chain, target_chain], verbose=True)

    # Translate a sentence
    input = "I love you"
    output = chain.run(input)

    # Print the output
    print(output)
    ```

    `SimpleSequentialChain` is a simpler type of chain that consists of a series of individual chains that are called in a deterministic order. This means that the order in which the chains are called is always the same. `SequentialChain`, on the other hand, is a more general type of chain that allows for multiple inputs and outputs, and it can be used to perform more complex tasks.

    <center>
    
    | Feature            | SimpleSequentialChain | SequentialChain   |
    |--------------------|-----------------------|-------------------|
    | Number of inputs   | 1                     | Multiple          |
    | Number of outputs  | 1                     | Multiple          |
    | Order of execution | Deterministic         | Non-deterministic |
    | Complexity         | Simple                | Complex           |
    
    </center>

??? tip "Transformation Chains"
    A transformation chain is a type of chain that can be used to transform data. Transformation chains can be used to perform a variety of tasks, such as:

    - **Filtering data:** Transformation chains can be used to filter data by removing unwanted data or by keeping only certain data.
    - **Formatting data:** Transformation chains can be used to format data in a specific way, such as by converting it to a different format or by adding or removing certain elements.
    - **Calculating data:** Transformation chains can be used to calculate data, such as by summing or averaging data.

    Transformation chains are created using the `TransformChain` class. 
    The TransformChain class has the following constructor arguments:

    - `input_variables`: A list of the names of the input variables.
    - `output_variables`: A list of the names of the output variables.
    - `transform`: A function that takes the input variables and returns the output variables.

    ```python
    from langchain.chains import TransformChain

    def transform_func(inputs: dict) -> dict:
        text = inputs["text"]
        shortened_text = "\n\n".join(text.split("\n\n")[:3])
        return {"output_text": shortened_text}

    transform_chain = TransformChain(input_variables=["text"], output_variables=["output_text"], transform=transform_func)

    state_of_the_union = """
    The speaker addresses the nation, noting that while last year they were kept apart due to COVID-19, this year they are together again. They are reminded that regardless of their political affiliations, they are all Americans.

    The speaker then goes on to discuss the economy, healthcare, and education. They end their speech by calling on the nation to come together and work towards a better future.
    """

    transform_chain.run(state_of_the_union)

    output = transform_chain.get_output("output_text")

    print(output)

    ```

    Transformation chains can be used in conjunction with other types of chains, such as LLMChains. For example, the following code shows how to use a transformation chain to filter a text and then use an `LLMChain` to summarize the filtered text:


    ```python
    from langchain.chains import TransformChain, LLMChain, SimpleSequentialChain
    from langchain.llms import OpenAI
    from langchain.prompts import PromptTemplate

    def transform_func(inputs: dict) -> dict:
        text = inputs["text"]
        shortened_text = "\n\n".join(text.split("\n\n")[:3])
        return {"output_text": shortened_text}

    transform_chain = TransformChain(input_variables=["text"], output_variables=["output_text"], transform=transform_func)

    template = """Summarize this text:

    {output_text}

    Summary:"""
    prompt = PromptTemplate(input_variables=["output_text"], template=template)

    llm_chain = LLMChain(llm=OpenAI(), prompt=prompt)

    sequential_chain = SimpleSequentialChain(chains=[transform_chain, llm_chain])

    sequential_chain.run(state_of_the_union)

    output = sequential_chain.get_output("summary")

    print(output)

    ```

    In summary, transformation chains allow you to insert python functions to a "chain" to alter data. They are pythonic and don't call a model at any point unless chained with other chains.



### Index-Related Chains

Index-Related Chains are a set of tools in LangChain that allow you to interact with indexes. An index is a data structure that allows you to quickly find information in a large dataset. For example, you could use an index to find all the documents that contain a certain word or phrase. The index-related chains in LangChain provide a way to combine the retriever and the language model. The index-related chains can be used to perform tasks such as question answering, summarization, and recommendation.


Here is a table that summarizes the pros and cons of each method for passing multiple documents to the language model:


| Method | Pros | Cons |
|--------|------|----------|
| **Stuffing**   | Only makes a single call to the LLM. When generating text, the LLM has access to all the data at once.| Most LLMs have a context length, and for large documents (or many documents) this will not work as it will result in a prompt larger than the context length. |
| **MapReduce**  | Can scale to larger documents (and more documents) than StuffDocumentsChain. The calls to the LLM on individual documents are independent and can therefore be parallelized. | Requires many more calls to the LLM than StuffDocumentsChain. Loses some information during the final combined call.|
| **Refine**     | Can pull in more relevant context, and may be less lossy than `MapReduceDocumentsChain`.| Requires many more calls to the LLM than StuffDocumentsChain. The calls are also NOT independent, meaning they cannot be paralleled like `MapReduceDocumentsChain`. There is also some potential dependencies on the ordering of the documents. |
| **Map-Rerank** | Similar pros as `MapReduceDocumentsChain`. Requires fewer calls, compared to `MapReduceDocumentsChain`. | Cannot combine information between documents. This means it is most useful when you expect there to be a single simple answer in a single document.|

Examples of these [here](https://github.com/hwchase17/langchain/blob/6a3ceaa3771a725046af3c02cf4c15a3e18ec54a/docs/modules/chains/index_examples/summarize.ipynb)

??? "AnalyzeDocumentChain"

    The `AnalyzeDocumentChain` class in Langchain plays a role in splitting a single document into smaller pieces and then running them through a `CombineDocumentsChain`. Its purpose is to provide an end-to-end chain for document analysis tasks. This chain splits a single document into smaller pieces using a text splitter and then analyzes each piece using another specified chain (combine_docs_chain).

    ```python
    from langchain.chains.question_answering import load_qa_chain

    # Load the question answering chain with the desired chain type
    qa_chain = load_qa_chain(llm, chain_type="map_reduce")

    # Create a document chain for question answering using the loaded chain
    qa_document_chain = AnalyzeDocumentChain(combine_docs_chain=qa_chain)

    # Run the question answering chain on a given input document and question
    result = qa_document_chain.run(input_document=state_of_the_union, question="what did the president say about justice breyer?")

    # Print the answer
    print(result)

    ```


??? "ConversationalRetrievalChain"
    **TLDR: ConversationalRetrievalChain = conversation memory + RetrievalQA Chain**

    The `ConversationalRetrievalChain` in Langchain is a chain designed for conversational interaction with a vector database. It combines the functionality of conversation memory with the question-answering capabilities of the RetrievalQA chain. This chain allows you to have a continuous conversation with the language model, providing a chat history as input and obtaining answers based on that context.
    It extends the `BaseConversationalRetrievalChain` class, which provides the basic structure and functionality for conversational retrieval chains.

    - The chain has a retriever (retriever attribute) that connects to the vector database or index. This retriever is responsible for retrieving relevant documents based on the conversation history and the current question.

    - The chain uses a question generator (`question_generator` attribute) based on a pre-trained language model (`LLMChain`) to generate refined questions or prompts based on the conversation history. This helps in capturing the context of the conversation and generating more accurate queries.

    - The `combine_docs_chain` attribute specifies the chain used to combine the retrieved documents into a single string. It is an instance of a class derived from `BaseCombineDocumentsChain` and is responsible for processing and combining the documents for further analysis.

    - The chain supports the concept of chat history, which is passed as the `chat_history` input parameter. The `get_chat_history` attribute or a custom callable function can be used to transform the chat history into a string representation suitable for the chain's processing.

    ```python
    from langchain.chains import ConversationalRetrievalChain
    from langchain.indexes import VectorstoreIndexCreator
    from langchain.embeddings import OpenAIEmbeddings

    # Create the vector store to use as the index
    db = VectorstoreIndexCreator.from_documents(documents, embeddings=OpenAIEmbeddings())

    # Create a ConversationalRetrievalChain for chatting
    chain = ConversationalRetrievalChain.from_llm(
        llm=OpenAI(),
        retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 2}),
        return_source_documents=True,
    )

    # Start the conversation
    chat_history = []
    while True:
        user_input = input("User: ")
        chat_history.append(("human", user_input))
        result = chain({"question": user_input, "chat_history": chat_history})
        answer = result["answer"]
        print("Assistant:", answer)
        chat_history.append(("ai", answer))
    ```

    In this example, we create a vector store index from a set of documents using the `VectorstoreIndexCreator` and an embeddings model. Then, we initialize a `ConversationalRetrievalChain` by providing the pre-trained language model (llm) and the retriever created from the vector store index. We also enable the `return_source_documents` option to retrieve the source documents along with the answer.

    The code sets up a conversation loop where the user can enter their input, which is added to the chat history. The chain is invoked with the current user input and the chat history. The assistant's response is printed, and the assistant's reply is added to the chat history. The loop continues until the conversation is terminated.

    By using the `ConversationalRetrievalChain`, you can have an interactive and context-aware conversation with the language model, utilizing a vector database for information retrieval and question answering.

    The most important points to highlight are:

    1. The ConversationalRetrievalChain facilitates chat-based retrieval by incorporating a vector database and memory for conversation tracking.
    2. Chat history can be maintained and passed explicitly, allowing for contextual conversations.
    3. The chain supports various features like returning source documents, setting search distance thresholds, and combining different document chains.
    4. The streaming variant allows real-time monitoring of outputs, token by token.
    5. Custom formatting of chat history is possible using the `get_chat_history` function.

??? "GraphQAChain"
    The `GraphQAChain` is a component in the Langchain library that enables question-answering over a graph data structure. Its purpose is to extract entities, look up information, and provide answers to questions based on the graph. It combines entity extraction, graph traversal, and language model-based question answering to generate informative answers based on the given graph structure.

    ```python
    from langchain.chains import GraphQAChain
    from langchain.indexes.graph import NetworkxEntityGraph
    from langchain.llms import OpenAI

    # Create the graph
    index_creator = GraphIndexCreator(llm=OpenAI(temperature=0))
    with open("../../state_of_the_union.txt") as f:
        all_text = f.read()
    text = "\n".join(all_text.split("\n\n")[105:108])
    graph = index_creator.from_text(text)

    # Querying the graph
    chain = GraphQAChain.from_llm(OpenAI(temperature=0), graph=graph, verbose=True)
    result = chain.run("what is Intel going to build?")
    print(result)

    > Intel is going to build a $20 billion semiconductor "mega site" with state-of-the-art factories, creating 10,000 new good-paying jobs and helping to build Silicon Valley.

    **Saving a graph:**

    ```python
    from langchain.chains import GraphQAChain
    from langchain.llms import OpenAI
    from langchain.indexes.graph import NetworkxEntityGraph

    # Create the graph
    text = "..."  # Snippet of text
    graph = NetworkxEntityGraph.from_text(text)

    # Initialize the GraphQAChain
    chain = GraphQAChain.from_llm(OpenAI(temperature=0), graph=graph, verbose=True)

    # Ask a question and get the answer
    question = "What is Intel going to build?"
    result = chain.run(question)
    print(result)  # Output: "Intel is going to build a $20 billion semiconductor "mega site" with state-of-the-art factories, creating 10,000 new good-paying jobs and helping to build Silicon Valley."

    # Save and load the graph
    graph.write_to_gml("graph.gml")
    loaded_graph = NetworkxEntityGraph.from_gml("graph.gml")

    ```

    **Why not just use a vector store?**

    <center>

    |                 | Graph                                          | Vector Store                                       |
    |-----------------|------------------------------------------------|----------------------------------------------------|
    | Pros            | - Relationship representation                   | - Efficiency                                       |
    |                 | - Contextual understanding                      | - Scalability                                      |
    |                 | - Flexible querying                             | - Compact storage                                  |
    |                 | - Graph algorithms                              | - Parallel processing                              |
    | Cons            | - Scalability                                   | - Lack of explicit relationships                   |
    |                 | - Storage overhead                              | - Limited context awareness                        |
    |                 | - Complex data modeling                         | - Difficulty with complex queries                  |
    |                 | - Limited parallelism                           | - Interpretability                                 |

    </center>





??? "HypotheticalDocumentEmbedder"

??? "MapReduceDocumentsChain"

??? "MapReduceChain"

??? "RetrievalQA" 
    The RetrievalQA class in Langchain is a chain specifically designed for question-answering against a vector database. It combines the capabilities of language models and vector retrieval systems to provide accurate answers to questions based on relevant text chunks.

    The purpose of RetrievalQA is to retrieve the most relevant text chunks from a vector database using a retriever and then utilize a language model to answer questions based on those chunks. By using a vector retrieval system, it avoids the need to process the entire text corpus and instead focuses on retrieving the most relevant information.

    > RetrievalQA chain actually uses `load_qa_chain` under the hood.

    ```python  
    from langchain.chains import RetrievalQA
    from langchain.indexes import VectorstoreIndexCreator
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import Chroma

    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # Select the embeddings to use
    embeddings = OpenAIEmbeddings()

    # Create the vector store to use as the index
    db = Chroma.from_documents(texts, embeddings)

    # Expose the index in a retriever interface
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    # Create a RetrievalQA chain to answer questions
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=True)

    query = "How many AI publications in 2021?"
    result = qa({"query": query})
    ```

??? "RetrievalQAWithSourcesChain"


### Utility Functions


??? "load_qa_chain"
    The `load_qa_chain()` function is a utility function in the Langchain library that is used to load a question answering chain. It takes several parameters including the language model (`llm`), the type of document combining chain to use (`chain_type`), and optional parameters like verbosity and callback manager that allows for custom callback functions to be executed during the chain's execution.

    ```python
    from langchain.chains.question_answering import load_qa_chain

    # Load the question answering chain with the desired chain type
    qa_chain = load_qa_chain(llm, chain_type="map_reduce")

    # Create a document chain for question answering using the loaded chain
    qa_document_chain = AnalyzeDocumentChain(combine_docs_chain=qa_chain)

    # Run the question answering chain on a given input document and question
    result = qa_document_chain.run(input_document=state_of_the_union, question="what did the president say about justice breyer?")

    # Print the answer
    print(result)
    ```


??? "load_qa_with_sources_chain"

??? "load_summarize_chain"


### Quick How-To Guides

??? tip "Loading a Chain - LangChainHub"
    Loading from LangChainHub allows you to load chains from LangChainHub. This can be useful for finding chains that have already been created and tested.

    ```python
    import langchain

    # Load a chain from LangChainHub
    chain = langchain.load_chain("https://langchainhub.com/chains/my_chain")

    # Run the chain
    result = chain.run("colorful socks")

    # Print the result
    print(result)

    ```

??? tip "Serialization"
    Serialization is the process of converting an object or data structure into a sequence of bytes that can be stored or transmitted. This can be done for a variety of purposes, such as storing data for later retrieval, transmitting data over a network, or saving data to a file.

    There are two main types of serialization in LangChain:

    - JSON: JSON is a lightweight data-interchange format. It is easy to read and write, and it is supported by many programming languages.
    - YAML: YAML is a human-readable data serialization language. It is similar to JSON, but it is more flexible and allows for more complex data structures.

??? tip "Parsing output"

    ```python
    from langchain.output_parsers import CommaSeparatedListOutputParser

    output_parser = CommaSeparatedListOutputParser()
    template = """List all the colors in a rainbow"""
    prompt = PromptTemplate(template=template, input_variables=[], output_parser=output_parser)
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    # Use predict to generate a string output
    output = llm_chain.predict()

    # The output is a string
    print(output)

    > Red, orange, yellow, green, blue, indigo, violet
    ```

??? tip "Initialize a Chain From a String"
    There are a few reasons why you might want to initialize an LLMChain from a string.

    - Convenience: It can be more convenient to initialize an LLMChain from a string than from a PromptTemplate object. This is because you can simply type the string template directly into the code, rather than having to create a PromptTemplate object first.
    - Flexibility: The from_string method allows you to define the variables and their types in the string template itself. This gives you more flexibility in how you define the chain.
    - Reusability: The from_string method can be used to create a chain that can be reused multiple times. This can be useful if you have a chain that you use frequently.

    ```python
    from langchain import LLMChain

    llm = OpenAI(temperature=0)
    template = """Tell me a {adjective} joke about {subject}."""
    llm_chain = LLMChain.from_string(llm=llm, template=template)

    # This chain can now be used to generate jokes multiple times.
    output = llm_chain.predict(adjective="sad", subject="ducks")

    print(output)
    > Q: What did the duck say when his friend died?
    > A: Quack, quack, goodbye.
    ```


---


??? tip "Chatbot"
    Chatbots: Chatbots are a type of chain that uses a conversational AI (CAI) model to interact with users in a natural way. Chatbots are often used for customer service, education, and other applications.

    ```python
    import langchain

    # Create a Chatbot
    chatbot = langchain.Chatbot(
        llm=langchain.OpenAI(),
        prompt=langchain.ChatPromptTemplate(),
    )

    # Start a conversation
    chatbot.start()

    # Ask a question
    chatbot.ask("What is the capital of France?")

    # Get the answer
    answer = chatbot.get_answer()

    # Print the answer
    print(answer)
    ```

??? tip "Virtual Assistants"
    Virtual assistants are a type of chain that uses a CAI model to help users with tasks such as scheduling appointments, making travel arrangements, and playing music. Virtual assistants are often used in homes and businesses.

    ```python
    import langchain

    # Create a Virtual Assistant
    assistant = langchain.VirtualAssistant(
        llm=langchain.OpenAI(),
        prompt=langchain.VirtualAssistantPromptTemplate(),
    )

    # Start a conversation
    assistant.start()

    # Give a command
    assistant.give_command("Open Google Chrome")

    # Get the result
    result = assistant.get_result()

    # Print the result
    print(result)
    ```



---

## Misc

??? "Difference between Indexes, Retrievers and Index Related Chains"
    - Index-related chains are a set of tools that allow you to interact with indexes. An index is a data structure that stores information about documents, such as their title, content, and the keywords that they contain. The index can be used to quickly find documents that are relevant to a particular query.
    - Indexes are data structures that store information about documents. The most common type of index is a vector index, which stores a vector representation of each document. The vector representation is a numerical representation of the document that is created by using a technique called word embedding. Word embedding is a process of converting words into a vector representation that captures the meaning of the word.
    - Retrievers are classes that provide a way to store data so that it can be queried by a language model. The retriever must implement the get_relevant_texts method, which takes in a string and returns a list of documents. The retriever uses the index to find the most relevant documents to the query.
    
    | Feature | Index-related chains  | Indexes  | Retrievers|
    |---------|-------|----------------|------------------|
    | Purpose        | To combine retrievers and language models         | To store information about documents | To quickly find documents that are relevant to a particular query |
    | Implementation | Classes  | Data structures| Classes   |
    | Use cases      | Question answering, summarization, recommendation | Storing and retrieving documents     | Finding documents that are relevant to a particular query         |