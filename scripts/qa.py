from collections.abc import Iterable

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.config import Config
from src.vectorstore import VectorStoreFactory


def main() -> None:
    load_dotenv()
    config = Config()
    question = input("Question?: ")

    vectorstore = VectorStoreFactory(config.chroma_config).create(
        "sample",
        embeddings=OpenAIEmbeddings(api_key=config.openai_config.api_key),
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    prompt = PromptTemplate(
        template="""
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Question: {question}
Context: {context}
Answer:
        """,
        input_variables=["context", "question"],
    )
    llm = ChatOpenAI(model="gpt-4o-mini")

    def format_docs(docs: Iterable[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain: Runnable = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke(question)
    print("Answer: ", answer)  # noqa: T201


if __name__ == "__main__":
    main()
