from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def main() -> None:
    load_dotenv()
    question = input("Question?: ")

    loader = DirectoryLoader(
        "",
        glob=["README.md", "docs/**/*.md"],
        recursive=True,
    )

    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
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

    def format_docs(docs: VectorStore) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke(question)
    print("Answer: ", answer)  # noqa: T201


if __name__ == "__main__":
    main()
