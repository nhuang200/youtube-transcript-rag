import streamlit as st
import openai
from typing import Any, List, Optional, Tuple
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import DistanceStrategy
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_core.language_models import LLM
from langchain_core.embeddings import Embeddings
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain.prompts import PromptTemplate
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import os
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env



# --------------------------
# Custom Embeddings & LLM Wrappers (Unchanged)
# --------------------------
class GiteeEmbeddings(Embeddings):
    api_key: str
    base_url: str
    model_name: str
    default_headers: dict

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        default_headers: dict,** kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.default_headers = default_headers
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        retry=retry_if_exception_type((Exception,)),
        reraise=True
    )
    def _embed(self, text: str) -> List[float]:
        client = openai.OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            default_headers=self.default_headers
        )
        response = client.embeddings.create(
            input=text,
            model=self.model_name
        )
        return response.data[0].embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


class ThirdPartyLLM(LLM):
    api_key: str
    base_url: str
    model_name: str
    default_headers: dict = {}

    @property
    def _llm_type(self) -> str:
        return "third-party"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,** kwargs: Any,
    ) -> str:
        client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            default_headers=self.default_headers
        )
    
        completion = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return completion.choices[0].message.content


# --------------------------
# Initialize Components with Score Tracking
# --------------------------
@st.cache_resource(ttl=0)
def init_components():
    # Configuration (match your settings)
    DB_NAME = "rag_db"
    COLLECTION_NAME = "video_embeddings"
    
    THIRD_PARTY_API_KEY = "sk-aEZsRpQnHqrbZvXSAdDa9d8571244028880e34E36459Ca9f"
    THIRD_PARTY_BASE_URL = "https://free.v36.cm/v1/"
    THIRD_PARTY_MODEL = "gpt-4o-mini"
    THIRD_PARTY_HEADERS = {"x-foo": "true"}

    EMBEDDING_BASE_URL = "https://ai.gitee.com/v1"
    EMBEDDING_API_KEY = "BNPGOZQ2MKEBFHUTEWLNMIQEBFZFQOVZOYLQ241R"
    EMBEDDING_MODEL = "Qwen3-Embedding-8B"
    EMBEDDING_HEADERS = {"X-Failover-Enabled": "true"}

    # Embeddings and vector store
    embeddings = GiteeEmbeddings(
        api_key=EMBEDDING_API_KEY,
        base_url=EMBEDDING_BASE_URL,
        model_name=EMBEDDING_MODEL,
        default_headers=EMBEDDING_HEADERS
    )

    # Use cloud database connection string
    db_conn_str = st.secrets["CLOUD_DB_CONN_STR"]  # From environment variable
    vector_store = PGVector(
        connection=db_conn_str,
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        distance_strategy=DistanceStrategy.COSINE,
    )

    # LLM and prompt
    llm = ThirdPartyLLM(    
        api_key=THIRD_PARTY_API_KEY,
        base_url=THIRD_PARTY_BASE_URL,
        model_name=THIRD_PARTY_MODEL,
        default_headers=THIRD_PARTY_HEADERS
    )

    prompt_template = """Use the following context to answer the question. 
If you don't know the answer, say "Here are the videos closest to your query!" – do not make up information.

Context: {context}
Question: {question}
Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # Retriever (exact similarity search)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5, "search_type": "similarity",  
        "distance_strategy": DistanceStrategy.COSINE })

    return llm, vector_store, PROMPT, retriever


# --------------------------
# Helper: Get Docs with Scores
# --------------------------
def get_docs_with_scores(vector_store, query: str, k: int = 5) -> List[Tuple[Any, float]]:
    """Fetch documents with their similarity scores"""
    return vector_store.similarity_search_with_score(query, k=k)


# --------------------------
# Streamlit GUI
# --------------------------
def main():
    st.set_page_config(
        page_title="Video RAG Chat (with Scores)",
        page_icon="🎥",
        layout="wide"
    )

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "llm" not in st.session_state:
        with st.spinner("Initializing..."):
            st.session_state.llm, st.session_state.vector_store, st.session_state.PROMPT, st.session_state.retriever = init_components()

    # Title
    st.title("🎥 Video RAG Chat (with Similarity Scores)")
    st.write("Ask questions and see relevance scores for sources")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("Sources (with similarity scores)"):
                    for source, score in zip(message["sources"], message["scores"]):
                        st.markdown(f"- {source} (Score: {score:.4f})")  # Show score here

    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get documents with scores
                    docs_with_scores = get_docs_with_scores(
                        st.session_state.vector_store,
                        query=prompt,
                        k=5
                    )
                    relevant_docs = [doc for doc, _ in docs_with_scores]
                    scores = [score for _, score in docs_with_scores]  # Extract scores

                    # Create context from docs
                    context = "\n\n".join([doc.page_content for doc in relevant_docs])
                    formatted_prompt = st.session_state.PROMPT.format(context=context, question=prompt)

                    # Get answer from LLM
                    answer = st.session_state.llm(formatted_prompt)

                    # Prepare sources with titles/URLs
                    sources = [
                        f"{doc.metadata.get('title')} ({doc.metadata.get('url')})"
                        for doc in relevant_docs
                    ]

                    # Display answer
                    st.markdown(answer)

                    # Display sources with scores
                    with st.expander("Sources (with similarity scores)"):
                        for source, score in zip(sources, scores):
                            st.markdown(f"- {source} (Score: {score:.4f})")  # Show score

                    # Save to history (include scores)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "scores": scores  # Store scores in history
                    })

                except Exception as e:
                    st.error(f"Error: {str(e)}")

    # Sidebar: Show top documents with scores for last question
    with st.sidebar:
        st.subheader("Top Matching Documents")
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            last_question = st.session_state.messages[-1]["content"]
            with st.spinner("Loading scores..."):
                docs_with_scores = get_docs_with_scores(
                    st.session_state.vector_store,
                    query=last_question,
                    k=3
                )
                for doc, score in docs_with_scores:
                    st.write(f"**Title:** {doc.metadata.get('title', 'No title')}")
                    st.write(f"**Similarity Score:** {score:.4f} (lower = more relevant)")
                    st.write(f"**Snippet:** {doc.page_content[:200]}...")
                    st.divider()
        else:
            st.write("Ask a question to see matching documents and scores!")


if __name__ == "__main__":
    main()
