from typing import List, Optional
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

class MedicalDocumentLoader:
    def __init__(self):
        self.wiki_tool = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(
                top_k_results=5,
                doc_content_chars_max=500
            )
        )

    def load_web_documents(self, urls: List[str]) -> str:
        loader = WebBaseLoader(urls)
        docs = loader.load()

        with open("docs_cache.txt", "w", encoding="utf-8") as f:
            for doc in docs:
                text = doc.page_content.replace(". ", ".\n")
                f.write(text + "\n\n")
        return "\n\n".join(doc.page_content for doc in docs)
    
    def load_wiki_content(self, query: str) -> str:
        return self.wiki_tool.run({'query':query})
    
    def combined_context(self, urls: List[str], wiki_query: Optional[str] = None) -> str:
        web_content = self.load_web_documents(urls=urls)
        if wiki_query:
            wiki_content = self.load_wiki_content(wiki_query)
            return f"{web_content}\n\n{wiki_content}"
        return web_content