from langchain_groq import ChatGroq
from prompts import templates
from loaders.document_loader import MedicalDocumentLoader

class MedicalReportChain:
    def __init__(self, model_name: str = "qwen/qwen3-32b", temperature: float = 0.1):
        self.llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            max_tokens=5000,
            reasoning_format="parsed"
        )

        self.loader = MedicalDocumentLoader()

    def create_context(self) -> str:
        urls = ["https://my.clevelandclinic.org/health/diseases/21564-acute-lymphocytic-leukemia",
                "https://www.mayoclinic.org/diseases-conditions/acute-lymphocytic-leukemia/symptoms-causes/syc-20369077"]

        return self.loader.combined_context(
            urls=urls,
            wiki_query="Acute Lymphomatic Leulemia"
        )