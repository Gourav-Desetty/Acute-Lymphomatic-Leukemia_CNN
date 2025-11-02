from leukemia.nlp.chains.report_chain import MedicalReportChain
from leukemia.nlp.prompts.templates import template

class LeukemiaReportGenerator:
    def __init__(self) -> None:
        self.report_chain = MedicalReportChain()

    def generate_report(self, prediction: str, confidence: float):
        context = self.report_chain.create_context()
        prompt = template.format(
            prediction=prediction,
            context=context
        )
        response = self.report_chain.llm.invoke(prompt)
        return response.content