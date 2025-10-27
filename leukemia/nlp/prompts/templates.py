from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", 
        "You are an experienced medical assistant that generates detailed clinical reports "
        "based on leukemia-related data. "
        "Always use professional medical terminology and maintain factual accuracy."),
    
    ("human", 
        "Diagnosis type: {prediction}\n\n"
        "If the diagnosis is 'HEM' (hematologically normal), state clearly that the patient's blood "
        "profile is normal and no leukemia or abnormality is found.\n\n"
        "If the diagnosis is 'ALL' (acute lymphoblastic leukemia), prepare a detailed medical report including:\n"
        "- Brief overview of the disease\n"
        "- Common symptoms\n"
        "- Possible causes or risk factors\n"
        "- Diagnostic markers or findings\n"
        "- Recommended treatments (chemotherapy, bone marrow transplant, etc.)\n"
        "- Prognosis and follow-up care\n\n"
        "Use this context to support your findings:\n\n{context}")
])  