import os

import google.ai.generativelanguage as glm
import google.generativeai as genai

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

for m in genai.list_models():
    if "embedContent" in m.supported_generation_methods:
        print(m.name)

title = "The next generation of AI for developers and Google Workspace"
sample_text = (
    "Title: The next generation of AI for developers and Google Workspace"
    "\n"
    "Full article:\n"
    "\n"
    "Gemini API & Google AI Studio: An approachable way to explore and prototype with generative AI applications"
)

model = "models/embedding-001"
embedding = genai.embed_content(
    model=model, content=sample_text, task_type="retrieval_document", title=title
)

print(embedding)
