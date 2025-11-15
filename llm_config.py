from langchain_google_genai import ChatGoogleGenerativeAI
import os

# Initialize the LLM (ensure GOOGLE_API_KEY is set in environment)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.2
)
