import streamlit as st
import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# Initialize LLM and tools
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)
tools = [TavilySearchResults(max_results=3, tavily_api_key=tavily_api_key)]
system_prompt = '''
You are a finance expert named Wakasiaka in the year 2024. Your goal is to conduct deep research on financial risks.

1️⃣ **Create a structured research plan** before answering.  
2️⃣ **Use multiple sources** to ensure completeness.  
3️⃣ **Cover all key risks** including **regulatory, economic downturns, and cybersecurity**.  
4️⃣ **Provide a final verdict** based on gathered insights.

Your responses should be accurate, complete, and well-structured.
'''
agent_executor = create_react_agent(llm, tools, state_modifier=system_prompt)

# Streamlit App Title
st.set_page_config(page_title="AI Financial Research Agent", layout="wide")
st.title("📊 AI Financial Research Agent")

# User Input
question = st.text_area("🔍 Enter a financial research question:", "What are the key financial risks of investing in tech startups?")

if st.button("Run Research Agent"):
    with st.spinner("🤖 Researching... Please wait..."):
        # Get agent response
        agent_response = agent_executor.invoke({"input": question})
        agent_response_text = agent_response.get("output", "No response generated.")

        # Display Response
        st.subheader("📢 Agent Response:")
        st.write(agent_response_text)

        # Ideal Answer (Static for now, but can be dynamically retrieved)
        ideal_response = "Key financial risks in tech startups include high failure rates, market competition, regulatory issues, cybersecurity risks, and economic downturn impacts."

        # LLM as a Judge Function
        def evaluate_agent(agent_response, ideal_response, question):
            """
            Uses an LLM as a judge to evaluate the agent's response compared to an ideal answer.
            """
            evaluation_prompt = PromptTemplate.from_template("""
            You are an impartial judge evaluating AI-generated responses.

            **Question:** {question}

            **Agent's Response:** {agent_response}

            **Ideal Response:** {ideal_response}

            Evaluate the agent's response based on:
            - **Accuracy (1-5):** Does the response contain correct information?
            - **Relevance (1-5):** How relevant is the response to the question?
            - **Completeness (1-5):** Does the response cover all necessary aspects?

            Provide scores and a brief explanation in JSON format:

            {{"accuracy": X, "relevance": Y, "completeness": Z, "explanation": "..."}}
            """)

            judge_llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key)
            response = judge_llm.invoke(evaluation_prompt.format(
                agent_response=agent_response,
                ideal_response=ideal_response,
                question=question
            ))

            try:
                return json.loads(response.content)
            except json.JSONDecodeError:
                return {"error": "Failed to parse evaluation response", "raw_response": response.content}

        # Run Evaluation
        evaluation_result = evaluate_agent(agent_response_text, ideal_response, question)

        # Display Evaluation Results
        st.subheader("📊 Evaluation Results:")
        st.json(evaluation_result)
