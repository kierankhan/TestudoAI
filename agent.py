import requests
from langchain.llms import OpenAI
from typing import Optional, Type
from langchain import LLMMathChain, SerpAPIWrapper, LLMChain
from langchain.agents import AgentType, initialize_agent, ZeroShotAgent, AgentExecutor, ConversationalChatAgent
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.agents.chat.prompt import (
    FORMAT_INSTRUCTIONS,
    HUMAN_MESSAGE,
    SYSTEM_MESSAGE_PREFIX,
    SYSTEM_MESSAGE_SUFFIX,
)
import ai_tools
from langchain.memory import ConversationBufferMemory
import datetime
import os.path
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)


embeddings = OpenAIEmbeddings()
chat = ChatOpenAI(model="gpt-3.5-turbo")


def create_db_from_review_data(review_data):
    loader = TextLoader(review_data)
    reviews = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(reviews)
    db = FAISS.from_documents(docs, embeddings)
    sim_search = db.similarity_search(request)

    return sim_search

tools = [
    ai_tools.GetCourseTool(),
    ai_tools.GetCoursesAsListTool(),
    ai_tools.GetProfsForCourseTool(),
    ai_tools.GetProfInfoTool(),
    ai_tools.SearchTool(),
    ai_tools.GetGradeDataTool(),
    ai_tools.GetProfReviews(),
    ai_tools.GetSectionTool()
]


prefix = """You are a Planet Terp AI Assistant that helps students with getting information on classes and professors so that they
may make informed decisions on which classes to take. Course names are identified as four letters followed by three numbers with no separation. Examples
include 'math141', 'CMSC330', 'chem135', 'MATH410'. Answer the following requests as best you can. When using tools that take a course name as input, 
make sure to stick with the proper format for course names. You have access to the following tools:"""
suffix = """Begin! Reminder to always use the exact characters `Final Answer` when responding.

{chat_history}
Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=SYSTEM_MESSAGE_PREFIX,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"],
)

conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)

llm = OpenAI(temperature=0)


llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True, max_iterations=3)
agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=3,
    memory=conversational_memory
)
# agent = initialize_agent(
#     tools,
#     llm,
#     agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=True,
#     max_iterations=3,
#     memory=conversational_memory
# )


request = input("What can I help you with? (Press q to quit) ")
while request != "q":
    agent_chain.run(f"Request: {request}")
    # conversational_memory.chat_memory.add_user_message(request)
    # conversational_memory.chat_memory.add_ai_message(response)
    # print(conversational_memory.chat_memory.messages)
    request = input("What can I help you with? ")

# grade_data = requests.get(f"https://api.planetterp.com/v1/grades?course={course_name}").json()

