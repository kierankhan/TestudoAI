from typing import Optional

import requests
from langchain import LLMChain
from langchain.agents import ZeroShotAgent, AgentExecutor
from langchain.agents.chat.prompt import (
    SYSTEM_MESSAGE_PREFIX,
)
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import BaseTool
from langchain.vectorstores import FAISS
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import json
import streamlit as st
from langchain.agents.chat import base



# chat = ChatOpenAI(model="gpt-3.5-turbo")


def create_db_from_review_data(review_data: str, chunk_size: int, overlap: int):
    loader = TextLoader(review_data)
    reviews = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    docs = text_splitter.split_documents(reviews)
    db = FAISS.from_documents(docs, embeddings)
    sim_search = db.similarity_search(request)

    return sim_search


# TOOLS
class GetCourseTool(BaseTool):
    name = "get_course"
    description = "Use this tool when you need to get information for a specific course, such as a " \
                  "description, number of credits, gen ed requirments, prerequisites, sections, and more. " \
                  "To use the tool you must provide only the following parameter ['course_name'] " \
                  "ONLY USE THE ONE PARAMETER ['course_name'] AS THE INPUT AND NOTHING ELSE! " \
                  "When providing information on the course make sure to include a short summary " \
                  "of the course description, how many credits it is, any prerequisites there are, " \
                  "and the average GPA at minimum. Also mention if the course fulfills any gen_ed " \
                  "requirements, if any."

    def _run(
        self, course_name: str
    ):
        """Use the tool, but only provide one parameter with the name 'course_name'"""
        query = f"https://api.umd.io/v1/courses/{course_name}"
        avg_gpa = requests.get(f"https://planetterp.com/api/v1/course?name={course_name}").json()["average_gpa"]
        data = requests.get(query).json()
        data[0]["average_gpa"] = avg_gpa
        return data

    async def _arun(self):
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")

class CourseSearchTool(BaseTool):
    name = "course_search"
    description = "Use this tool when you need to search for a course based on inputs OTHER THAN the course name. Your " \
                  "input to this tool should be a comma separated list of strings with the parameter name and value desired. " \
                  " The three possible parameters are 'credits', 'gen_ed', and 'dept_id'. Input the corresponding values " \
                  "to the parameters based on what the user has said. " \
                  "For example, 'credits:4,dept_id:MATH' would be the input if you wanted to search courses that are" \
                  "4 credits in the math department. Another example is 'gen_ed:DVUP' if you wanted to search for " \
                  "courses that fulfill the DVUP requirement. Your input should be at least one of these three parameters: " \
                  "'credits', 'gen_ed', and 'dept_id'. The gen_ed codes are four letter acronyms and include only the following: " \
                  "'DSNS', 'FSAR', 'DVUP', 'FSAW', 'FSMA', 'FSOC', 'FSPW', 'DSHS', 'DSNL', 'DSSP' 'DVCC', and 'SCIS'" \
                  "The department id codes are also four letter acronyms (these include any four letter acronyms OTHER THAN the gen ed acronyms" \
                  "If a link is returned from the tool, it is because the request exceeded the allotted token count and therefore " \
                  "we must send them to the testudo website. Remember, the purpose of this tool is to give users a list" \
                  "of courses that match their criteria, no need to do anything extra after that. Also, if you observe " \
                  "that no courses matched the search criteria, that is ok--let the user know and await their response." \

    def _run(
        self,
        all: Optional[str] = None,
    ):
        # print("gened: " + gen_ed + ", credits: " + credits + ", dept_id: " + dept_id)
        credits = None
        gen_ed = None
        dept_id = None
        if all.find("credits:") != -1: credits = all[all.find("credits")+8]
        if all.find("gen_ed") != -1: gen_ed = all[all.find("gen_ed")+7:all.find("gen_ed")+11]
        if all.find("dept_id") != -1: dept_id = all[all.find("dept_id")+8:all.find("dept_id")+12]
        # if credits is not None: credits = int(credits[-1])
        # if gen_ed is not None: gen_ed = gen_ed[-4:-1]
        # if dept_id is not None: dept_id = int(dept_id[len(dept_id)-1])
        """Use the tool, but only provide one parameter with the name 'course_name'"""
        query = "https://api.umd.io/v1/courses?sort=course_id,-credits&per_page=50&page=1"
        if credits is not None:
            query += f"&credits={credits}"
        else:
            credits = "3"
        if gen_ed is not None:
            query += f"&gen_ed={gen_ed}"
        else:
            gen_ed = ""
        if dept_id is not None:
            query += f"&dept_id={dept_id}"
        else:
            dept_id = ""
        course_data = requests.get(query).json()
        query += "&page=2"
        course_data2 = requests.get(query).json()
        if json.dumps(course_data2) != "[]":
            testudo_link = "https://app.testudo.umd.edu/soc/search?courseId="+dept_id+"&sectionId=&termId=202308&" \
                           "_openSectionsOnly=on&creditCompare=%3D&credits="+str(credits)+"&courseLevelFilter=ALL&instructor=&" \
                           "_facetoface=on&_blended=on&_online=on&courseStartCompare=&courseStartHour=" \
                           "&courseStartMin=&courseStartAM=&courseEndHour=&courseEndMin=&courseEndAM=" \
                           "&teachingCenter=ALL&_classDay1=on&_classDay2=on&_classDay3=on&_classDay4=on&_classDay5=on"
            return "Unfortunately I cannot analyze that many courses--here is the link to all of them on Testudo: " + testudo_link
        print(course_data2)
        minified_data = []
        for i in course_data:
            dict = {}
            dict["course_id"] = i["course_id"]
            dict["dept_id"] = i["dept_id"]
            dict["department"] = i["department"]
            dict["credits"] = i["credits"]
            dict["gen_ed"] = i["gen_ed"]
            minified_data.append(dict)
        course_data_str = json.dumps(minified_data)
        print(query)
        # text = raw_data.text
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        # docs = text_splitter.split_documents(text)

        # f = open("courses.txt", "w")
        # f.write(course_data_str)
        # docs = create_db_from_review_data("courses.txt", 1000, 100)

        if len(minified_data) > 0: return minified_data
        return "There are no courses that match this criteria"

    async def _arun(self):
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")

class GetProfsForCourseTool(BaseTool):
    # TODO: Make the amount of professors you get back a parameter
    name = "get_profs_for_course"
    description = "Use this tool when you need to get the professors (also known as 'profs', " \
                  "'instructors', or 'teachers') that teach a specific course. " \
                  "To use the tool you must provide only the following parameter ['course_name'] " \
                  "ONLY USE THE ONE PARAMETER ['course_name'] AS THE INPUT AND NOTHING ELSE!" \
                  "Course names are identified as four letters followed by three numbers with no separation. Examples " \
                  "include 'math141', 'CMSC330', 'chem135', 'MATH410'. Make sure your input to this tool is in this" \
                  "format with NO SPACES BETWEEN ANY LETTER OR NUMBER!" \
                  "List the most recent professors, and make sure to say that those are the most recent in your response"

    def _run(
        self, course_name: str
    ):
        """Use the tool, but only provide one parameter with the name 'course_name'"""
        query = f"https://planetterp.com/api/v1/course?name={course_name}"
        data = requests.get(query).json()

        return data["professors"][-6:]

    async def _arun(self):
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")

class GetProfInfoTool(BaseTool):
    name = "get_profs_info"
    description = "Use this tool when you need to get information about a specific professor (also known as 'profs', " \
                  "'instructors', or 'teachers'). This includes what courses the professors teach, average gpa, and more. " \
                  "To use the tool you must provide only the following parameter ['prof_name'] " \
                  "ONLY USE THE ONE PARAMETER ['prof_name'] AS THE INPUT AND NOTHING ELSE!" \
                  "The input to this tool should be the professors full name as given by the user. Provide a short " \
                  "summary of the professor, including what courses he/she teaches, type, and average rating."

    def _run(
        self, prof_name: str
    ):
        #prof_name = prof_name[10:]
        """Use the tool, but only provide one parameter with the name 'course_name'"""
        query = f"https://planetterp.com/api/v1/professor?name={prof_name}"
        data = requests.get(query).json()

        return data

    async def _arun(self):
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")

class GetProfReviews(BaseTool):
    name = "get_profs_reviews"
    description = "Use this tool when you need to get the reviews for a specific professor (also known as profs, " \
                  "instructors, or teachers). If the user wants to know if a professor is good or bad, or other " \
                  "students' opinion on the professor, use this tool. You will receive documents with reviews of the" \
                  "professor. Please only use factual information that you get from the documents provided. Your answers" \
                  " should be verbose and detailed, and most importantly they should answer the USER'S ORIGINAL QUESTION. " \
                  "Please make your response around a paragraph long. The input to this tool should be the professors" \
                  "name with no quotation marks." \
                  "To use the tool you must provide only the following parameter ['prof_name'] " \
                  "ONLY USE THE ONE PARAMETER ['prof_name'] AS THE INPUT AND NOTHING ELSE!" \

    def _run(
        self, prof_name: str
    ):
        """Use the tool, but only provide one parameter with the name 'course_name'"""
        query = f"https://planetterp.com/api/v1/professor?name={prof_name}&reviews=true"
        data = requests.get(query).json()["reviews"]
        review_data = ""
        for i in data:
            review_data += i["review"]
        f = open("reviews.txt", "w")
        f.write(review_data)
        docs = create_db_from_review_data("reviews.txt", 500, 50)
        # db.similarity_search(request)

        return docs

    async def _arun(self):
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")

class GetGradeDataTool(BaseTool):
    name = "get_grade_data"
    description = "Use this tool when you need to get the grade data for a specific course or professor" \
                  "To use the tool you must provide at least one of the following parameters ['course', 'professor']" \
                  "Your input to this tool should be a comma separated list of strings with the parameter name " \
                  "and value desired ending with a semicolon. Also, if you need to input a professor, make sure to " \
                  "put it at the end of the list. For example, 'semester:202108,professor:larry herman;' " \
                  "would be the input if you wanted the grade data for professor larry herman in the Fall 2021. " \
                  "Another example is 'course:INST154;' if you just wanted the grade data for that specific course. " \
                  "If the user provides a semester, use that as the input ['semester']. The input to semester will " \
                  "be a six digit number where the first four digits are the year and the last two numbers specify " \
                  "fall or spring. 01 means Spring and 08 means Fall. For example, 202001 means Spring 2020." \
                  "Your response should the course name and/or professor and a very brief summarization of the grade data " \
                  ". You should draw conclusions based off this data on whether this is a favorable grade distribution or not."

    def _run(
        self,
        all: Optional[str] = None
    ):
        course = None
        professor = None
        semester = None
        if all.find("course") != -1: course = all[all.find("course")+7:all.find("course")+14]
        if all.find("professor") != -1: professor = all[all.find("professor") + 10 : all.find(";")]
        if all.find("semester") != -1: semester = all[all.find("semester") + 9 : all.find("semester")+15]
        query = "https://planetterp.com/api/v1/grades?"
        if course is not None and professor is not None:
            query += f"course={course}&professor={professor}"
        elif professor is not None:
            query += f"professor={professor}"
        else:
            query += f"course={course}"

        if semester is not None:
            query += f"&semester={semester}"
        print(query)
        """Use the tool"""
        raw_data = requests.get(query)
        json_data = raw_data.json()

        total = 0

        result = {}
        for i in json_data[-40:]:
            result["A+"] = int(result.get("A+", "0")) + int(i["A+"])
            result["A"] = int(result.get("A", "0")) + int(i["A"])
            result["A-"] = int(result.get("A-", "0")) + int(i["A-"])
            result["B+"] = int(result.get("B+", "0")) + int(i["B+"])
            result["B"] = int(result.get("B", "0")) + int(i["B"])
            result["B-"] = int(result.get("B-", "0")) + int(i["B-"])
            result["C+"] = int(result.get("C+", "0")) + int(i["C+"])
            result["C"] = int(result.get("C", "0")) + int(i["C"])
            result["C-"] = int(result.get("C-", "0")) + int(i["C-"])
            result["D"] = int(result.get("D", "0")) + int(i["D+"])
            result["D"] = int(result.get("D", "0")) + int(i["D"])
            result["D"] = int(result.get("D", "0")) + int(i["D-"])
            result["F"] = int(result.get("F", "0")) + int(i["F"])
            result["W"] = int(result.get("W", "0")) + int(i["W"])

            # plt.rcParams["figure.figsize"] = [7.00, 3.50]
            # plt.rcParams["figure.autolayout"] = True
        names = list(result.keys())
        vals = list(result.values())
        total = 0.0
        for i in vals:
            total += i
        for j in range(0, len(vals)):
            vals[j] = (vals[j] /total) * 100

        colors = [
            '#63c27c', '#55ab6c', '#47915b', '#6bb3c7', '#60a3b5', '#5693a3', '#b88dd6', '#a982c4', '#9876b0', '#c286a7',
            # '#ab7693', '#996a84',
            '#c94b5b', '#d1904f'
        ]
        plt.pie(x=vals, labels=names, autopct='%1.0f%%', colors=colors, startangle=90, pctdistance=0.83,)
        hole = plt.Circle((0, 0), 0.65, facecolor='white')
        plt.gcf().gca().add_artist(hole)
        if professor is not None:
            title = professor
        if course is not None:
            title += " " + course
        # plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%d%%'))
        plt.title(title + " Grade Data")
        plt.show()
        st.pyplot(plt)


        return result

    async def _arun(self):
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")

class GetSectionTool(BaseTool):
    name = "get_section"
    description = "Use this tool when you need to get information about a specific section. A section ID is the " \
                  "course name followed by a dash and a four digit number. The following are examples of section IDs: " \
                  "MATH141-0101, CMSC132-0206, CHEM135-0302, ENGL101-0401" \
                  "To use the tool you must provide only the following parameter ['section_id'] " \
                  "ONLY USE THE ONE PARAMETER ['section_id'] AS THE INPUT AND NOTHING ELSE!" \
                  "The input to this tool should be the section ID"

    def _run(
        self, section_id: str
    ):
        """Use the tool, but only provide one parameter with the name 'course_name'"""
        query = f"https://api.umd.io/v1/courses/sections/{section_id}"
        data = requests.get(query).json()

        return data

    async def _arun(self):
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")

class SearchTool(BaseTool):
    name = "search_planetterp"
    description = "Use this tool when a course or professor could no be found. MAKE SURE TO USE THE TOOL WHEN YOU" \
                  "RECIEVE 'error': 'course not found'. To use the tool you must provide " \
                  "only the following parameter ['search']. ONLY USE THE ONE PARAMETER ['search'] AS THE " \
                  "INPUT AND NOTHING ELSE! The input to this tool should be the professors full name or course name " \
                  "as given by the user. RETURN TO THE USER A LIST of the names of the courses or professors returned " \
                  "by the search so that the USER can decide which result is correct. RETURN A LIST OF THE SEARCH " \
                  "RESULTS AND DO NOTHING ELSE!! If the search came up with nothing, say 'No results found'."

    def _run(
        self, search: str
    ):
        """Use the tool, but only provide one parameter with the name 'course_name'"""
        query = f"https://planetterp.com/api/v1/search?query={search}"
        data = requests.get(query).json()

        result = []
        for i in data:
            result.append({"name": i["name"]})

        return result

    async def _arun(self):
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")


##################################################################################################################
###################################TOOLS DONE#####################################################################
tools = [
    GetCourseTool(),
    CourseSearchTool(),
    GetProfsForCourseTool(),
    GetProfInfoTool(),
    SearchTool(),
    GetGradeDataTool(),
    GetProfReviews(),
    GetSectionTool()
]

prefix = """You are a Planet Terp AI Assistant that helps students with getting information on classes and professors so that they
may make informed decisions on which classes to take. Course names are identified as four letters followed by three numbers with no separation. Examples
include 'math141', 'CMSC330', 'chem135', 'MATH410'. Answer the following requests as best you can. When using tools that take a course name as input, 
make sure to stick with the proper format for course names. Please be helpful, but do NOT do more than you think is necessry, 
like taking extra steps when they are not needed. You have access to the following tools:"""
suffix = """Begin!"

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

#llm = OpenAI(temperature=0)


# agent = initialize_agent(
#     tools,
#     llm,
#     agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=True,
#     max_iterations=3,
#     memory=conversational_memory
# )
###################################################################################################

st.set_page_config(page_title='🦜🔗 TestudoAI')
st.title('🦜🔗 TestudoAI')



if "openai_api_key" in st.secrets:
    openai_api_key = st.secrets.openai_api_key
else:
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Enter an OpenAI API Key to continue")
    st.stop()

def generate_response(input_query):
    print(openai_api_key)
    print("SDOFDOSF")
    response = agent_chain.run(input_query)
    return st.success(response)



query_text = st.chat_input(placeholder='Ask me anything course/professor related!',
                           disabled=not openai_api_key)


llm_chain = LLMChain(llm=OpenAI(temperature=0, openai_api_key=openai_api_key), prompt=prompt)
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True, max_iterations=3)
agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=3,
    memory=conversational_memory,
    early_stopping_method="generate"
)

if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
if query_text and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            st.write(agent_chain.run(query_text))

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if def_prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(def_prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": def_prompt})

    response = f"Echo: {prompt}"
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

request = input("What can I help you with? (Press q to quit) ")
# while request != "q":
#     agent_chain.run(f"Request: {request}")
#     # conversational_memory.chat_memory.add_user_message(request)
#     # conversational_memory.chat_memory.add_ai_message(response)
#     # print(conversational_memory.chat_memory.messages)
#     request = input("What can I help you with? ")

# grade_data = requests.get(f"https://api.planetterp.com/v1/grades?course={course_name}").json()
