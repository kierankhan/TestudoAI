from typing import Optional

import requests
from langchain.tools import BaseTool

import agent


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

class GetCoursesAsListTool(BaseTool):
    name = "get_courses_as_list"
    description = "Use this tool when you need to get a list of courses in a specific department.  " \
                  "To use the tool you must provide only the following parameter ['dep_name'] " \
                  "ONLY USE THE ONE PARAMETER ['dep_name'] AS THE INPUT AND NOTHING ELSE! " \
                  "Your response should include a list of the course names, titles, and number of " \
                  "credits. LIST IN ASCENDING ORDER IN COURSE NAME"

    def _run(
        self, dep_name: str
    ):
        """Use the tool, but only provide one parameter with the name 'course_name'"""
        raw_data = requests.get(f"https://planetterp.com/api/v1/courses?department={dep_name}")
        json_data = raw_data.json()
        data = []
        # text = raw_data.text
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        # docs = text_splitter.split_documents(text)

        for i in json_data:
            to_add = {"name": i["name"]}
            data.append(to_add)

        return data

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
                  "'instructors', or 'teachers'). This includes what courses the professors teach, grade data, and more. " \
                  "To use the tool you must provide only the following parameter ['prof_name'] " \
                  "ONLY USE THE ONE PARAMETER ['prof_name'] AS THE INPUT AND NOTHING ELSE!" \
                  "The input to this tool should be the professors full name as given by the user. Provide a short " \
                  "summary of the professor, including what courses he/she teaches, type, and average rating."

    def _run(
        self, prof_name: str
    ):
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
                  "students' opinion on the professor, use this tool. You will documents with reviews of the" \
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
        docs = agent.create_db_from_review_data("reviews.txt")
        # db.similarity_search(request)

        return docs

    async def _arun(self):
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")

class GetGradeDataTool(BaseTool):
    name = "get_grade_data"
    description = "Use this tool when you need to get the grade data for a specific course or professor" \
                  "To use the tool you must provide at least one of the following parameters ['course', 'professor']" \
                  "Do NOT input a course by doing 'course: [course]', just the course name will do. " \
                  "MUST PROVIDE AT LEASE ONE OF EITHER A COURSE NAME OR PROFESSOR NAME AS THE INPUT! If the user provides " \
                  "a semester, use that as the input ['semester']. The input to semester will be a six digit " \
                  "number where the first four digits are the year and the last two numbers specify fall or spring. " \
                  "01 means Spring and 08 means Fall. For example, 202001 means Spring 2020." \
                  "Your response should the course name and/or professor and ALL of the grade data. You should " \
                  "draw conclusions based off this data on whether this is a favorable grade distribution or not."

    def _run(
        self,
        course: Optional[str] = None,
        professor: Optional[str] = None,
        semester: Optional[int] = None
    ):
        query = "https://planetterp.com/api/v1/grades?"
        if course is not None:
            query += f"course={course}"
        elif professor is not None:
            query += f"professor={professor}"
        else:
            query += f"course={course}&professor={professor}"

        if semester is not None:
            query += f"&semester={semester}"

        """Use the tool"""
        raw_data = requests.get(query)
        json_data = raw_data.json()

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
            result["D+"] = int(result.get("D+", "0")) + int(i["D+"])
            result["D"] = int(result.get("D", "0")) + int(i["D"])
            result["D-"] = int(result.get("D-", "0")) + int(i["D-"])
            result["F"] = int(result.get("F", "0")) + int(i["F"])
            result["W"] = int(result.get("W", "0")) + int(i["W"])


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