# 🐢🔗 https://testudoai.streamlit.app/

# What it does
TestudoAI is an AutoGPT App where users can ask questions about University of Maryland Courses (course descriptions, average gpa, section specific information, etc.), 
Professors (average rating on PlanetTerp, reviews, what courses they teach, etc.), grade data for courses and professors, and more! With TestudoAI, users have the 
power of PlanetTerp and Testudo at their fingertips.

# How to use it
If you have a question, just ask. TestudoAI has a familiar ChatGPT-like frontend interface. You'll need to supply your own OpenAI API key first, which you can get from 
https://platform.openai.com/account/api-keys. There are some example questions on the left to give you an idea of what you can ask.

# How it's made
TestudoAI is <strong>not</strong> a pre-trained model. It is using vanilla GPT-3.5 Turbo as the LLM (Large Language Model). So how does it know all about the University of 
Maryland's courses, sections, professors, and more? [LangChain](https://python.langchain.com/docs/get_started/introduction.html) is what makes this possible. At a basic level, 
LangChain's [MRKL](https://arxiv.org/abs/2205.00445) Agent is able to leverage custom-made tools to turn a user request into action (as in calling those tools). TestudoAI has 
8 different tools, all of which call the [umd.io](https://beta.umd.io/) and/or the [PlanetTerp](https://planetterp.com/api/) API's. Because these APIs stay up-to-date with
course seats, grade data, and professor reviews, TestudoAI is able to be dynamic in that it is providing accurate and up-to-date information where a pre-trained model could 
not. 
### Grade Data
Implemented with [Matplotlib](https://matplotlib.org/3.5.3/api/_as_gen/matplotlib.pyplot.html), TestudoAI is able to supply the grade distribution for a desired course or professor (or both!). Simply ask for the grade data users will be given a pie chart showing grade data that is up to data with PlanetTerp.
<h3 align="center">
<img src="public/gr_data_ss.png" alt="Logo" width="600px">
</a>
</h3>

### Conversational Memory
Conversational Memory has been implemented with TestudoAI, meaning that it remembers past interactions as context. The result is that talking to TestudoAI has a natural 
flow of conversation without the user having to unnecessarily repeat information.
<h3 align="center">
<img src="public/conv_mem_ss.png" alt="Logo" width="600px">
</a>
</h3>

### Vector Stores, Embeddings, and FAISS Similarity Search
A big part of what makes TestudoAI useful is that it can access large amounts of data supplied by the previously mentioned APIs and give the user only the relevant summarized version
of that data. However, this was actually challenging to implement since the GPT-3.5 LLM can be supplied a maximum of 4096 tokens (around 3000 words). This was a problem because 
all of the review data for one professor could easily surpass that max token count. To be able to give the LLM the review data without surpassing the max token count, the following
had to be implemented: The review data requested from the API had to first be split up into chunks of 800 tokens, converted to something called [Vector Embeddings](https://cloud.google.com/blog/topics/developers-practitioners/meet-ais-multitool-vector-embeddings), then put into a [Vector Store](https://python.langchain.com/docs/modules/data_connection/vectorstores/) which provide efficient storage and retrieval by storing data as vectors. Finally, [FAISS Similary Search](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/) was performed to retrieve only the relevent chunks of review data based on the user request.
