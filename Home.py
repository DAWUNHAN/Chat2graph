from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores.faiss import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable.base import Runnable


from langchain_anthropic import ChatAnthropic
import streamlit as st

from dotenv import load_dotenv
load_dotenv() 

st.set_page_config(
    page_title="chat2graph", 
    page_icon="🌳"
)

llm = ChatOpenAI(
    model_name="gpt-4-0613",
    temperature=0.1,
    streaming=True
)

def send_message(message, role):
    with st.chat_message(role):
        st.markdown(message)

def _sanitize_output(text: str):
    # 입력된 텍스트에서 파이썬 코드 블록을 추출합니다.
    _, after = text.split("```python")
    # 추출된 코드 블록에서 코드 부분만 반환합니다.
    return after.split("```")[0]


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a powerful code Generation AI. You need to write code as follows:
            Write Python code using Manim to create an animated graph based on the user's description. The code should:
            1. Include all necessary library imports for running the Python code with Manim.
            2. Begin with the magic command '%%manim -ql -v WARNING Graph' at the beginning of the code cell to ensure it runs with appropriate settings.
            3. Define the class 'Graph' that extends from Manim's 'Scene' class, which will be used to render the graph animation.
            4. Set appropriate x-axis and y-axis ranges in the graph to ensure the user can clearly observe the output. Ensure that the x-axis range does not include values that could result in undefined operations for the functions being plotted, especially logarithmic functions which are not defined for non-positive values.
            5. Explicitly not use 'get_graph' for plotting. Instead, use 'axes.plot' for all graph plotting, ensuring to specify color and other properties directly within this method.
            6. Ensure all data processed through 'axes.plot' is free from 'inf' and 'NaN' values by appropriately managing the range of x-values, especially avoiding negative or zero values for logarithmic functions.
            7. Avoid 'get_graph' entirely in your implementation, focusing solely on 'axes.plot' to enhance control over the visual output.
            8. Illustrate the construction of the graph step-by-step, including the drawing of axes, functions, intersection points, and shaded areas between these points under f(x) and above g(x).
            9. Animate the addition of each element to the scene sequentially to enhance understanding of the graph's components.
            10. Write only Python code without any additional explanation. Do not use 'get_x_axis_label' and avoid explanations outside of code comments.
            11. Include comments within the code to explain each step, especially the use of symbolic computation for finding intersections and the choice of axis ranges.
            12. Ensure the animation clearly illustrates the graph as described by the user, dynamically showing the development of the graph lines or curves as appropriate, and guaranteeing all calculations are within valid ranges. Ensure to use Manim Community v0.18.0.post0.
            
            """
        ),
        ("human", "{question}")
    ]
)


st.title("chat2graph")

st.markdown(
    """
### Welcome to chat2graph!
            
"""
)
message = st.chat_input("Tell me what you want to draw")
if message:
    try:
        st.info("Sending your request to the AI...")
        chain = {
            "question": RunnablePassthrough()
        } | prompt | llm 
        result = chain.invoke(message)
        st.success("Response received!")
        st.code(result.content, language='python')  
        print(result)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        import traceback
        st.text(traceback.format_exc()) 
