from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
import subprocess
import streamlit as st
import os

from dotenv import load_dotenv
load_dotenv() 

st.set_page_config(
    page_title="chat2graph", 
    page_icon="🌳"
)

llm = ChatOpenAI(
    model_name="gpt-4o",
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
            2. Define the class 'Graph' that extends from Manim's 'Scene' class, which will be used to render the graph animation.
            3. Set appropriate x-axis and y-axis ranges in the graph to ensure the user can clearly observe the output. Ensure that the x-axis range does not include values that could result in undefined operations for the functions being plotted, especially logarithmic functions which are not defined for non-positive values. When plotting the intersections of the graphs, set the x-axis and y-axis ranges to ensure the intersections are clearly visible.
            4. Explicitly not use 'get_graph' for plotting. Instead, use 'axes.plot' for all graph plotting, ensuring to specify color and other properties directly within this method.
            5. Ensure all data processed through 'axes.plot' is free from 'inf' and 'NaN' values by appropriately managing the range of x-values, especially avoiding negative or zero values for logarithmic functions.
            6. Avoid 'get_graph' entirely in your implementation, focusing solely on 'axes.plot' to enhance control over the visual output.
            7. Illustrate the construction of the graph step-by-step, including the drawing of axes, functions, intersection points, and shaded areas between these points under f(x) and above g(x).
            8. Animate the addition of each element to the scene sequentially to enhance understanding of the graph's components.
            9. Write only Python code without any additional explanation. Do not use 'get_x_axis_label' and avoid explanations outside of code comments.
            10. Include comments within the code to explain each step, especially the use of symbolic computation for finding intersections and the choice of axis ranges.
            11. Ensure the animation clearly illustrates the graph as described by the user, dynamically showing the development of the graph lines or curves as appropriate, and guaranteeing all calculations are within valid ranges. Ensure to use Manim Community v0.18.0.post0.
            
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
message = st.chat_input("그리고 싶은 그래프에 대해 자세히 설명해주세요.")
if message:
    try:
        st.info("Sending your request to the AI...")
        chain = {
            "question": RunnablePassthrough()
        } | prompt | llm 
        result = chain.invoke(message)
        st.success("Response received!")

        # Extract the code from the response
        code_content = _sanitize_output(result.content)
        st.code(code_content, language='python')

        # Save the code to a .py file
        with open("graph.py", "w") as file:
            file.write(code_content)

        # Execute the code to generate the video
        st.info("Generating the video...")
        result = subprocess.run(["manim", "-ql", "-v", "WARNING", "graph.py", "Graph"], capture_output=True, text=True)

        if result.returncode == 0:
            st.success("Video generated successfully!")
            video_path = "./media/videos/graph/480p15/Graph.mp4"  # adjust path based on Manim output

            # Display the video in Streamlit
            st.video(video_path)
        else:
            st.error("Error generating video!")
            st.text(result.stderr)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        import traceback
        st.text(traceback.format_exc())