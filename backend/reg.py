from pydantic import BaseModel
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END, START



class user_storys_generator(TypedDict):
    input_file: str =""
    extract_text : str = ""
    summery : str =""
    project_requirement_document : str =""
    user_storys : str =""

def extract_text_from_dif_types(state :user_storys_generator):
    filePath =state.input_file
    ful_text = ""
    if filePath.endswith('mp4'):
        text = "i am video file"
        ful_text =text
        return ful_text
    elif filePath.endswith('.wav'):
        text1 = 'i am audio file'
        ful_text = text1
        return ful_text
    elif filePath.endswith('documen'):
        text2 ="i am document"
        ful_text = text2
    
        return ful_text
    state.extract_text = ful_text
    
    return state 

def summery_text(state :user_storys_generator ):
    pass 
    return state 
def project_requirement_document_from_text(state :user_storys_generator):
    pass
    return
def generate_user_story(state :user_storys_generator):
    pass
    return state



workflow = StateGraph(user_storys_generator)
workflow.add_node("process_text", extract_text_from_dif_types())
workflow.set_entry_point("process_text")
workflow.set_finish_point("process_text")
app = workflow.compile()

# Run the workflow







if __name__ == "__main__":
   # app = workflow.compile()
    
   # iinitial_state = user_storys_generator(input_file ="mp4")
  #  initial_state = user_storys_generator(input_file="path/to/file.mp3") 
   # result = app.invoke(initial_state)

    # ✅ Compile Workflow
    app = workflow.compile()

# ✅ Initialize State Properly
    initial_state = user_storys_generator(input_file="path/to/file.mp3")  # ✅ Correct way

# ✅ Run Workflow
    result = app.invoke(initial_state)  # ✅ Pass correct state instance

# ✅ Print Result
    print(result)