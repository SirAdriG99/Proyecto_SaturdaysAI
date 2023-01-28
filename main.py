import gradio as gr

def greet( text ):
    return "Hi " + text + "!" 

demo = gr.Interface( fn = greet , inputs = "text" , outputs = "text" )

demo.launch()