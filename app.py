import gradio as gr
from PIL import Image
from src.deploy.xray_classification import XrayClassification

img_classifier = XrayClassification()

def load_and_display_image(image_file):
    image = Image.open(image_file.name)
    return image, img_classifier.process_file(image_file)

try:
    demo.close()
except:
    pass

with gr.Blocks(title="X-ray Image Defect Classification") as demo:
    gr.Markdown(
        """
        <div style="text-align: center; padding: 20px;">
            <h1>X-ray Image Defect Classification</h1>
            <p>Upload an X-ray image to start the classification. The image and the prediction {"Disease Detected", "No Finding"} will be displayed below.</p>
        </div>
        """
    )
    
    with gr.Column():
        prediction_output = gr.Textbox(
            label="Prediction", 
            placeholder="Your prediction will appear here", 
            show_copy_button=True, 
            autofocus=True,
            max_lines=2, 
            elem_id="prediction_output"
          )
        with gr.Row():
            uploaded_image = gr.File(label="Upload Image", file_count="single")
            displayed_image = gr.Image(label="Preview", elem_id="displayed_image", width=400, height=400)
        
    with gr.Row():
        submit_btn = gr.Button("Submit", variant="primary")
        clear_btn = gr.Button("Clear", variant="secondary")

    def submit_action(image_file):
        return load_and_display_image(image_file)

    def clear_action():
        return None, None, ""

    submit_btn.click(submit_action, inputs=uploaded_image, outputs=[displayed_image, prediction_output])
    clear_btn.click(clear_action, inputs=None, outputs=[uploaded_image, displayed_image, prediction_output])

demo.launch()
