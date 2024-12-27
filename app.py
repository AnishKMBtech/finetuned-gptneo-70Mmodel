import gradio as gr
from transformers import pipeline
import os

# Initialize pipelines with specified models
image_caption_pipe = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
text_generation_pipe = pipeline("text-generation", model="tiiuae/Falcon3-1B-Instruct")

# Define the inbuilt custom prompt
inbuilt_custom_prompt = "can you explain what this sentence says and make it brief :"

def multi_agent_inference(image):
    # Generate caption from image
    caption = image_caption_pipe(image)[0]['generated_text']
    yield (caption, "Processing text model...")  # Stream the caption first and show processing status

    # Concatenate inbuilt custom prompt with caption
    input_text = f"{caption} {inbuilt_custom_prompt}"

    # Generate enhanced description from caption
    enhanced_description = text_generation_pipe(input_text, max_length=100, num_return_sequences=1)[0]['generated_text']
    
    # Remove the inbuilt custom prompt from the enhanced description
    enhanced_description = enhanced_description.replace(inbuilt_custom_prompt, "").strip()
    
    yield (caption, enhanced_description)

def end_program():
    print("Ending the program...")
    os._exit(0)

if __name__ == '__main__':
    with gr.Blocks(css="custom.css") as iface:
        gr.Markdown("<h1 style='text-align: center;'>Hybrid Model Combination of Language and Feature Extractor</h1>")
        gr.Markdown("<p style='text-align: center;'>Upload an image, and the model will generate a description and enhance it with more details.</p>")
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Upload Image", interactive=True)
                generate_btn = gr.Button("Generate")
                end_btn = gr.Button("End Program")
            with gr.Column():
                caption_out = gr.Textbox(label="Image Model Output (vit-gpt2-image-captioning)", lines=5)
                text_out = gr.Textbox(label="Text Model Output (Falcon3-1B-Instruct)", lines=5)
        
        generate_btn.click(fn=multi_agent_inference, inputs=input_image, outputs=[caption_out, text_out])
        end_btn.click(fn=end_program, inputs=None, outputs=None)

    iface.launch(share=True, show_api=False)
