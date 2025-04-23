import gradio as gr
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load YOLO model
model = YOLO("firstModel.pt")

# Define the prediction function with error handling
def detect_species(files):
    annotated_images = []
    error_message = None

    try:
        # Ensure files are uploaded
        if not files:
            error_message = "No files were uploaded. Please upload one or more images."
            return None, error_message

        for file in files:  # Loop through the list of uploaded files
            try:
                # Process each file
                image = Image.open(file)
                image = np.array(image)
                results = model(image)
                annotated_image = results[0].plot()
                annotated_images.append(annotated_image)
            except Exception as e:
                # Handle individual image processing errors
                error_message = f"Error processing file {file.name}: {str(e)}"
                return None, error_message

        # If everything works, return the images and no error message
        return annotated_images, None

    except Exception as e:
        # For unexpected errors, set a custom error message but let Gradio handle the rest
        error_message = f"An unexpected error occurred: {str(e)}"
        return None, error_message  # Gradio will also display its pop-up


# Create the Gradio app
app = gr.Interface(
    fn=detect_species,
    inputs=gr.Files(file_types=["image"], label="Upload Images"),  # Allow multiple image uploads
    outputs=[
        gr.Gallery(label="Detection Results"),  # Annotated images
        gr.Textbox(label="Error Messages", interactive=False),  # Error box
    ],
    title="Plant Species Detector",
    description="Upload one or more leaf images, and the model will detect the species."
)

# Launch the app
app.launch(show_error=False)  # Keep Gradio's built-in error display enabled
