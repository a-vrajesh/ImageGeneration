from flask import Flask, request, send_file, render_template
import io
from diffusers import AutoPipelineForText2Image
import torch
from PIL import Image
import matplotlib.pyplot as plt

# Initialize Flask app
app = Flask(__name__)

# Initialize the model
model_id1 = "dataautogpt3/OpenDalleV1.1"
pipe = AutoPipelineForText2Image.from_pretrained(model_id1, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

def generate_image_function(prompt):
    image = pipe(prompt=prompt).images[0]
    
    # Convert to PIL Image
    image = Image.fromarray(image)
    
    return image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generateimage', methods=['POST'])
def generate_image():
    #data = request.json
    prompt = request.form.get("prompt")
    
    if not prompt:
        return {"error": "No prompt provided"}, 400
    
    try:
        # Generate the image using the provided prompt
        image = generate_image_function(prompt)
        
        # Save the generated image to a BytesIO object
        img_io = io.BytesIO()
        image.save(img_io, 'JPEG', quality=70)
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/jpeg')
        
    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == '__main__':
    app.run(debug=True)