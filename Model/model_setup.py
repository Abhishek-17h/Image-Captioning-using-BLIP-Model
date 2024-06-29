from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image

# Load the BLIP processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image_path):
    # Open image
    image = Image.open(image_path).convert("RGB")

    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt")

    # Generate caption
    with torch.no_grad():
        outputs = model.generate(**inputs)
        caption = processor.decode(outputs[0], skip_special_tokens=True)

    return caption
