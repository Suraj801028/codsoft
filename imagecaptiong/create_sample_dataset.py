import os
from PIL import Image, ImageDraw, ImageFont
import random

# Create data directory
data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(data_dir, exist_ok=True)

# Sample captions paired with simple visual indicators
samples = [
    ('dog_running.jpg', 'A dog running in the park.', (100, 200, 100)),
    ('red_car.jpg', 'A red car parked on the street.', (255, 100, 100)),
    ('blue_house.jpg', 'A blue house with a garden.', (100, 100, 255)),
    ('cat_sleeping.jpg', 'A cat sleeping on a comfortable bed.', (200, 150, 100)),
    ('yellow_flower.jpg', 'A yellow flower blooming in spring.', (255, 255, 100)),
    ('person_cycling.jpg', 'A person riding a bicycle on a path.', (150, 100, 200)),
    ('green_tree.jpg', 'A tall green tree in the forest.', (50, 200, 50)),
    ('beach_sunset.jpg', 'A beautiful sunset over the beach.', (255, 150, 50)),
    ('mountain_snow.jpg', 'Snowy mountains under a clear sky.', (200, 200, 255)),
    ('bird_flying.jpg', 'A bird flying high in the blue sky.', (150, 200, 255)),
]

captions_list = []

for filename, caption, color in samples:
    # Create a simple colored image
    img = Image.new('RGB', (224, 224), color=color)
    draw = ImageDraw.Draw(img)
    
    # Add text to the image for visual distinction
    text = caption.split()[0:3]  # First few words
    text_str = '\n'.join(text)
    try:
        draw.text((10, 10), text_str, fill=(0, 0, 0))
    except:
        pass  # Font not available, skip text
    
    # Save image
    img_path = os.path.join(data_dir, filename)
    img.save(img_path)
    print(f'Created {filename}')
    
    captions_list.append(f'{filename}|{caption}')

# Write captions.txt
captions_path = os.path.join(data_dir, 'captions.txt')
with open(captions_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(captions_list))

print(f'\nCreated {len(captions_list)} images and captions.txt')
print(f'Dataset ready at: {data_dir}')
