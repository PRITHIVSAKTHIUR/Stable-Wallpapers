#!/usr/bin/env python
#patch 0.01 ()
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# ...

import json
import gradio as gr
import numpy as np
from PIL import Image
import spaces
import torch
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
import os
import uuid
import random

# Description for the Gradio interface
DESCRIPTIONx = """## INSTANT WALLPAPER 🌅 """

# CSS for styling the Gradio interface
css = '''
.gradio-container{max-width: 575px !important}
h1{text-align:center}
footer {
    visibility: hidden
}
'''

# Example prompts for the user to try
examples = [
    "Illustration of A starry night camp in the mountains. Low-angle view, Minimal background, Geometric shapes theme, Pottery, Split-complementary colors, Bicolored light, UHD",
    "Chocolate dripping from a donut against a yellow background, in the style of brocore, hyper-realistic oil --ar 2:3 --q 2 --s 750 --v 5  --ar 2:3 --q 2 --s 750 --v 5"
]

# Environment variables and defaults for configuration
MODEL_ID = os.getenv("MODEL_USED") #SG161222/RealVisXL_V4.0 / SG161222/Realistic_Vision_V5.1_noVAE / SG161222/RealVisXL_V4.0_Lightning  (1/3)
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "4096"))
USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE", "0") == "1"
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD", "0") == "1"
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))

# Setting the device to GPU if available, otherwise CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Loading the Stable Diffusion model
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    use_safetensors=True,
    add_watermarker=False,
).to(device)

# Configuring the scheduler for the model
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# Compiling the model for performance improvement if enabled
if USE_TORCH_COMPILE:
    pipe.compile()

# Enabling CPU offload to save GPU memory if enabled
if ENABLE_CPU_OFFLOAD:
    pipe.enable_model_cpu_offload()

# Maximum seed value for randomization
MAX_SEED = np.iinfo(np.int32).max

# Function to save the generated image
def save_image(img):
    unique_name = str(uuid.uuid4()) + ".png"
    img.save(unique_name)
    return unique_name

# Function to randomize the seed if needed
def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

# Defining the main generation function with GPU acceleration
@spaces.GPU(duration=60, enable_queue=True)
def generate(
    prompt: str,
    negative_prompt: str = "",
    use_negative_prompt: bool = False,
    seed: int = 1,
    width: int = 1024,
    height: int = 1024,
    guidance_scale: float = 3,
    num_inference_steps: int = 25,
    randomize_seed: bool = False,
    use_resolution_binning: bool = True, 
    num_images: int = 1,
    progress=gr.Progress(track_tqdm=True),
):
    # Randomizing the seed if required
    seed = int(randomize_seed_fn(seed, randomize_seed))
    generator = torch.Generator(device=device).manual_seed(seed)

    # Setting up the options for the image generation
    options = {
        "prompt": [prompt] * num_images,
        "negative_prompt": [negative_prompt] * num_images if use_negative_prompt else None,
        "width": width,
        "height": height,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "generator": generator,
        "output_type": "pil",
    }

    if use_resolution_binning:
        options["use_resolution_binning"] = True

    # Generating images in batches
    images = []
    for i in range(0, num_images, BATCH_SIZE):
        batch_options = options.copy()
        batch_options["prompt"] = options["prompt"][i:i+BATCH_SIZE]
        if "negative_prompt" in batch_options:
            batch_options["negative_prompt"] = options["negative_prompt"][i:i+BATCH_SIZE]
        images.extend(pipe(**batch_options).images)

    # Saving the generated images
    image_paths = [save_image(img) for img in images]
    return image_paths, seed

# Function to set the wallpaper size based on the selected option

def set_wallpaper_size(size):
    if size == "phone":
        return 1080, 1920
    elif size == "desktop":
        return 1920, 1080
    return 1024, 1024

# Function to load predefined images for display

def load_predefined_images():
    predefined_images = [
        "assets/1.png",
        "assets/2.png",
        "assets/3.png",
        "assets/4.png",
        "assets/5.png",
        "assets/6.png",
        "assets/7.png",
        "assets/8.png",
        "assets/9.png",
    ]
    return predefined_images

# Defining the Gradio interface with blocks
with gr.Blocks(css=css, theme="bethecloud/storj_theme") as demo:
    gr.Markdown(DESCRIPTIONx) 
    with gr.Group():
        with gr.Row():
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
                container=False,
            )
            run_button = gr.Button("Run", scale=0)
        result = gr.Gallery(label="Result", columns=1, show_label=False)

    with gr.Group():
        wallpaper_size = gr.Radio(
            choices=["phone", "desktop", "custom"],
            label="Wallpaper Size",
            value="desktop"
        )
        width = gr.Slider(
            label="Width",
            minimum=512,
            maximum=MAX_IMAGE_SIZE,
            step=64,
            value=1920,
            visible=False,
        )
        height = gr.Slider(
            label="Height",
            minimum=512,
            maximum=MAX_IMAGE_SIZE,
            step=64,
            value=1080,
            visible=False,
        )

        # Changing the wallpaper size based on user selection
        wallpaper_size.change(
            fn=set_wallpaper_size,
            inputs=wallpaper_size,
            outputs=[width, height],
            api_name="set_wallpaper_size"
        )

    # Advanced options for image generation
    with gr.Accordion("Advanced options", open=False, visible=False):
        num_images = gr.Slider(
            label="Number of Images",
            minimum=1,
            maximum=4,
            step=1,
            value=1,
        )
        with gr.Row():
            use_negative_prompt = gr.Checkbox(label="Use negative prompt", value=True)
            negative_prompt = gr.Text(
                label="Negative prompt",
                max_lines=5,
                lines=4,
                placeholder="Enter a negative prompt",
                value="(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation",
                visible=True,
            )
        seed = gr.Slider(
            label="Seed",
            minimum=0,
            maximum=MAX_SEED,
            step=1,
            value=0,
        )
        randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
        with gr.Row():
            guidance_scale = gr.Slider(
                label="Guidance Scale",
                minimum=0.1,
                maximum=6,
                step=0.1,
                value=3.0,
            )
            num_inference_steps = gr.Slider(
                label="Number of inference steps",
                minimum=1,
                maximum=25,
                step=1,
                value=20,
            )

    # Adding examples for the user to try
    gr.Examples(
        examples=examples,
        inputs=prompt,
        cache_examples=False
    )

    # Changing the visibility of the negative prompt based on user selection
    use_negative_prompt.change(
        fn=lambda x: gr.update(visible=x),
        inputs=use_negative_prompt,
        outputs=negative_prompt,
        api_name=False,
    )

    # Setting up the triggers and linking them to the generate function
    gr.on(
        triggers=[
            prompt.submit,
            negative_prompt.submit,
            run_button.click,
        ],
        fn=generate,
        inputs=[
            prompt,
            negative_prompt,
            use_negative_prompt,
            seed,
            width,
            height,
            guidance_scale,
            num_inference_steps,
            randomize_seed,
            num_images
        ],
        outputs=[result, seed],
        api_name="run",
    )

    # Adding a predefined gallery section
    gr.Markdown("### Generated Wallpapers")
    predefined_gallery = gr.Gallery(label="Generated Images", columns=3, show_label=False, value=load_predefined_images())

    # Adding a disclaimer
    gr.Markdown("**Disclaimer:**")
    gr.Markdown("This is the demo space for generating wallpapers using detailed prompts. This space works best for desktop-sized images (1920x1080). Reasonable quality images can be generated for mobile sizes (1080x1920), and custom images (1024x1024) can also be generated with better quality. Mobile settings may become disfigured. Try the sample prompts for generating higher quality images.<a href='https://huggingface.co/spaces/prithivMLmods/INSTANT-WALLPAPER/blob/main/sample_prompts.txt' target='_blank'>Try prompts</a>.")

    # Adding a note about user responsibility
    gr.Markdown("**Note:**")
    gr.Markdown("⚠️ users are accountable for the content they generate and are responsible for ensuring it meets appropriate ethical standards.")

# Launching the Gradio interface
if __name__ == "__main__":
    demo.queue(max_size=40).launch()