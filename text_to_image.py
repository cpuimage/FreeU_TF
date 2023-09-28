"""
Title: Generate an image from a text prompt using StableDiffusion
Author: fchollet
Date created: 2022/09/24
Last modified: 2022/09/24
Description: Use StableDiffusion to generate an image according to a short text
             description.
"""

from PIL import Image

from stable_diffusion.stable_diffusion import StableDiffusion

model = StableDiffusion(img_height=512, img_width=512, jit_compile=True, active_free_u=True, b1=1.2, b2=1.4, s1=0.9,
                        s2=0.2)
img = model.text_to_image(
    "Photograph of a beautiful horse running through a field", num_steps=25, seed=123)
Image.fromarray(img[0]).save("horse_freeu.jpg")
print("Saved at horse_freeu.jpg")
