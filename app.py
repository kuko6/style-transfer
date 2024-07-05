import gradio as gr
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from src.model import Model
import os

device = "cuda" if torch.cuda.is_available() else "cpu"


def denorm_img(img: torch.Tensor):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
    return torch.clip(img * std + mean, min=0, max=1)


def main(inp1, inp2, alph, out_size=256):
    # print("inp1 ", inp1)
    # print("inp2 ", inp2)

    model = Model()
    model.load_state_dict(torch.load("./models/model_puddle.pt", map_location=torch.device(device)))
    model.eval()

    model.alpha = alph
    style = TF.to_tensor(inp1["composite"])
    content = TF.to_tensor(inp2["composite"])

    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(out_size, antialias=True)
    ])

    style, content = norm(style), norm(content)
    style, content = transform(style), transform(content)

    style, content = style.unsqueeze(0).to(device), content.unsqueeze(0).to(device)
    out = model(content, style)

    return denorm_img(out[0].detach()).permute(1, 2, 0).numpy()

def update_crop_size(crop_size):
    return gr.update(crop_size=(crop_size, crop_size))

with gr.Blocks() as demo:
    gr.Markdown("# Style Transfer with AdaIN")
    with gr.Row(variant="compact", equal_height=False):
        inp1 = gr.ImageEditor(
            type="pil",
            sources=["upload", "clipboard"],
            crop_size=(256, 256),
            eraser=False,
            brush=False,
            layers=False,
            label="Style",
            image_mode="RGB",
            transforms="crop",
            canvas_size=(512, 512)
        )
        inp2 = gr.ImageEditor(
            type="pil",
            sources=["upload", "clipboard"],
            crop_size=(256, 256),
            eraser=False,
            brush=False,
            layers=False,
            label="Content",
            image_mode="RGB",
            transforms="crop",
            canvas_size=(512, 512)
        )
        out = gr.Image(type="pil", label="Output")
    
    with gr.Row():
        out_size = gr.Dropdown(
            choices=[256, 512],
            value=256,
            multiselect=False,
            interactive=True,
            allow_custom_value=True,
            label="Output size",
            info="Size of the output image"
        )
        out_size.change(fn=update_crop_size, inputs=out_size, outputs=inp1)
        out_size.change(fn=update_crop_size, inputs=out_size, outputs=inp2)

        alph = gr.Slider(0, 1, value=1, label="Alpha", info="How much to change the original image", interactive=True, scale=3)

    with gr.Row():
        with gr.Column():
            gr.Markdown("## Style Examples")
            gr.Examples(
                examples=[
                    os.path.join(os.path.dirname(__file__), "data/styles/25.jpg"),
                    os.path.join(os.path.dirname(__file__), "data/styles/2272.jpg"),
                    os.path.join(os.path.dirname(__file__), "data/styles/2314.jpg"),
                ],
                inputs=inp1,
            )
        with gr.Column():
            gr.Markdown("## Content Examples")
            gr.Examples(
                examples=[
                    # os.path.join(os.path.dirname(__file__), "data/content/bear.jpg"),
                    os.path.join(os.path.dirname(__file__), "data/content/cat.jpg"),
                    os.path.join(os.path.dirname(__file__), "data/content/cow.jpg"),
                    os.path.join(os.path.dirname(__file__), "data/content/ducks.jpg"),
                ],
                inputs=inp2,
            )
    
    btn = gr.Button("Run")
    btn.click(fn=main, inputs=[inp1, inp2, alph, out_size], outputs=out)

demo.launch()
