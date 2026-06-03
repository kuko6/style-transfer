import gradio as gr
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from src.model import Model
import os
from functools import lru_cache
from pathlib import Path
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "model_puddle.pt"


def is_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as file:
            return file.read(48).startswith(b"version https://git-lfs.github.com/spec/v1")
    except FileNotFoundError:
        return False


def denorm_img(img: torch.Tensor):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
    return torch.clip(img * std + mean, min=0, max=1)


def editor_image_to_tensor(value):
    if value is None:
        raise gr.Error("Please provide both a style image and a content image.")

    image = (value.get("composite") or value.get("background")) if isinstance(value, dict) else value
    if image is None:
        raise gr.Error("Please provide both a style image and a content image.")

    if isinstance(image, dict):
        image = image.get("path") or image.get("url")

    if isinstance(image, (str, Path)):
        image_path = Path(image)
        if is_lfs_pointer(image_path):
            raise gr.Error("Selected image is a Git LFS pointer file. Run `git lfs pull`, then restart the app.")
        image = Image.open(image_path)

    return TF.to_tensor(image.convert("RGB") if hasattr(image, "convert") else image)


def available_examples(paths):
    return [path for path in paths if not is_lfs_pointer(Path(path))]


@lru_cache(maxsize=1)
def load_model():
    if is_lfs_pointer(MODEL_PATH):
        raise gr.Error(
            "Model weights are Git LFS pointer files. Install git-lfs and run `git lfs pull`, "
            "then restart the app."
        )

    model = Model(pretrained_encoder=False)
    state_dict = torch.load(MODEL_PATH, map_location=torch.device(device), weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def main(inp1, inp2, alph, out_size=256):
    model = load_model()
    model.alpha = alph
    style = editor_image_to_tensor(inp1)
    content = editor_image_to_tensor(inp2)

    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(int(out_size), antialias=True)
    ])

    style, content = norm(style), norm(content)
    style, content = transform(style), transform(content)

    with torch.no_grad():
        style, content = style.unsqueeze(0).to(device), content.unsqueeze(0).to(device)
        out = model(content, style)

    return TF.to_pil_image(denorm_img(out[0].detach().cpu()))

def update_crop_size(crop_size):
    size = int(crop_size)
    return gr.update(canvas_size=(size, size))

with gr.Blocks() as demo:
    gr.Markdown("# Style Transfer with AdaIN")
    with gr.Row(variant="compact", equal_height=False):
        inp1 = gr.ImageEditor(
            type="pil",
            sources=["upload", "clipboard"],
            eraser=False,
            brush=False,
            layers=False,
            label="Style",
            image_mode="RGB",
            transforms=["crop"],
            canvas_size=(512, 512)
        )
        inp2 = gr.ImageEditor(
            type="pil",
            sources=["upload", "clipboard"],
            eraser=False,
            brush=False,
            layers=False,
            label="Content",
            image_mode="RGB",
            transforms=["crop"],
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
                examples=available_examples([
                    os.path.join(os.path.dirname(__file__), "data/styles/25.jpg"),
                    os.path.join(os.path.dirname(__file__), "data/styles/2272.jpg"),
                    os.path.join(os.path.dirname(__file__), "data/styles/2314.jpg"),
                ]),
                inputs=inp1,
            )
        with gr.Column():
            gr.Markdown("## Content Examples")
            gr.Examples(
                examples=available_examples([
                    # os.path.join(os.path.dirname(__file__), "data/content/bear.jpg"),
                    os.path.join(os.path.dirname(__file__), "data/content/cat.jpg"),
                    os.path.join(os.path.dirname(__file__), "data/content/cow.jpg"),
                    os.path.join(os.path.dirname(__file__), "data/content/ducks.jpg"),
                ]),
                inputs=inp2,
            )
    btn = gr.Button("Run")
    btn.click(fn=main, inputs=[inp1, inp2, alph, out_size], outputs=out)

if __name__ == "__main__":
    demo.launch()
