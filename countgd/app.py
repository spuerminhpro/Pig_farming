import spaces
import gradio as gr
import random
import torch
from PIL import Image
import numpy as np
import argparse
from util.slconfig import SLConfig, DictAction
from util.misc import nested_tensor_from_tensor_list
import datasets.transforms as T
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
# https://github.com/PhyscalX/gradio-image-prompter/tree/main/backend/gradio_image_prompter/templates/component
import io
import os
cwd = os.getcwd()
# Suppress warnings to avoid overflowing the log.
import warnings
warnings.filterwarnings("ignore")

from gradio_image_prompter import ImagePrompter

CONF_THRESH = 0.23

# MODEL:
def get_args_parser():
    """
    Example eval command:
    >> python main.py --output_dir ./gdino_test -c config/cfg_fsc147_vit_b_test.py --eval --datasets config/datasets_fsc147.json --pretrain_model_path ../checkpoints_and_logs/gdino_train/checkpoint_best_regular.pth --options text_encoder_type=checkpoints/bert-base-uncased --sam_tt_norm --crop
    """
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument(
        "--device", default="cuda", help="device to use for inference"
    )
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file.",
    )

    # dataset parameters
    parser.add_argument("--remove_difficult", action="store_true")
    parser.add_argument("--fix_size", action="store_true")

    # training parameters
    parser.add_argument("--note", default="", help="add some notes to the experiment")
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--pretrain_model_path",
        help="load from other checkpoint",
        default="checkpoint_best_regular.pth",
    )
    parser.add_argument("--finetune_ignore", type=str, nargs="+")
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_false")
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--find_unused_params", action="store_true")
    parser.add_argument("--save_results", action="store_true")
    parser.add_argument("--save_log", action="store_true")

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument(
        "--rank", default=0, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--local_rank", type=int, help="local rank for DistributedDataParallel"
    )
    parser.add_argument(
        "--local-rank", type=int, help="local rank for DistributedDataParallel"
    )
    parser.add_argument("--amp", action="store_true", help="Train with mixed precision")
    return parser

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

# Get counting model.
def build_model_and_transforms(args):
    normalize = T.Compose(
        [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    data_transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            normalize,
        ]
    )
    cfg = SLConfig.fromfile("cfg_app.py")
    cfg.merge_from_dict({"text_encoder_type": "checkpoints/bert-base-uncased"})
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k, v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    # fix the seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS

    assert args.modelname in MODULE_BUILD_FUNCS._module_dict

    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, _, _ = build_func(args)

    checkpoint = torch.load(args.pretrain_model_path, map_location="cpu")["model"]
    model.load_state_dict(checkpoint, strict=False)

    model.eval()

    return model, data_transform

# APP:
def get_box_inputs(prompts):
    box_inputs = []
    for prompt in prompts:
        if prompt[2] == 2.0 and prompt[5] == 3.0:
            box_inputs.append([prompt[0], prompt[1], prompt[3], prompt[4]])

    return box_inputs

def get_ind_to_filter(text, word_ids, keywords):
    if len(keywords) <= 0:
        return list(range(len(word_ids)))
    input_words = text.split()
    keywords = keywords.split(",")
    keywords = [keyword.strip() for keyword in keywords]

    word_inds = []
    for keyword in keywords:
        if keyword in input_words:
            if len(word_inds) <= 0:
                ind = input_words.index(keyword)
                word_inds.append(ind)
            else:
                ind = input_words.index(keyword, word_inds[-1])
                word_inds.append(ind)
        else:
            raise Exception("Only specify keywords in the input text!")

    inds_to_filter = []
    for ind in range(len(word_ids)):
        word_id = word_ids[ind]
        if word_id in word_inds:
            inds_to_filter.append(ind)

    return inds_to_filter

def get_xy_from_boxes(image, boxes):
    """
    Get box centers and return in image coordinates
    """
    (w, h) = image.size
    x = w * boxes[:, 0]
    y = h * boxes[:, 1]

    return x, y

def generate_heatmap(image, boxes):
    # Plot results.
    (w, h) = image.size
    det_map = np.zeros((h, w))
    x, y = get_xy_from_boxes(image, boxes)

    # Box centers are floating point, convert to int and clip them at edge of box
    x = np.clip(np.around(x).astype(int), 0, w - 1)
    y = np.clip(np.around(y).astype(int), 0, h - 1)

    det_map[y, x] = 1
    det_map = ndimage.gaussian_filter(
        det_map, sigma=(w // 200, w // 200), order=0
    )
    plt.imshow(image)
    plt.imshow(det_map[None, :].transpose(1, 2, 0), 'jet', interpolation='none', alpha=0.7)
    plt.axis('off')
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight')
    plt.close()

    output_img = Image.open(img_buf)
    return output_img
    
def generate_output_label(text, num_exemplars):
    out_label = "Detected instances predicted with"
    if len(text.strip()) > 0:
        out_label += " text"
        if num_exemplars == 1:
            out_label += " and " + str(num_exemplars) + " visual exemplar."
        elif num_exemplars > 1:
            out_label += " and " + str(num_exemplars) + " visual exemplars."
        else:
            out_label += "."
    elif num_exemplars > 0:
        if num_exemplars == 1:
            out_label += " " + str(num_exemplars) + " visual exemplar."
        else:
            out_label += " " + str(num_exemplars) + " visual exemplars."
    else:
        out_label = "Nothing specified to detect."
    
    return out_label

def preprocess(transform, image, input_prompts = None):
    if input_prompts == None:
        prompts = { "image": image, "points": []}
    else:
        prompts = input_prompts

    input_image, _ = transform(image, None)
    exemplar = get_box_inputs(prompts["points"])
    # Wrapping exemplar in a dictionary to apply only relevant transforms
    input_image_exemplar, exemplar = transform(prompts['image'], {"exemplars": torch.tensor(exemplar)})
    exemplar = exemplar["exemplars"]

    return input_image, input_image_exemplar, exemplar

def get_boxes_from_prediction(model_output, text, keywords = ""):
    logits = model_output["pred_logits"].sigmoid()[0][:, :]
    boxes = model_output["pred_boxes"][0]
    box_mask = logits.max(dim=-1).values > CONF_THRESH
    boxes = boxes[box_mask, :].cpu().numpy()
    logits = logits[box_mask, :].cpu().numpy()
    return boxes, logits

def predict(model, transform, image, text, prompts, device):
    keywords = "" # do not handle this for now
    input_image, input_image_exemplar, exemplar = preprocess(transform, image, prompts)

    input_images = input_image.unsqueeze(0).to(device)
    input_image_exemplars = input_image_exemplar.unsqueeze(0).to(device)
    exemplars = [exemplar.to(device)]
    
    with torch.no_grad():
        model_output = model(
                nested_tensor_from_tensor_list(input_images),
                nested_tensor_from_tensor_list(input_image_exemplars),
                exemplars,
                [torch.tensor([0]).to(device) for _ in range(len(input_images))],
                captions=[text + " ."] * len(input_images),
            )
        
    keywords = ""
    return get_boxes_from_prediction(model_output, text, keywords)

examples = [
    ["strawberry.jpg", "strawberry", {"image": "strawberry.jpg"}],
    ["strawberry.jpg", "blueberry", {"image": "strawberry.jpg"}],
    ["bird-1.JPG", "bird", {"image": "bird-2.JPG"}],
    ["fish.jpg", "fish", {"image": "fish.jpg"}],
    ["women.jpg", "girl", {"image": "women.jpg"}],
    ["women.jpg", "boy", {"image": "women.jpg"}],
    ["balloon.jpg", "hot air balloon", {"image": "balloon.jpg"}],
    ["deer.jpg", "deer", {"image": "deer.jpg"}],
    ["apple.jpg", "apple", {"image": "apple.jpg"}],
    ["egg.jpg", "egg", {"image": "egg.jpg"}],
    ["stamp.jpg", "stamp", {"image": "stamp.jpg"}],
    ["green-pea.jpg", "green pea", {"image": "green-pea.jpg"}],
    ["lego.jpg", "lego", {"image": "lego.jpg"}]
]


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Counting Application", parents=[get_args_parser()])
    args = parser.parse_args()
    device = get_device()
    model, transform = build_model_and_transforms(args)
    model = model.to(device)

    def _predict(image, text, prompts):
        return predict(model, transform, image, text, prompts, device)

    @spaces.GPU
    def count_main(image, text, prompts):
        if prompts is None:
            prompts = {"image": image, "points": []}
        boxes, _ = _predict(image, text, prompts)
        predicted_count = len(boxes)
        output_img = generate_heatmap(image, boxes)
        num_exemplars = len(get_box_inputs(prompts["points"]))
        out_label = generate_output_label(text, num_exemplars)

        return (gr.Image(output_img, visible=True, label=out_label, show_label=True), gr.Number(label="Predicted Count", visible=True, value=predicted_count))

    def remove_label(image):
        return gr.Image(show_label=False)

    with gr.Blocks(title="CountGD: Multi-Modal Open-World Counting", theme="soft", head="""<meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=1">""") as demo:
        gr.Markdown(
            """
            # <center>CountGD: Multi-Modal Open-World Counting
            <center><h3>Count objects with text, visual exemplars, or both together.</h3>
            <h3>Scroll down to try more examples</h3>
            <h3><a href='https://arxiv.org/abs/2407.04619' target='_blank' rel='noopener'>[paper]</a>
                <a href='https://github.com/niki-amini-naieni/CountGD/' target='_blank' rel='noopener'>[code]</a></h3>
            Limitation: this app does not support fine-grained counting based on attributes or visual grounding inputs yet. Note: if the exemplar and text conflict each other, both will be counted.</center>
            """
            )

        with gr.Row():
            with gr.Column():
                input_image_main = gr.Image(type='pil', label='Input Image', show_label='True', value="strawberry.jpg", interactive=True)
                input_text_main = gr.Textbox(label="What would you like to count?", placeholder="", value="strawberry")
                exemplar_image_main = ImagePrompter(type='pil', label='Visual Exemplar Image', show_label=True, value={"image": "strawberry.jpg", "points": []}, interactive=True)
            with gr.Column():
                detected_instances_main = gr.Image(label="Detected Instances", show_label='True', interactive=False)
                pred_count_main = gr.Number(label="Predicted Count")
                submit_btn_main = gr.Button("Count", variant="primary")
                clear_btn_main = gr.ClearButton(variant="secondary")
        submit_btn_main.click(fn=remove_label, inputs=[detected_instances_main], outputs=[detected_instances_main]).then(fn=count_main, inputs=[input_image_main, input_text_main, exemplar_image_main], outputs=[detected_instances_main, pred_count_main])
        clear_btn_main.add([input_image_main, input_text_main, exemplar_image_main, detected_instances_main, pred_count_main])


    demo.queue().launch(allowed_paths=['back-icon.jpg', 'paste-icon.jpg', 'upload-icon.jpg', 'button-legend.jpg'],share=True)