from __future__ import division
import warnings
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import cv2
from Networks.models import base_patch16_384_token, base_patch16_384_gap

warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description='TransCrowd Inference')
    parser.add_argument('--model_path', type=str, default='./save_file/Pig_detection/model_best.pth',
                        help='path to the model')
    parser.add_argument('--input', type=str, default='',
                        help='path to input image or directory')
    parser.add_argument('--output', type=str, default='./results',
                        help='path to save results')
    parser.add_argument('--model_type', type=str, default='token',
                        help='model type: token or gap')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='gpu id')
    args = parser.parse_args()
    return args

def prepare_img(img_path):
    """Prepare image for inference"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(img_path).convert('RGB')
    orig_img = np.array(img)
    
    # Handle images larger than 384x384 by splitting them
    width, height = img.size
    m = int(width / 384)
    n = int(height / 384)
    
    if m == 0 or n == 0:  # Image smaller than 384x384
        img_tensor = transform(img).unsqueeze(0)
        return img_tensor, orig_img
    
    img_crops = []
    for j in range(0, n):
        for i in range(0, m):
            crop = img.crop((i * 384, j * 384, (i + 1) * 384, (j + 1) * 384))
            crop_tensor = transform(crop).unsqueeze(0)
            img_crops.append(crop_tensor)
    
    img_tensor = torch.cat(img_crops, 0)
    return img_tensor, orig_img

def inference(model, img_tensor, device):
    """Run model inference on the input tensor"""
    model.eval()
    with torch.no_grad():
        if len(img_tensor.shape) == 3:  # Single image
            img_tensor = img_tensor.unsqueeze(0)
        
        img_tensor = img_tensor.to(device)
        out = model(img_tensor)
        
        if img_tensor.shape[0] > 1:  # Multiple crops
            count = torch.sum(out).item()
        else:
            count = out.item()
            
    return count

def visualize_result(img, count, output_path):
    """Add count to the image and save result"""
    # Create a copy of the image
    result_img = img.copy()
    
    # Add text with count
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"Pig Count: {int(count)}"
    cv2.putText(result_img, text, (10, 30), font, 1, (0, 0, 255), 3, cv2.LINE_AA)
    
    # Save the result
    cv2.imwrite(output_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
    
    return result_img

def main():
    args = parse_args()
    
    # Set device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    # Load model
    if args.model_type == 'token':
        model = base_patch16_384_token(pretrained=False)
    elif args.model_type == 'gap':
        model = base_patch16_384_gap(pretrained=False)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    model = nn.DataParallel(model)
    
    # Load checkpoint
    if os.path.isfile(args.model_path):
        print(f"Loading checkpoint '{args.model_path}'")
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise FileNotFoundError(f"No checkpoint found at '{args.model_path}'")
    
    model = model.to(device)
    
    # Process input
    if os.path.isfile(args.input):
        # Single image
        img_tensor, orig_img = prepare_img(args.input)
        img_tensor = img_tensor.to(device)
        
        # Run inference
        count = inference(model, img_tensor, device)
        
        # Visualize and save result
        output_path = os.path.join(args.output, os.path.basename(args.input))
        visualize_result(orig_img, count, output_path)
        
        print(f"Image: {args.input}, Predicted count: {count:.2f}")
        print(f"Result saved to {output_path}")
    
    elif os.path.isdir(args.input):
        # Directory of images
        image_files = [f for f in os.listdir(args.input) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            print(f"No images found in {args.input}")
            return
        
        results = []
        
        for image_file in image_files:
            img_path = os.path.join(args.input, image_file)
            img_tensor, orig_img = prepare_img(img_path)
            
            # Run inference
            count = inference(model, img_tensor, device)
            
            # Visualize and save result
            output_path = os.path.join(args.output, image_file)
            visualize_result(orig_img, count, output_path)
            
            print(f"Image: {image_file}, Predicted count: {count:.2f}")
            results.append((image_file, count))
        
        # Save summary of results
        with open(os.path.join(args.output, 'results.txt'), 'w') as f:
            f.write("Image,Count\n")
            for image_file, count in results:
                f.write(f"{image_file},{count:.2f}\n")
        
        print(f"Results saved to {args.output}")
    
    else:
        print(f"Input path {args.input} does not exist")

if __name__ == '__main__':
    main() 