import os
import argparse
import json
import numpy as np
import torch
from glob import glob
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, CLIPModel, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


# Clip score for frame consistency
def clip_score_frame(frames, prompt):
    inputs = processor(images=frames, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs).detach().cpu().numpy()
    cosine_sim_matrix = cosine_similarity(image_features)
    np.fill_diagonal(cosine_sim_matrix, 0)  # set diagonal elements to 0
    score = cosine_sim_matrix.sum() / (len(frames) * (len(frames) - 1))
    return score


# PickScore: https://github.com/yuvalkirstain/PickScore
def pick_score(frames, prompt):
    image_inputs = processor(images=frames, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
    text_inputs = processor(text=prompt, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
    with torch.no_grad():
        image_embs = model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
        score_per_image = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
        score_per_image = score_per_image.detach().cpu().numpy()
        score = score_per_image.mean()
    return score


eval_functions = {
    "clip_score_frame": clip_score_frame,
    "pick_score": pick_score,
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default="./OmniBench/frame/scenary/object", help="path to video frames folder")
    parser.add_argument("--json_path", type=str, default="./OmniBench/OmniBench-99-Object.json", help="path to JSON file with video order")
    parser.add_argument("--prompt_path", type=str, default="./OmniBench/prompt/object/scenary.txt", help="path to prompt file")
    parser.add_argument("--metric", type=str, default="pick_score", choices=['clip_score_frame', 'pick_score'])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device

    # Load JSON data to get the order of video names
    with open(args.json_path, "r") as json_file:
        data = json.load(json_file)
    video_names = [video_info["Video Name"] for video_info in data]

    # Load prompts from text file
    with open(args.prompt_path, "r") as prompt_file:
        prompts = [line.strip() for line in prompt_file.readlines()]

    # Initialize model and processor based on selected metric
    if args.metric == "clip_score_frame":
        preatrained_model_path = "openai/clip-vit-large-patch14"
        model = CLIPModel.from_pretrained(preatrained_model_path).to(device)
        processor = AutoProcessor.from_pretrained(preatrained_model_path)
    elif args.metric == "pick_score":
        preatrained_model_path = "pickapic-anonymous/PickScore_v1"
        model = AutoModel.from_pretrained(preatrained_model_path).to(device)
        processor_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        processor = AutoProcessor.from_pretrained(processor_path)
    else:
        raise NotImplementedError(args.metric)

    scores = []
    eval_function = eval_functions[args.metric]
    prompt_index = 0  # Index to track the current prompt for each video copy

    # Process each video name and its copies
    for video_name in video_names:
        # Find all matching video folders with "_copy_x" suffix for the current video name
        video_copies = sorted(glob(f"{args.video_path}/{video_name}_copy_*"))

        if not video_copies:
            print(f"No copies found for {video_name}")
            continue

        # Process each copy folder for the current video
        for video_path in video_copies:
            # Get the corresponding prompt
            if prompt_index >= len(prompts):
                raise ValueError("Not enough prompts for the number of video copies")

            prompt = prompts[prompt_index]
            prompt_index += 1  # Move to the next prompt for each video copy

            # Load frames from the current video copy folder
            frames = [Image.open(f) for f in sorted(glob(f"{video_path}/*.jpg"))]
            if not frames:
                print(f"No frames found in {video_path}")
                continue

            # Calculate score for the frames with the given prompt
            score = eval_function(frames, prompt)
            scores.append(score)
            print(f"Processed {video_path}, with {prompt} Score: {score:.3f}")

    # Print the overall mean score
    print("{}: {:.3f}".format(args.metric, np.mean(scores)))