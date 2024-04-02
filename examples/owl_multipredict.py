
import argparse
import PIL.Image
import time
import torch
import os
from nanoowl.owl_predictor import (
    OwlPredictor
)
from nanoowl.owl_drawing import (
    draw_owl_output
)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="../assets/xindalu/person/")
    parser.add_argument("--prompt", type=str, default="[person]")
    parser.add_argument("--threshold", type=str, default="0.001")
    parser.add_argument("--output_path", type=str, default="../data/person/")
    parser.add_argument("--model", type=str, default="google/owlvit-base-patch32")
    parser.add_argument("--image_encoder_engine", type=str, default="../data/owl_image_encoder_patch32.engine")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--num_profiling_runs", type=int, default=30)
    args = parser.parse_args()

    prompt = args.prompt.strip("][()")
    text = prompt.split(',')
    print(text)

    thresholds = args.threshold.strip("][()")
    thresholds = thresholds.split(',')
    if len(thresholds) == 1:
        thresholds = float(thresholds[0])
    else:
        thresholds = [float(x) for x in thresholds]
    print(thresholds)
    
    images = os.listdir(args.image_path)

    for image_name in images:
        output_path = os.path.join(args.output_path, image_name)
        image_path = os.path.join(args.image_path, image_name)
        predictor = OwlPredictor(
            args.model,
            image_encoder_engine=args.image_encoder_engine
        )

        image = PIL.Image.open(image_path)
        
        text_encodings = predictor.encode_text(text)

        output = predictor.predict(
            image=image, 
            text=text, 
            text_encodings=text_encodings,
            threshold=thresholds,
            pad_square=False
        )

        if args.profile:
            torch.cuda.current_stream().synchronize()
            t0 = time.perf_counter_ns()
            for i in range(args.num_profiling_runs):
                output = predictor.predict(
                    image=image, 
                    text=text, 
                    text_encodings=text_encodings,
                    threshold=thresholds,
                    pad_square=False
                )
            torch.cuda.current_stream().synchronize()
            t1 = time.perf_counter_ns()
            dt = (t1 - t0) / 1e9
            print(f"PROFILING FPS: {args.num_profiling_runs/dt}")

        image = draw_owl_output(image, output, text=text, draw_text=True)

        image.save(output_path)