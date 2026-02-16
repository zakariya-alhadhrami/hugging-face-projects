import gradio as gr
import torch
from PIL import Image, ImageDraw, ImageFont

from transformers import AutoImageProcessor
from transformers import AutoModelForObjectDetection




model_save_path = "Zakariya007/rt_detrv2_finetuned_trashify_box_detector_v1"


image_processor = AutoImageProcessor.from_pretrained(model_save_path)
model = AutoModelForObjectDetection.from_pretrained(model_save_path)


device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)


id2label = model.config.id2label


color_dict = {
    "bin": "green",
    "trash": "blue",
    "hand": "purple",
    "trash_arm": "yellow",
    "not_trash": "red",
    "not_bin": "red",
    "not_hand": "red",
}



def predict_on_image(image, conf_threshold):
    image = image.convert("RGB")
    # Make sure model is in eval mode
    model.eval()

    # Make a prediction on target image
    with torch.no_grad():
        inputs = image_processor(images=[image], return_tensors="pt",do_resize=True,size={"height": 640, "width": 640})
        model_outputs = model(**inputs.to(device))

        target_sizes = torch.tensor([[image.size[1], image.size[0]]]) # -> [batch_size, height, width]

        # Post process the raw outputs from the model
        results = image_processor.post_process_object_detection(model_outputs,
                                                                threshold=conf_threshold,
                                                                target_sizes=target_sizes)[0]

    # Return all items in results to CPU (we'll want this for displaying outputs with matplotlib)
    for key, value in results.items():
        try:
            results[key] = value.item().cpu() # can't get scalar as .item() so add try/except block
        except:
            results[key] = value.cpu()



    # Can return results as plotted on a PIL image (then display the image)
    draw = ImageDraw.Draw(image)

    # Get a font from ImageFont
    font = ImageFont.load_default(size=20)

    # Get class names as text for print out
    detected_class_name_text_labels = []

    # Iterate through the predictions of the model and draw them on the target image
    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        # Create coordinates
        x, y, x2, y2 = tuple(box.tolist())

        # Get label_name
        label_name = id2label[label.item()]
        targ_color = color_dict[label_name]
        detected_class_name_text_labels.append(label_name)

        # Draw the rectangle
        draw.rectangle(xy=(x, y, x2, y2),
                       outline=targ_color,
                       width=3)

        # Create a text string to display
        text_string_to_show = f"{label_name} ({round(score.item(), 3)})"

        # Draw the text on the image
        draw.text(xy=(x, y),
                  text=text_string_to_show,
                  fill="white",
                  font=font)

    # Remove the draw each time
    del draw



    # Setup set of target items to discover
    target_items = {"trash", "bin", "hand"}
    detected_items = set(detected_class_name_text_labels)

    # If no items detected or trash, bin, hand not in detected items, return notification
    if not detected_items & target_items:
        return_string = (
            f"No trash, bin or hand detected at confidence threshold {conf_threshold}. "
            "Try another image or lowering the confidence threshold."
        )
        print(return_string)
        return image, return_string

    # If there are missing items, say what the missing items are
    missing_items = target_items - detected_items
    if missing_items:
        return_string = (
            f"Detected the following items: {sorted(detected_items & target_items)}. But missing the following in order to get +1: {sorted(missing_items)}. "
            "If this is an error, try another image or altering the confidence threshold. "
            "Otherwise, the model may need to be updated with better data."
        )
        print(return_string)
        return image, return_string

    # If all target items are present (the final remaining case)
    return_string = f"+1! Found the following items: {sorted(detected_items)}, thank you for cleaning up the area!"
    print(return_string)
    return image, return_string


description = """
Snap, Clean, Earn. Transform local cleanup into a game!

How it works:

 - Identify: Find litter near a bin.

 - Action: Take a photo of your hand placing trash in the bin.

 - Reward: Our AI verifies all three elements to award you +1 Point.

Make an impact, one snap at a time.

Model is a fine-tuned version of [RT-DETRv2](https://huggingface.co/docs/transformers/main/en/model_doc/rt_detr_v2#transformers.RTDetrV2Config)

"""

demo = gr.Interface(
    fn=predict_on_image,
    inputs=[
        gr.Image(type="pil", label="Target Image"),
        gr.Slider(minimum=0, maximum=1, value=0.3, label="Confidence Threshold")
    ],
    outputs=[
        gr.Image(type="pil", label="Image Output"),
        gr.Text(label="Text Output")
    ],
    title="🚮 Trashify Object Detection Demo",
    description=description,

    examples=[
        ["trashify_examples/trashify_example_1.jpeg", 0.3],
        ["trashify_examples/trashify_example_2.jpeg", 0.3],
        ["trashify_examples/trashify_example_3.jpeg", 0.3],
    ],
    cache_examples=True
)


demo.launch()

