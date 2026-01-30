import torch
import gradio as gr
from transformers import pipeline

from typing import Dict

def food_not_food_classifier(text: str) -> Dict[str, float]:
  
  food_not_food_classifier_pipeline = pipeline(task="text-classification",
                                               model='Zakariya007/hf_food_not_food_distilbert_base_uncased',
                                                batch_size=32,
                                              device="cuda" if torch.cuda.is_available() else "cpu",
                                              top_k=None) # top_k=None => return all possible labels

 
  outputs = food_not_food_classifier_pipeline(text)[0]

 
  output_dict = {}
  for item in outputs:
    output_dict [item["label"] ] = item["score"]

  return output_dict


description = """
This demo uses a fine-tuned Natural Language Processing (NLP) model to distinguish between descriptions of actual food and non-food related text. 
While many classifiers struggle with linguistic nuances, this model is designed to recognize the difference between a recipe and a metaphorical phrase (like "piece of cake").
"""

demo = gr.Interface(fn=food_not_food_classifier,
              inputs="text",
              outputs=gr.Label(num_top_classes=2), # show top 2 classes (that's all we have)
              title=" 🍓 Is it Food?",
              description=description,
              examples=
               [
                   ["I whipped up a fresh batch of code, but it seems to have a syntax error."],
              ["A delicious photo of a plate of scrambled eggs, bacon and toast. "],
                        ['A delicious plate of spaghetti with meatballs']
              ])


if __name__ == "__main__":
    demo.launch()
