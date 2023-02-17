from transformers import BertForSequenceClassification, PreTrainedModel
import gradio as gr
import util
import torch
import numpy as np

model = BertForSequenceClassification.from_pretrained(
    './model/'
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def greet( text ):
    new_sentence = text

    # We need Token IDs and Attention Mask for inference on the new sentence
    test_ids = []
    test_attention_mask = []

    # Apply the tokenizer
    encoding = util.preprocessing(new_sentence, util.tokenizer)

    # Extract IDs and Attention Mask
    test_ids.append(encoding['input_ids'])
    test_attention_mask.append(encoding['attention_mask'])
    test_ids = torch.cat(test_ids, dim = 0)
    test_attention_mask = torch.cat(test_attention_mask, dim = 0)

    # Forward pass, calculate logit predictions
    with torch.no_grad():
      output = model(test_ids.to(device), token_type_ids = None, attention_mask = test_attention_mask.to(device))

    prediction = 'Hate' if np.argmax(output.logits.cpu().numpy()).flatten().item() == 1 else 'No hate'

    return prediction

demo = gr.Interface( fn = greet , inputs = "text" , outputs = "text" )

demo.launch()