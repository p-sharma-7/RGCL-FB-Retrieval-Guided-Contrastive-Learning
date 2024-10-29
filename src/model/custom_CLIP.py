import torch
import torch.nn as nn
import clip
from model.classifier import classifier_2L, classifier_3L
### This file is used for fine tuning CLIP end-to-end
# We cannot load our fined tuned weights directly to CLIP
# as our fine tuned model has CLIP.xxx rather than xxx 
# as the name of the layer

# The easiest way of loading the fine tuned weights of CLIP 
# is not to create this custom fine tuned CLIP model,
# to match the state dict's entry
class CLIP_fine_tuned(nn.Module):
    def __init__(self, CLIP) -> None:
        super(CLIP_fine_tuned, self).__init__()
        self.CLIP = CLIP
    def encode_image(self, images):
        return self.CLIP.encode_image(images)
    def encode_text(self, texts):
        return self.CLIP.encode_text(texts)
    
class CLIP_Classifier(nn.Module):
    def __init__(self, CLIP, test_img, test_text, classifier_choice="3L", device="cuda") -> None:
        super(CLIP_Classifier, self).__init__()
        self.CLIP = CLIP
        self.CLIP.eval()
        if device == "cpu":
            self.CLIP.float()
        else:
            #clip.model.convert_weights(self.CLIP)
            self.CLIP.float()
        # Do an initial run to get the length of embeddings to build the model
        with torch.no_grad():
            img_feats = self.CLIP.encode_image(test_img)
            text_feats = self.CLIP.encode_text(test_text)
            length = img_feats.shape[-1] + text_feats.shape[-1]
        print("Total feature length of CLIP: {}".format(length))
        if classifier_choice == "2L":
            self.classifier = classifier_2L(length)
        elif classifier_choice == "3L":
            self.classifier = classifier_3L(length)
        self.params = nn.ModuleDict({
            'classifier': nn.ModuleList([self.classifier])})
        
    def forward(self, images, texts):
        #print(images.shape, texts.shape)
        logits_per_image, logits_per_text = self.CLIP(images, texts)
        img_feats = self.CLIP.encode_image(images)
        text_feats = self.CLIP.encode_text(texts)
        output = self.classifier(torch.cat((img_feats, text_feats),1))
        return output, logits_per_image, logits_per_image

def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        if p.grad != None:
            p.grad.data = p.grad.data.float()   