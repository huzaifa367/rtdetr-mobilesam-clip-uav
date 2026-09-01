
"""
Name: Adaptive and Multi Attention Detecor using a data-dependent gating mechanism

Learnable Gate Projection (self.gate_proj):
The Linear layer in self.gate_proj maps the concatenated mean embeddings from attn_output1 and attn_output2 into the same embedding dimension (embed_dim).
The weights and biases of this linear layer are learned during training to optimize the gating mechanism.

Dynamic Gating (gate):
The gating mechanism (gate) is computed based on the outputs of the attention mechanisms. It varies dynamically for every input batch.
The Sigmoid activation ensures the gating values are in the range [0, 1], allowing for a soft balance between the two attention outputs.

Weighted Combination:
The gating values gate are used to weigh the contributions of attn_output1 and attn_output2.
Since the gate values are computed from the inputs, the combination is dynamic and adjusts during training.
"""
# from mobile_sam import sam_model_registry#, SamAutomaticMaskGenerator, SamPredictor

import torch
import torch.nn as nn
import os
import urllib.request
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DetrModel
from transformers import CLIPProcessor, CLIPModel
import os
import urllib.request
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

import os
import urllib
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel, AutoImageProcessor, AutoModelForDepthEstimation
# from segment_anything import SamPredictor,sam_model_registry
from ultralytics.models.sam import Predictor as SAMPredictor

class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


class FusionGating(nn.Module):
    def __init__(self, 
                 num_queries, num_classes, num_heads=8,neurons=128, 
                 models=None):
        super(FusionGating, self).__init__()

        if models is None:
            raise ValueError('Empty models set found.')

        self.models = models
        embed_dim = 256

        # 🔹 Load Pretrained FusionGate of DETR Model
        # self.detr_FusionGate = DetrModel.from_pretrained("facebook/detr-resnet-50").FusionGate

        # Initialize the layers for each model type
        self.model_layers = nn.ModuleDict()

        # 🔹 For SAM model (if included)
        if 'sam' in models:
            self.model_layers['sam'] = nn.Sequential(
                # nn.Conv2d(256, embed_dim, kernel_size=3, padding=1),  # Output: (batch, embed_dim, 64, 64)
                nn.Flatten(start_dim=2),  # Output: (batch, embed_dim, 64*64)
                Transpose(1, 2),
                nn.Linear(256, embed_dim),  # Fix: Match expected input for Linear layer
                nn.Linear(embed_dim, 128),
            )

        # 🔹 For Depth model (if included)
        if 'depth' in models:
            self.model_layers['depth'] = nn.Sequential(
                nn.Linear(384, embed_dim),
                nn.LayerNorm((1370, embed_dim)),
                nn.Linear(embed_dim, 128),
            )

        # 🔹 For CLIP model (if included)
        if 'clip' in models:
            self.model_layers['clip'] = nn.Sequential(
                nn.Linear(768, embed_dim),
                nn.LayerNorm((50, embed_dim)),
                nn.Linear(embed_dim, 128),
            )

        # 🔹 Adaptive attention layers (for each model)
        self.attn_layers = nn.ModuleDict()
        for model in models:
            self.attn_layers[model] = nn.MultiheadAttention(embed_dim=128, num_heads=num_heads, batch_first=True, dropout=0.1)

    # 🔹 Gating mechanism for adaptive fusion (if 3 models are included)
        # self.gate_proj1 = nn.Linear(128, 160)
        # self.gate_proj2 = nn.Linear(160, 160)
        self.gate_layer = nn.SiLU(inplace=True)

        # 🔹 Query embeddings (Learnable)
        # self.query_embed = nn.Embedding(num_queries, 256)
        # self.query_attn = nn.MultiheadAttention(embed_dim=256, num_heads=num_heads, batch_first=True, dropout=0.1)

        """        
        Dropout before classification & regression prevents overfitting.
        Low dropout (0.1) in attention layers stabilizes training without hurting performance.
        No dropout inside feature extractors (SAM, CLIP, Depth) to avoid losing valuable information.
        Dropout in gating mechanism (0.1).
        """

        self.dropout_gate = nn.Dropout(p=0.1) 
        self.gate_conv = None
        self.gate_proj1 = None


    def forward(self,  sam_features=None, clip_features=None, depth_features=None):

        # 🔹 Dynamic Processing based on models used
        processed_inputs = {}
        for model_name in self.models:
            # Process each input through its respective model
            if model_name == 'sam':
                x = self.model_layers['sam'](sam_features)
                # size = x.shape

            elif model_name == 'clip':
                x = self.model_layers['clip'](clip_features)
                # size = x.shape
            
            elif model_name == 'depth':
                x = self.model_layers['depth'](depth_features)
                # size = x.shape

            processed_inputs[model_name] = x

        # 🔹 Apply attention for each model
        attn_outputs = {}
        if len(self.models) == 1:
            # Use the same model for all attention inputs if only one model is used
            model_name = self.models[0]
            attn_output, _ = self.attn_layers[model_name](processed_inputs[model_name], processed_inputs[model_name], processed_inputs[model_name])
            attn_outputs[model_name] = attn_output

        elif len(self.models) == 2:
            # For two models, use the second model twice
            model_1, model_2 = self.models
            attn_output_1, _ = self.attn_layers[model_1](processed_inputs[model_1], processed_inputs[model_2], processed_inputs[model_2])
            attn_output_2, _ = self.attn_layers[model_2](processed_inputs[model_2], processed_inputs[model_1], processed_inputs[model_1])
            attn_outputs[model_1] = attn_output_1
            attn_outputs[model_2] = attn_output_2
            
        elif len(self.models) == 3:
            model_1, model_2, model_3 = self.models

            attn_output_1, _ = self.attn_layers[model_1](
                processed_inputs[model_1],  # Query from Model 1
                processed_inputs[model_2],  # Key from Model 2
                processed_inputs[model_2]   # Value from Model 2
            )

            attn_output_2, _ = self.attn_layers[model_2](
                processed_inputs[model_2],  # Query from Model 2
                processed_inputs[model_3],  # Key from Model 3
                processed_inputs[model_3]   # Value from Model 3
            )

            attn_output_3, _ = self.attn_layers[model_3](
                processed_inputs[model_3],  # Query from Model 3
                processed_inputs[model_1],  # Key from Model 1
                processed_inputs[model_1]   # Value from Model 1
            )

            attn_outputs[model_1] = attn_output_1
            attn_outputs[model_2] = attn_output_2
            attn_outputs[model_3] = attn_output_3


        # 🔹 Adaptive fusion using gating mechanism
        combined = torch.cat(list(attn_outputs.values()), dim=1).transpose(2,1)
        # combined = self.gate_proj2(combined)
        B,C,S = combined.shape
        if self.gate_conv is None:
            self.gate_proj1 = nn.Linear(S, 256)
            self.gate_conv = nn.Sequential(
            # Step 1: Increase channels from 1 -> 16
            nn.Conv2d(C, 48, kernel_size=5, stride=1, padding=1),
            nn.Upsample(size=(160, 160), mode="bilinear", align_corners=False)
            )
        combined = self.gate_proj1(combined)
        gate = self.gate_conv(combined.unsqueeze(2).expand(-1, -1, 256, 256))

        gate = self.dropout_gate(gate)
        gate = self.gate_layer(gate)
        return gate


class CombinedModel(nn.Module):
    def __init__(self,
                 FusionGate_neurons=128, num_heads=16, num_queries=50, num_classes=10,
                 features=['sam', 'clip', 'depth'], load_FusionGate=True, load_encoders=True,
                 freeze_encoders = False):
        super(CombinedModel, self).__init__()

        self.use_sam = 'sam' in features
        self.use_depth = 'depth' in features
        self.use_clip = 'clip' in features
        self.use_FusionGate = load_FusionGate
        self.load_encoders = load_encoders
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.f = -1
        self.i=0
        
        # Load SAM encoder if enabled
        self.sam_encoder = None
        if self.use_sam and load_encoders:
            self.load_sam_encoder()
            # models.append('sam')

        # Load Depth Anything model if enabled
        self.depth_processor = None
        self.depth_anything_model = None
        if self.use_depth and load_encoders:
            self.load_depth_model("LiheYoung/depth-anything-small-hf")
            # models.append('depth')

        # Load CLIP model if enabled
        self.clip_model = None
        self.clip_processor = None
        if self.use_clip and load_encoders:
            self.load_clip_model()
            # models.append('clip')

        # Define the adaptive attention FusionGate if enabled
        self.FusionGate = None
        if self.use_FusionGate:
            self.FusionGate = FusionGating(
                neurons=FusionGate_neurons,
                num_heads=num_heads,
                num_queries=num_queries,
                num_classes=num_classes,
                models=features
            )
        if freeze_encoders:
            for param in self.clip_model.parameters():
                param.requires_grad = True
            
            for param in self.sam_predictor.model.parameters():
                param.requires_grad = True
    
    def load_clip_model(self):
        """Load the CLIP model and processor"""
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").vision_model
        # self.clip_model.eval()

    # def load_sam_encoder(self):
    #     """Loads the SAM encoder and returns it."""
    #     sam_model_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    #     model_path = os.path.join("models", "sam_vit_b.pth")

    #     if not os.path.exists(model_path):
    #         os.makedirs('models', exist_ok=True)
    #         urllib.request.urlretrieve(sam_model_url, model_path)

    #     # self.sam_encoder = sam_model_registry["vit_b"](checkpoint=model_path).image_encoder
    #     self.sam_encoder = sam_model_registry["vit_b"](checkpoint=model_path)
    #     self.sam_predictor = SamPredictor(self.sam_encoder)
        # self.sam_encoder.eval()  # Set to evaluation mode


    def load_sam_encoder(self):
        self.sam_predictor = SAMPredictor(overrides=dict(conf=0.25, task="segment", mode="predict", imgsz=640, model="mobile_sam.pt"))
        if self.sam_predictor.model is None:
            self.sam_predictor.setup_model(model=None)
            

    def load_depth_model(self, depth_model_name):
        """Loads the Depth Anything model and returns its processor and backbone."""
        self.depth_processor = AutoImageProcessor.from_pretrained(depth_model_name, use_fast=True)
        self.depth_anything_model = AutoModelForDepthEstimation.from_pretrained(depth_model_name).backbone
        # self.depth_anything_model.eval()  # Set to evaluation mode

    # def forward_sam(self, image):
    #     self.sam_predictor.set_image(image)
    #     return self.sam_predictor.features

    def forward_sam(self, images):
        self.sam_predictor.setup_source(images)
        self.sam_predictor.model.set_imgsz(self.sam_predictor.imgsz)

        imgs = []
        
        if len(self.sam_predictor.dataset) == 1:
            for img in self.sam_predictor.dataset:
                imgs.append(self.sam_predictor.preprocess(img[1]).squeeze(0))
        else:
            for batch in self.sam_predictor.dataset:
                for img in batch[1]:
                    imgs.append(self.sam_predictor.preprocess(img).squeeze(0))

        imgs = torch.stack(imgs,dim=0)
        if len(imgs.shape)==3:
            imgs = imgs.unsqueeze(0)

        return self.sam_predictor.model.image_encoder(imgs)

        # self.sam_predictor.set_image(image)
        # return self.sam_predictor.features

    def forward_clip(self, image):
        """Pass an image through the CLIP encoder and extract features."""
        if self.clip_model is None or self.clip_processor is None:
            raise ValueError("CLIP model is not enabled.")
        
        # Preprocess image for CLIP
        clip_inputs = self.clip_processor(images=image, return_tensors="pt", padding=True,
                                          do_rescale=False)
        
        # Forward pass through CLIP
        outputs = self.clip_model(**clip_inputs)

        return outputs.last_hidden_state  # This returns the image features (embeddings)
    
    def forward_depth(self, image):
        """Pass an image through the Depth Anything encoder only."""
        if self.depth_anything_model is None or self.depth_processor is None:
            raise ValueError("Depth Anything model is not enabled.")
        
        depth_inputs = self.depth_processor(images=image, return_tensors="pt", do_rescale=False).to(next(self.parameters()).device)
        return self.depth_anything_model(**depth_inputs).feature_maps[0]
    
    def forward_FusionGate(self, sam_features=None, clip_features=None, depth_features=None):
        return self.FusionGate(sam_features=sam_features, 
                            clip_features=clip_features, 
                            depth_features=depth_features)
    
    def forward(self, image=None, sam_features=None, clip_features=None, depth_features=None):

        # Initialize feature maps if not provided
        if sam_features is None and self.use_sam and self.load_encoders:
            sam_features = self.forward_sam(image)  # Output: [1, 256, 64, 64]

        if depth_features is None and self.use_depth and self.load_encoders:
            depth_features = self.forward_depth(image)  # Output: [1, 1370, 384]

        if clip_features is None and self.use_clip and self.load_encoders:
            clip_features = self.forward_clip(image)  # Output: [1, 512]
        
        # if not image==None:
        #     if self.use_sam:
        #         sam_features = sam_features / torch.max(sam_features)  # Normalize
        #     if self.use_depth:
        #         depth_features = depth_features / torch.max(depth_features)  # Normalize
        #     if self.use_clip:
        #         clip_features = clip_features / torch.max(clip_features)  # Normalize

        # If FusionGate is enabled, pass through it
        if self.use_FusionGate:
            return self.forward_FusionGate(sam_features=sam_features, 
                            clip_features=clip_features, 
                            depth_features=depth_features)
        
        # If FusionGate is disabled, return only extracted features
        return {
            "sam_features": sam_features,
            "depth_features": depth_features,
            "clip_features": clip_features
        }


    