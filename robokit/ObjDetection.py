import os
import torch
import logging
import warnings
import numpy as np
from PIL import Image as PILImg
from torchvision.ops import box_convert
from huggingface_hub import hf_hub_download
from groundingdino.models import build_model
import groundingdino.datasets.transforms as T
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.inference import predict
from groundingdino.util.utils import clean_state_dict
# from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

os.system("python setup.py build develop --user")
# os.system("pip install packaging==21.3")
warnings.filterwarnings("ignore")


class Logger:
    """
    This is a logger class
    """
    def __init__(self):
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    

class ObjectPredictor(Logger):
    """
    Root class for object predicton
    All other object prediction classes should inherit this
    """
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def bbox_to_scaled_xyxy(self, bboxes: torch.tensor, img_w, img_h):
        """
        Convert bounding boxes to scaled xyxy format.

        Parameters:
        - bboxes (torch.tensor): Input bounding boxes in cxcywh format.
        - img_w (int): Image width.
        - img_h (int): Image height.

        Returns:
        - torch.tensor: Converted bounding boxes in xyxy format.
        """
        try:
            bboxes = bboxes * torch.Tensor([img_w, img_h, img_w, img_h])
            bboxes_xyxy = box_convert(boxes=bboxes, in_fmt="cxcywh", out_fmt="xyxy")
            return bboxes_xyxy
        
        except Exception as e:
            self.logger.error(f"Error during bounding box conversion: {e}")
            raise e


class GroundingDINOObjectPredictor(ObjectPredictor):
    """
    This class implements Object detection using HuggingFace GroundingDINO
    Here instead of using generic language query, we fix the text prompt as "objects" which enables
    getting compact bounding boxes arounds generic objects.
    Hope is that these cropped bboxes when used with OpenAI CLIP yields good classification results.
    """
    def __init__(self, use_vitb=False, threshold=0.10):
        super().__init__()
        self.ckpt_repo_id = "ShilongLiu/GroundingDINO"
        if use_vitb:
            self.ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"  #
            self.config_file = "robokit/cfg/gdino/GroundingDINO_SwinB_cfg.py"
        else:
            self.ckpt_filenmae = "groundingdino_swint_ogc.pth"
            self.config_file = "robokit/cfg/gdino/GroundingDINO_SwinT_OGC.py"
        self.model = self.load_model_hf(
            self.config_file, self.ckpt_repo_id, self.ckpt_filenmae
        )
        self.threshold = threshold
    

    def load_model_hf(self, model_config_path, repo_id, filename):
        """
        Load model from Hugging Face hub.

        Parameters:
        - model_config_path (str): Path to model configuration file.
        - repo_id (str): ID of the repository.
        - filename (str): Name of the file.

        Returns:
        - torch.nn.Module: Loaded model.
        """
        try:
            args = SLConfig.fromfile(model_config_path) 
            model = build_model(args)
            args.device = self.device

            cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
            checkpoint = torch.load(cache_file, map_location=self.device)
            log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
            print("Model loaded from {} \n => {}".format(cache_file, log))
            _ = model.eval()
            return model    

        except Exception as e:
            # Log error and raise exception
            self.logger.error(f"Error loading model from Hugging Face hub: {e}")
            raise e

    def image_transform_grounding(self, image_pil):
        """
        Apply image transformation for grounding.

        Parameters:
        - image_pil (PIL.Image): Input image.

        Returns:
        - tuple: Tuple containing original PIL image and transformed tensor image.
        """
        try:
            transform = T.Compose([
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            image, _ = transform(image_pil, None) # 3, h, w
            return image_pil, image
        
        except Exception as e:
            self.logger.error(f"Error during image transformation for grounding: {e}")
            raise e

    def image_transform_for_vis(self, image_pil):
        """
        Apply image transformation for visualization.

        Parameters:
        - image_pil (PIL.Image): Input image.

        Returns:
        - torch.tensor: Transformed tensor image.
        """
        try:
            transform = T.Compose([
                T.RandomResize([800], max_size=1333),
            ])
            image, _ = transform(image_pil, None) # 3, h, w
            return image
        
        except Exception as e:
            self.logger.error(f"Error during image transformation for visualization: {e}")
            raise e
    
    def predict(self, image_pil: PILImg, det_text_prompt: str = "objects"):
        """
        Get predictions for a given image using GroundingDINO model.
        Paper: https://arxiv.org/abs/2303.05499
        Parameters:
        - image_pil (PIL.Image): PIL.Image representing the input image.
        - det_text_prompt (str): Text prompt for object detection
        Returns:
        - bboxes (list): List of normalized bounding boxeS in cxcywh
        - phrases (list): List of detected phrases.
        - configs (list): List of confidences.

        Raises:
        - Exception: If an error occurs during model prediction.
        """
        try:
            _, image_tensor = self.image_transform_grounding(image_pil)
            # default: box_threshold=0.25, text_threshold=0.25
            bboxes, conf, phrases = predict(self.model, image_tensor, det_text_prompt, box_threshold=self.threshold, text_threshold=self.threshold, device=self.device)
            return bboxes, phrases, conf        
        except Exception as e:
            self.logger.error(f"Error during model prediction: {e}")
            raise e


class SegmentAnythingPredictor(ObjectPredictor):
    """
    Predictor class for segmenting objects using the Segment Anything model.

    Inherits from ObjectPredictor.

    Attributes:
    - device (str): The device used for inference, either "cuda" or "cpu".
    - sam (torch.nn.Module): The Segment Anything model.
    - mask_generator (SamAutomaticMaskGenerator): The mask generator for the SAM model.
    - predictor (SamPredictor): The predictor for the SAM model.
    """

    def __init__(self, vit_model="vit_t"):
        """
        Initialize the SegmentAnythingPredictor object.
        """
        super().__init__()
        sam_weight_path = {
            "vit_t": "ckpts/mobilesam/vit_t.pth",
            "vit_b": "ckpts/sam_weights/sam_vit_b_01ec64.pth",
            "vit_h": "ckpts/sam_weights/sam_vit_h_4b8939.pth",
            "vit_l": "ckpts/sam_weights/sam_vit_l_0b3195.pth",
        }
        self.sam = sam_model_registry[vit_model](checkpoint=sam_weight_path[vit_model])
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)  # generate masks for entire image
        self.sam.to(device=self.device)
        self.sam.eval()
        self.predictor = SamPredictor(self.sam)

    def predict(self, image, prompt_bboxes):
        """
        Predict segmentation masks for the input image.

        Parameters:
        - image: The input image as a numpy array.
        - prompt_bboxes: Optional prompt bounding boxes as a list of lists of integers [x_min, y_min, x_max, y_max].

        Returns:
        - A tuple containing the input bounding boxes (if provided) and the segmentation masks as torch Tensors.

        Raises:
        - ValueError: If the input image is not a numpy array.
        """
        try:
            # Convert input image to numpy array
            image = np.array(image)

            # Check if prompt_bboxes is provided
            if prompt_bboxes is not None:
                # Convert prompt bounding boxes to torch tensor
                input_boxes = torch.tensor(prompt_bboxes, device=self.predictor.device)
                transformed_boxes = self.predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
                self.predictor.set_image(image)
                masks, _, _ = self.predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                )
            else:
                input_boxes = None
                masks = self.mask_generator.generate(image)
            
            return input_boxes, masks

        except ValueError as ve:
            print(f"ValueError: {ve}")
            return None, None


