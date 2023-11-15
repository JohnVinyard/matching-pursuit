from util.load_trained_weights import load_trained_weights_for_inference
from .experiment import Model as ResonanceInferenceModel
from config.dotenv import Config

current_path = __file__

model = load_trained_weights_for_inference(
    current_path, 
    ResonanceInferenceModel, 
    device='cpu', 
    s3_bucket=Config.s3_bucket())


