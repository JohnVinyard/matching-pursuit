from util.load_trained_weights import load_trained_weights_for_inference
from .experiment import Model as ResonanceInferenceModel
from config.dotenv import Config

model = load_trained_weights_for_inference(
    __file__, 
    ResonanceInferenceModel, 
    device='cpu', 
    s3_bucket=Config.s3_bucket())


