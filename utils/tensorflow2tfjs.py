import numpy as np
import os
import subprocess
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


en_drate_schedule = np.linspace(0.0, 1.0, 5)
de_drate_schedule = np.linspace(0.0, 1.0, 5)

onnx_path = "onnx_models"
tf_save_path = "tf_models"
tfjs_target_dir = "tfjs_models"

for en_drate in en_drate_schedule:
    for de_drate in de_drate_schedule:
        encoder_tf_path = os.path.join(tf_save_path, f"encoder-{en_drate}-{de_drate}-big")
        decoder_tf_path = os.path.join(tf_save_path, f"decoder-{en_drate}-{de_drate}-big")
        vae_tf_path = os.path.join(tf_save_path, f"vae_model-{en_drate}-{de_drate}-big")
        for saved_model_dir in [encoder_tf_path, decoder_tf_path, vae_tf_path]:
            tfjs_path = os.path.join(tfjs_target_dir, os.path.basename(saved_model_dir))
            if not os.path.exists(tfjs_path):
                os.makedirs(tfjs_path)
            command_tfjs = [
                "tensorflowjs_converter",
                "--input_format=tf_saved_model",
                saved_model_dir,
                tfjs_path
            ]
            subprocess.run(command_tfjs, check=True)