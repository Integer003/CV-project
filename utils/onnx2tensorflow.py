import numpy as np
import os
import subprocess

en_drate_schedule = np.linspace(0.0, 1.0, 5)
de_drate_schedule = np.linspace(0.0, 1.0, 5)


for en_drate in en_drate_schedule:
    for de_drate in de_drate_schedule:
        # if en_drate != 0.0 or de_drate != 0.0:
        #     break
        onnx_path = "onnx_models"
        encoder_onnx_path = os.path.join(onnx_path, f"encoder-{en_drate}-{de_drate}-big.onnx")
        print(f"---A4---Encoder onnx path: {encoder_onnx_path}")
        decoder_onnx_path = os.path.join(onnx_path, f"decoder-{en_drate}-{de_drate}-big.onnx")
        vae_onnx_path = os.path.join(onnx_path, f"vae_model-{en_drate}-{de_drate}-big.onnx")
            
        tf_save_path = "tf_models"
        encoder_tf_path = os.path.join(tf_save_path, f"encoder-{en_drate}-{de_drate}-big")
        decoder_tf_path = os.path.join(tf_save_path, f"decoder-{en_drate}-{de_drate}-big")
        vae_tf_path = os.path.join(tf_save_path, f"vae_model-{en_drate}-{de_drate}-big")
        
        for onnx_file, output_dir in [(encoder_onnx_path, encoder_tf_path),
                                      (decoder_onnx_path, decoder_tf_path),
                                      (vae_onnx_path, vae_tf_path)]:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            command = [
                "onnx2tf",
                "-i", onnx_file,
                "-o", output_dir,
                "-b", "1",
                "--output_signaturedefs"  # Optional: output standard SavedModel
            ]
            subprocess.run(command, check=True)

        
