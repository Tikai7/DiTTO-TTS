import subprocess
import os
from tqdm import tqdm

class Processing:

    @staticmethod
    def remove_metadata_from_audio_folder(input_folder, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for root, _, files in os.walk(input_folder):
            for file in tqdm(files):
                if file.endswith(".opus"):
                    input_file = os.path.join(root, file)

                    relative_path = os.path.relpath(root, input_folder)
                    output_dir = os.path.join(output_folder, relative_path)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    output_file = os.path.join(output_dir, file)
                    if os.path.exists(output_file):
                        print(f"Skipping {output_file}, already processed.")
                        continue  

                    command = [
                        "ffmpeg", "-i", input_file,
                        "-map_metadata", "-1", "-c", "copy",
                        output_file
                    ]

                    try:
                        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    except subprocess.CalledProcessError as e:
                        print(f"Error processing {input_file}: {e}")


if __name__ == "__main__":
    from Config import Config

    Processing.remove_metadata_from_audio_folder(Config.TRAIN_PATH+"/"+"audio", Config.TRAIN_PATH+"/"+"audio_clean",)
    Processing.remove_metadata_from_audio_folder(Config.TEST_PATH+"/"+"audio", Config.TEST_PATH+"/"+"audio_clean",)
    Processing.remove_metadata_from_audio_folder(Config.DEV_PATH+"/"+"audio", Config.DEV_PATH+"/"+"audio_clean",)