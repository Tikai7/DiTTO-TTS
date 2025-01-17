import subprocess
import os
from tqdm import tqdm

class Processing():

    @staticmethod
    def remove_metadata_from_audio_folder(input_folder, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        else:
            return

        # Parcourir tous les sous-dossiers et fichiers
        for root, _, files in os.walk(input_folder):
            for file in tqdm(files):
                if file.endswith(".opus"):
                    input_file = os.path.join(root, file)
                    
                    relative_path = os.path.relpath(root, input_folder)
                    output_dir = os.path.join(output_folder, relative_path)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    
                    output_file = os.path.join(output_dir, file)

                    command = [
                        "ffmpeg", "-i", input_file,
                        "-map_metadata", "-1", "-c", "copy",
                        output_file
                    ]

                    try:
                        # print(f"Processing {input_file}...")
                        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        # print(f"Processed: {output_file}")
                    except subprocess.CalledProcessError as e:
                        print(f"Error processing {input_file}: {e}")