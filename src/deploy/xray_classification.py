import os
import shutil
from typing import List
import pandas as pd
from huggingface_hub import hf_hub_download
from src.inference.generate_prediction import GeneratePrediction
from typing import List

class XrayClassification:
    def __init__(self):
        self.output_file_paths: List[str] = []
        self.files: List[str] = []
        self.output_file_path: str = None
        self.file_name: str = None

    def process_file(self, File) -> str:
        """
        Processes an uploaded file, saves it to a directory, and makes predictions using a model.

        Args:
        ----------
        File: A file object

        Returns:
        ----------
        str: A message indicating the prediction result ("Disease Detected" or "No Finding").
        """
        directory = ['/flagged/', '/src/uploaded_images/']
        # Define directory paths
        dir_path = os.getcwd() + directory[1]
        if not os.path.exists(dir_path): os.makedirs(dir_path)
        # Handle file saving and updating lists
        self.file_name = os.path.basename(File)
        output_file_path = os.path.join(dir_path, self.file_name)
        shutil.copyfile(File.name, output_file_path)

        if output_file_path not in self.output_file_paths:
            self.files.append(self.file_name)
            self.output_file_paths.append(output_file_path)
            msg = f"File: < {self.file_name} > uploaded./n/nLoaded files:/n{self.files}"
        else:
            self.files.remove(self.file_name)
            self.output_file_paths.remove(output_file_path)
            self.files.append(self.file_name)
            self.output_file_paths.append(output_file_path)
            msg = f"File: < {self.file_name} > already uploaded./n/nLoaded files:/n{self.files}"
       
        # Handle model downloading
        local_model_dir = "./src/models/downloaded_models/"
        local_filename = "resnetv2_152x2_bit.goog_teacher_in21k_ft_in1k_finetuned_best_score.pth"
        local_model_path = os.path.join(local_model_dir, local_filename)
        model_path = local_model_path

        try:
            if os.path.exists(local_model_path):
                print(f"Model already exists at {local_model_path}. No need to download again.")
            else:
                model_path = hf_hub_download(
                    repo_id="kyang4881/resnetv2_152x2_bit.goog_teacher_in21k_ft_in1k_finetuned_best_score", 
                    filename=local_filename,
                    local_dir=local_model_dir
                )
                print(f"Model downloaded and saved at {model_path}.")
        except Exception as e:
            print(f"An error occurred: {e}")

        # Generate predictions
        test_pred_gen = GeneratePrediction(
            model_path = model_path,
            test_data = pd.DataFrame({'file_path': [output_file_path]})
        )
        test_results = test_pred_gen.predict()

        # Clean up directories
        for d in directory:
            try:
                folder_path = os.path.join(os.getcwd(), d.strip('/\\'))
                
                if os.path.exists(folder_path):
                    for item in os.listdir(folder_path):
                        item_path = os.path.join(folder_path, item)
                        if os.path.isfile(item_path) or os.path.islink(item_path):
                            os.unlink(item_path) 
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path) 
                else:
                    print(f"Directory does not exist: {folder_path}")
            except Exception as e:
                print(f"Error deleting contents of {folder_path}: {e}")

        return "Disease Detected" if test_results['preds'][0] == 1 else "No Finding"