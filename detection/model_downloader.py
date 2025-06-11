import os
import requests
from pathlib import Path
import gdown

def download_model():
    """Download the ONNX model from Google Drive during deployment"""
    # Your Google Drive file ID
    file_id = "19Ktzyh8MKENzjdwtwmZrhKspF-DMKQqJ"
    model_path = Path("cattle_disease_model.onnx")
    
    if not model_path.exists():
        print("Downloading AI model from Google Drive...")
        try:
            # Google Drive direct download URL
            url = f"https://drive.google.com/uc?export=download&id={file_id}"
            
            # Download with requests (handles large files)
            response = requests.get(url, stream=True)
            
            # Handle Google Drive's virus scan warning for large files
            if 'download_warning' in response.text:
                # Extract the actual download link
                import re
                confirm_token = re.search(r'confirm=([0-9A-Za-z_]+)', response.text)
                if confirm_token:
                    confirm_url = f"{url}&confirm={confirm_token.group(1)}"
                    response = requests.get(confirm_url, stream=True)
            
            response.raise_for_status()
            
            # Save the file
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            print(f"Model downloaded successfully! Size: {model_path.stat().st_size / (1024*1024):.1f} MB")
            return True
            
        except Exception as e:
            print(f"Error downloading model: {e}")
            print("Trying alternative method...")
            
            # Alternative: Use gdown library
            try:
                import gdown
                gdown.download(f"https://drive.google.com/uc?id={file_id}", str(model_path), quiet=False)
                print("Model downloaded successfully with gdown!")
                return True
            except Exception as e2:
                print(f"Alternative download also failed: {e2}")
                return False
    else:
        print("Model already exists!")
        return True

if __name__ == "__main__":
    download_model()
