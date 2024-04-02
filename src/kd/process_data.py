import torch

def extract_targets(model_state_dict_path:str):
    state_dict = torch.load(model_state_dict_path, map_location="cpu")['model']
    
    

if __name__ == "__main__":
    extract_targets()