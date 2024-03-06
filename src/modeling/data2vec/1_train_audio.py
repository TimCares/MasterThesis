from src.models.data2vec_audio import Data2VecAudioModel, Data2VecAudioConfig
from src.data.unimodal import get_raw_librispeech_dataset
import logging
import torch
import os
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PATH_TO_MODEL = '' + os.sep
MODEL_NAME = "data2vec_audio.pt"
EPOCHS = 50
BATCH_SIZE = 128
PRINT_STATS_EVERY = 100 # ... batches
DATASET = "train-clean-100"

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = get_raw_librispeech_dataset(dataset=DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=-1)

    model_params = {
    }

    config = Data2VecAudioConfig(**model_params)
    model = Data2VecAudioModel(config)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        running_loss = 0
        last_loss = 0

        for idx, (x, padding_mask) in enumerate(dataset):
            optimizer.zero_grad()
            x = x.to(device)
            padding_mask = padding_mask.to(device)
            result = model.forward(source=x, padding_mask=padding_mask, mask=False) # mask=False -> Teach model has no masked input
            loss = result["losses"]["regression"]
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if idx % PRINT_STATS_EVERY == (PRINT_STATS_EVERY-1):
                last_loss = running_loss / PRINT_STATS_EVERY
                logger.info(f"Epoch: {epoch}, Batch: {idx}, Avg. loss per Batch: {last_loss}, \n\tTarget Var: {loss.item()}, Pred Var: {loss.item()}")
                running_loss = 0

    torch.save(model.state_dict(), PATH_TO_MODEL+MODEL_NAME)

    train_params = {
        "model_params": model_params,
        "config_obj": config,
        "dataset": DATASET,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "print_stats_every": PRINT_STATS_EVERY,
        "path_to_model": PATH_TO_MODEL+MODEL_NAME,
    }

    with open(PATH_TO_MODEL+"data2vec_audio_params.json", "w") as f:
        json.dump(train_params, f)


if __name__ == "__main__":
    train()