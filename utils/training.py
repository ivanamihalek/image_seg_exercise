import torch
from tqdm import tqdm

import settings


def train_fn(data_loader, model, optimizer, verbose=False):
    model.train()
    total_loss = 0.0

    # tqdm is a progress meter
    iterator =  tqdm(data_loader) if verbose else data_loader
    for images, masks in iterator:

        images = images.to(settings.DEVICE)
        masks = masks.to(settings.DEVICE)

        optimizer.zero_grad()
        logits, loss = model(images, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        return total_loss / len(data_loader)


def eval_fn(data_loader, model, verbose=False):

    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        iterator =  tqdm(data_loader) if verbose else data_loader
        for images, masks in iterator:

            images = images.to(settings.DEVICE)
            masks = masks.to(settings.DEVICE)

            logits, loss = model(images, masks)

            total_loss += loss.item()

    return total_loss / len(data_loader)
