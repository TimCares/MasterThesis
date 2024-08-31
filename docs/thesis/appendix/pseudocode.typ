== Pseudocode

#figure(
  rect(
    ```python
    # model: pretrained (e.g. distilled) model
    # layer_norm: layer normalization layer
    # cls_head: linear classifier -> nn.Linear(D, C)
    # x: batch of images (B, 3, H, W)
    def image_finetune_forward(model, layer_norm, cls_head, x, linear_probe):
        
        if linear_probe:
          with torch.no_grad():
            x = model(x) # (B, T, D)
        else:
          x = model(x) # (B, T, D)
        
        x = x[:, 1:] # remove cls token (B, T-1, D)
        x = x.mean(dim=1) # mean over all patches (B, D)
        x = layer_norm(x)
        x = cls_head(x) # (B, C)
        pred = x.argmax(dim=-1) # (B, )
        return pred
    ```
  ), 
caption: [Pytorch pseudocode forward pass for finetuning a pretrained model on image classification tasks. The output
of the forward pass is the predicted class index for each image in the batch.],
) <zero_shot_retrieval>