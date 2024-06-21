import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from typing import Dict, Any, List, Tuple
import numpy as np
import os
from omegaconf import OmegaConf
import logging
import pytorch_lightning as L
from dataclasses import dataclass, field
from data2vec_fairseq.data.modality import Modality
from transformers.optimization import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from modules import MOMEBlock
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from beit2.modeling_pretrain import VisionTransformerForMaskedImageModeling
from beit2 import modeling_pretrain
from timm.models.layers import PatchEmbed

logger = logging.getLogger(__name__)


class AMMData2VecPreTrainingLightningModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.last_n_layer_targets = self.cfg.model.n_fuzed_layers
        self.itc = self.cfg.model.itc

        self.model = AMMData2Vec(cfg=self.cfg.model)

        self.teacher = self.load_beit2_teacher()

        self.model._freeze(self.teacher)
        
        self.save_hyperparameters()

    def forward(self, input_dict):
        return self.model(**input_dict, features_only=False,)

    def training_step(self, batch:Dict[str, Any], batch_idx:int):
        return self._step(batch, batch_idx, stage='train')
    
    def validation_step(self, batch:Dict[str, Any], batch_idx:int):
        return self._step(batch, batch_idx, stage='val')

    def _step(self, batch:Dict[str, Any], batch_idx, stage:str='train'):   
        if 'target' in batch:
            batch.pop('target') # unused, layer activations are the targets

        # assert torch.unique(batch['id']).shape[0] == batch['id'].shape[0], "IDs must be unique for ITC loss."
        # batch['id'] later important for ITC loss?

        text = batch['text']
        padding_mask = batch['padding_mask']
        image = batch['image']

        output_dict_text = self({
            'x': text,
            'modality': Modality.TEXT,
            'padding_mask': padding_mask,
        }) # call "forward"

        output_dict_image = self({
            'x': image,
            'modality': Modality.IMAGE
        }) # call "forward"

        # no mask
        bool_masked_pos = torch.zeros((image.shape[0], self.teacher.patch_embed.num_patches), dtype=torch.bool).to(image.device)

        with torch.no_grad():
            target = self.teacher.forward_features(
                x=image,
                bool_masked_pos=bool_masked_pos,
            )

        kd_losses = []
        for output_dict in [output_dict_text, output_dict_image]:
            _kd_loss = F.mse_loss(output_dict['x'][:, 0], target[:, 0], reduction="mean")
            kd_losses.append(_kd_loss)
        
        kd_loss = sum(kd_losses)
        
        if self.itc:
            kd_loss = kd_loss / 2

            text_features = output_dict_text['x'][:, 0]
            image_features = output_dict_image['x'][:, 0]
            itc_loss = self.itc_loss(text_features=text_features, image_features=image_features)
            self.log(f"{stage}/itc_loss", itc_loss)
        else:
            itc_loss = torch.tensor(0.0).to(kd_loss.device)
        
        loss = kd_loss + itc_loss

        self.log(f"{stage}/kd_text_loss", kd_losses[0])
        self.log(f"{stage}/kd_image_loss", kd_losses[1])
        self.log(f"{stage}/kd_loss", kd_loss)
        self.log(f"{stage}/loss", loss, prog_bar=True)
        
        return loss
    
    def itc_loss(self, text_features:torch.Tensor, image_features:torch.Tensor, stage:str='train') -> torch.Tensor:
        text_features = self.model.itc_head(text_features)
        image_features = self.model.itc_head(image_features)

        scale = self.model.logit_scale.exp()

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        logits_per_image = scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        self._log_similarity(logits_per_image, stage)

        target = torch.arange(len(logits_per_image)).long().to(logits_per_image.device)

        img_itc_acc = (logits_per_image.argmax(dim=1) == target).float().mean()
        text_itc_acc = (logits_per_text.argmax(dim=1) == target).float().mean()
        self.log(f"{stage}/itc_text_acc", text_itc_acc)
        self.log(f"{stage}/itc_image_acc", img_itc_acc)
        self.log(f"{stage}/itc_acc", (img_itc_acc + text_itc_acc) / 2, prog_bar=True)

        itc_loss = (
            F.cross_entropy(logits_per_image.float(), target)
            + F.cross_entropy(logits_per_text.float(), target)
        ) / 2
        return itc_loss
    
    
    def load_beit2_teacher(self):
        sd = torch.load(self.cfg.beit2_args.pretrained_path)['model']
        for key in list(sd.keys()):
            if "cls_pt_layers" in key:
                del sd[key]
        kwargs = OmegaConf.to_container(self.cfg.beit2_args, resolve=True)
        kwargs.pop("pretrained_path")

        beit2 = VisionTransformerForMaskedImageModeling(
            patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
        )

        result = beit2.load_state_dict(sd)
        logger.info(f"Loaded BEiT2 teacher state dict with result: {result}")
        del beit2.lm_head
        return beit2
    
    def _log_similarity(self, logits_per_image: torch.Tensor, stage:str='train') -> None:
        diagonal_mask = torch.eye(logits_per_image.size(0)).bool()
        mean_pos_sim = logits_per_image[diagonal_mask].mean()
        mean_neg_sim = logits_per_image[~diagonal_mask].mean()
        self.log(f"{stage}/itc_mean_pos_similarity", mean_pos_sim)
        self.log(f"{stage}/itc_mean_neg_similarity", mean_neg_sim)

    def configure_optimizers(self):
        wd_params, non_wd_params = self._get_param_groups()
        assert len(wd_params) + len(non_wd_params) == len(list(self.model.parameters()))
        optimizer = torch.optim.AdamW(
            params=[
                {"params": wd_params, "weight_decay": self.cfg.optimizer.weight_decay},
                {"params": non_wd_params, "weight_decay": 0}
            ],
            lr=self.cfg.optimizer.lr,
            betas=tuple(self.cfg.optimizer.betas),
            eps=self.cfg.optimizer.eps,
            # weight_decay=self.cfg.optimizer.weight_decay -> not needed becuase of param groups
        )
        
        if self.cfg.optimizer.warmup:
            name = self.cfg.optimizer_schedule.type
            if name == 'cosine':
                scheduler = get_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=self.cfg.optimizer_schedule.warmup_steps,
                    num_training_steps=self.cfg.optimizer_schedule.max_steps,
                )
            else:
                scheduler = get_constant_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=self.cfg.optimizer_schedule.warmup_steps,
                )
            return [optimizer], [{"scheduler": scheduler, "interval": "step", "name": name}]
        else:
            return optimizer
        
    def _get_param_groups(self):
        wd_params, non_wd_params = [], []
        for name, param in self.model.named_parameters():
            if len(param.shape) == 1 or name.endswith(".bias") or "extra_tokens" in name or 'embed_tokens' in name:
                non_wd_params.append(param)
            else:
                wd_params.append(param)
        return wd_params, non_wd_params
        
    def log(self, *args, **kwargs):
        super().log(batch_size=self.cfg.data.dataloader.batch_size, sync_dist=True, *args, **kwargs)


@dataclass
class PretrainedStateDictsConfig():
    audio:str = 'base_libri.pt'
    image:str = 'base_imagenet.pt'
    text:str = 'nlp_base.pt'

@dataclass
class AMMData2VecConfig():
    pretrained_path:str = '../models'
    pretrained: PretrainedStateDictsConfig = field(default_factory=PretrainedStateDictsConfig)

    shared_attn: bool = True
    n_fuzed_layers: int = 2

    use_tte: bool = True
    itc: bool = True

    embed_dim: int = 768

    depth: int = 6
    num_heads: int = 12
    mlp_ratio: float = 4.0
    encoder_dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.0
    post_mlp_drop: float = 0.1
    norm_eps: float = 1e-6
    norm_affine: bool = True
    layer_norm_first: bool = False
    dropout_input: float = 0.0
    start_drop_path_rate: float = 0
    end_drop_path_rate: float = 0
    layerdrop: float = 0.0

    seed: int = 42

class AMMData2Vec(nn.Module):
    def __init__(self,
                 cfg: AMMData2VecConfig,
                 ):
        super(AMMData2Vec, self).__init__()
        self.cfg = cfg
        self.supported_modalities = [Modality.IMAGE, Modality.TEXT]
        self.img_size = 224
        self.patch_size = 16

        self.patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=3,
            embed_dim=self.cfg.embed_dim,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.cfg.embed_dim))

        self.text_embeddings = nn.Embedding(self.cfg.vocab_size, self.cfg.embed_dim)

        self.token_type_embedding = nn.Embedding(len(self.supported_modalities), self.cfg.embed_dim)

        if self.cfg.itc:
            self.itc_head = nn.Linear(self.cfg.embed_dim, self.cfg.embed_dim, bias=False)
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        if self.cfg.use_tte:
            self.token_type_embedding = nn.Embedding(len(self.supported_modalities), self.cfg.embed_dim)
            self.post_tte_norm = nn.LayerNorm(self.cfg.embed_dim, eps=self.cfg.norm_eps, elementwise_affine=self.cfg.norm_affine)

        self.dropout_input = nn.Dropout(self.cfg.dropout_input)

        dpr = np.linspace(self.cfg.start_drop_path_rate, self.cfg.end_drop_path_rate, self.cfg.depth)

        blocks:List[MOMEBlock] = []
        fuzed_block_indices = list(range(self.cfg.depth - self.cfg.n_fuzed_layers, self.cfg.depth))
        logger.info(f"Fuzed block indices: {fuzed_block_indices}")
        for i in range(self.cfg.depth):
            if self.cfg.depth - i <= self.cfg.n_fuzed_layers:
                blocks.append(self.make_block(drop_path=dpr[i], multimodal=True, with_fuzed=True))
            else:
                blocks.append(self.make_block(drop_path=dpr[i], multimodal=True, with_fuzed=False))

        self.blocks:nn.ModuleList[str, MOMEBlock] = nn.ModuleList(blocks)

        self.layerdrop = self.cfg.layerdrop

        self.apply(init_bert_params)

        # init pretrained later, so that they are not part of the model's parameters when model is initialized
        assert hasattr(self, 'blocks'), "Blocks must be initialized before initializing the model."
        self._init_from_pretrained()
        
    def make_block(self, drop_path, dim=None, heads=None, multimodal=False, with_fuzed=False, shared_attn=True):
        make_layer_norm = partial(
            nn.LayerNorm, eps=self.cfg.norm_eps, elementwise_affine=self.cfg.norm_affine
        )

        return MOMEBlock(
            self.cfg.embed_dim if dim is None else dim,
            self.cfg.num_heads if heads is None else heads,
            self.cfg.mlp_ratio,
            qkv_bias=True,
            drop=self.cfg.encoder_dropout,
            attn_drop=self.cfg.attention_dropout,
            mlp_drop=self.cfg.activation_dropout,
            drop_path=drop_path,
            norm_layer=make_layer_norm,
            multimodal=multimodal,
            with_fuzed=with_fuzed,
            shared_attn=shared_attn,
        )

    def _init_from_pretrained(self) -> None:
        state_dict, pretrained_beit2_img_rpbt = self._get_beit2_components()
        d2v_sd = self._get_d2v_text_components()
        state_dict.update(d2v_sd)
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        assert len(unexpected_keys) == 0, f"Unexpected keys in state dict: {unexpected_keys}"
        logger.info(f"Missing keys in state dict: {missing_keys}")
        if self.cfg.text_pretraining:
            del self.patch_embed
            del self.cls_token
            for block in self.blocks:
                block.remove_modality(Modality.IMAGE)
                self._freeze(block.norm1)
                self._freeze(block.attn)

        self.build_relative_position_embed(pretrained_beit2_img_rpbt)

    def _get_beit2_components(self) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        state_dict = torch.load(os.path.join(self.cfg.pretrained_path, self.cfg.beit2_state_dict))['model']
        to_pop = ['mask_token', 'norm.weight', 'norm.bias', 'lm_head.weight', 'lm_head.bias',]
        for key in to_pop:
            state_dict.pop(key)

        pretrained_beit2_img_rpbt = state_dict.pop('rel_pos_bias.relative_position_bias_table')

        # remove layers too deep for this model
        state_dict = self._pop_blocks_from_n(state_dict, self.cfg.depth - self.cfg.n_fuzed_layers)

        for key in list(state_dict.keys()):
            if "cls_pt_layers" in key: # remove beit2 patch aggregation layers
                del state_dict[key]
                continue

            if 'cls_token' in key:
                continue

            if 'attn' in key or 'norm1' in key:
                continue

            if 'mlp' in key:
                block_elem = 'mlp'
            elif 'norm2' in key:
                block_elem = 'norm2'
            else:
                del state_dict[key]
                continue

            state_dict[key.replace(f'blocks.{block_elem}', f'blocks.image.{block_elem}')] = state_dict.pop(key)

        return state_dict, pretrained_beit2_img_rpbt

    def _get_d2v_text_components(self) -> Dict[str, torch.Tensor]:
        d2v_sd = torch.load(os.path.join(self.cfg.pretrained_path, self.cfg.d2v_text_state_dict))['model']
        d2v_sd.pop('_ema')
        d2v_sd = {k:v for k,v in d2v_sd.items() if 'decoder' not in k}
        d2v_sd = self._pop_blocks_from_n(d2v_sd, self.cfg.depth-self.cfg.n_fuzed_layers)
        text_embed_weight = d2v_sd.pop('modality_encoders.TEXT.local_encoder.embed_tokens.weight')

        for key in list(d2v_sd.keys()):
            if 'mlp' in key:
                block_elem = 'mlp'
            elif 'norm2' in key:
                block_elem = 'norm2'
            else:
                del d2v_sd[key]
                continue

            d2v_sd[key.replace(f'blocks.{block_elem}', f'blocks.text.{block_elem}')] = d2v_sd.pop(key)
        d2v_sd['text_embeddings.weight'] = text_embed_weight
        return d2v_sd

    def _pop_blocks_from_n(self, state_dict:Dict[str, torch.Tensor], n:int) -> Dict[str, torch.Tensor]:
        for key in list(state_dict.keys()):
            if 'block' in key and int(key.split('.')[1]) >= n:
                del state_dict[key]
        return state_dict

    # from VLMo: https://github.com/microsoft/unilm/blob/master/vlmo/vlmo/modules/vlmo_module.py
    def build_relative_position_embed(self, pretrained_beit2_img_rpbt:torch.Tensor):
        window_size = (14, 14)
        num_heads = self.cfg.num_heads
        max_text_len_of_initckpt = 197 # from text pretaining -> frozen vision self-attention used, used to 197 tokens (224x224 image with window size 14x14)
        max_text_len = self.cfg.max_text_len
        max_imag_len = window_size[0] * window_size[1] + 1 #197
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.text_num_relative_distance = 2 * max_text_len_of_initckpt
        self.all_num_relative_distance = self.num_relative_distance + self.text_num_relative_distance + 2

        # contains both text and image relative positions
        # relative_position_index (image) and text_relative_position_index (text) are used to get the relative position bias
        # ... for the corresponding modality (so there is no actual shared relative position bias between image and text)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.all_num_relative_distance, num_heads))
        
        # init relatie position bias table from beit2 for images
        assert pretrained_beit2_img_rpbt.size(0) == self.num_relative_distance
        self.relative_position_bias_table.data[:self.num_relative_distance, :num_heads] = pretrained_beit2_img_rpbt[:, :num_heads]
        
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = \
            torch.zeros(size=(window_size[0] * window_size[1] + 1, ) * 2, dtype=relative_coords.dtype)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1
        self.image_relative_position_index = relative_position_index
        
        text_position_ids = torch.arange(max_text_len-1)
        text_rel_pos_mat = text_position_ids.unsqueeze(-2) - text_position_ids.unsqueeze(-1)
        min_distance = int(2-max_text_len_of_initckpt) #-194
        # rank_zero_info("min_distance: {}".format(min_distance))
        text_rel_pos_mat = text_rel_pos_mat - min_distance
        text_rel_pos_mat += (self.num_relative_distance + 2)
        text_relative_position_index = \
            torch.zeros(size=(max_text_len, ) * 2, dtype=relative_coords.dtype)
        text_relative_position_index[1:, 1:] = text_rel_pos_mat
        text_relative_position_index[0, 0:] = self.all_num_relative_distance - 3
        text_relative_position_index[0:, 0] = self.all_num_relative_distance - 2
        text_relative_position_index[0, 0] = self.all_num_relative_distance - 1
        self.text_relative_position_index = text_relative_position_index
        
        # text2imag_relative_position_index = torch.ones(max_text_len, max_imag_len) * (self.num_relative_distance)
        # imag2text_relative_position_index = torch.ones(max_imag_len, max_text_len) * (self.num_relative_distance + 1)

        # text_row_relative_position_index = torch.cat((text_relative_position_index, text2imag_relative_position_index), 1)
        # imag_row_relative_position_index = torch.cat((imag2text_relative_position_index, relative_position_index), 1)
        # text_imag_relative_position_index = torch.cat((text_row_relative_position_index, imag_row_relative_position_index), 0)
        # self.text_imag_relative_position_index = text_imag_relative_position_index

    def get_relative_position_bias(self, modality:Modality) -> torch.Tensor:
        if modality == Modality.IMAGE:
            relative_position_bias = \
                self.relative_position_bias_table[self.image_relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
            return relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        elif modality == Modality.TEXT:
            relative_position_bias = \
                self.relative_position_bias_table[self.text_relative_position_index.view(-1)].view(
                    self.cfg.max_text_len, -1)  # nT,nH
            return relative_position_bias.permute(1, 0).contiguous()  # nH, nT
        else:
            raise ValueError(f"Modality {modality} not supported")

    def forward(
        self,
        x:torch.Tensor,
        modality:Modality,
        id:torch.Tensor=None,
        padding_mask:torch.Tensor=None,
        features_only:bool=False,
    ):

        if modality == Modality.IMAGE:
            embed_func = self.visual_embed
        elif modality == Modality.TEXT:
            embed_func = self.text_embed
        else:
            raise ValueError(f"Modality {modality} not supported")
        x = embed_func(x)

        if self.cfg.use_tte:
            token_ids = torch.full(x.shape[:-1], modality.value-2).to(x.device)
            x = x + self.token_type_embedding(token_ids)
            x = self.post_tte_norm(x)

        if self.dropout_input is not None:
            x = self.dropout_input(x)
        
        relative_position_bias = self.get_relative_position_bias(modality)
        for blk in self.blocks:
            x, _ = blk(
                x,
                modality=modality,
                padding_mask=padding_mask,
                relative_position_bias=relative_position_bias,
            )

        out = {
            "x": x,
        }
        return out
    
    def visual_embed(self, x:torch.Tensor):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        token_ids = torch.ones(x.shape[:-1]).to(x.device)
        x = x + self.token_type_embedding(token_ids)
        return x
    
    def text_embed(self, x:torch.Tensor):
        x = self.text_embeddings(x)
        token_ids = torch.zeros(x.shape[:-1]).to(x.device)
        x = x + self.token_type_embedding(token_ids)
        return x
    

    def extract_features(
        self, x:torch.Tensor, modality:Modality, padding_mask=None
    ):
        res = self.forward(
            x=x,
            modality=modality,
            padding_mask=padding_mask,
            features_only=True,
        )
        return res
    
    def encode_modality(self, x:torch.Tensor, modality:Modality, padding_mask=None, normalize:bool=True):
        output = self.extract_features(
            x=x,
            modality=modality,
            padding_mask=padding_mask,
        )['x']

        output = output[:, 0]
        if self.cfg.itc:
            output = self.itc_head(output)

        if normalize:
            output = output / output.norm(dim=-1, keepdim=True)
        
        return output # shape: (batch_size, embed_dim)

    def encode_audio(self, audio, padding_mask, normalize:bool=True):
        return self.encode_modality(x=audio,
                                    modality=Modality.AUDIO,
                                    padding_mask=padding_mask,
                                    normalize=normalize)
    
    def encode_image(self, image, normalize:bool=True):
        return self.encode_modality(x=image,
                                    modality=Modality.IMAGE,
                                    normalize=normalize)

    def encode_text(self, text, padding_mask, normalize:bool=True):
        return self.encode_modality(x=text,
                                    modality=Modality.TEXT,
                                    padding_mask=padding_mask,
                                    normalize=normalize)
    
    def freeze_attention_blocks(self):
        for block in self.blocks:
            self._freeze(block.attn)

    def unfreeze_attention_blocks(self):
        for block in self.blocks:
            self._unfreeze(block.attn)

    def _freeze(self, module:nn.Module) -> None:
        for param in module.parameters():
            param.requires_grad = False
        module.eval()

    def _unfreeze(self, module:nn.Module) -> None:
        for param in module.parameters():
            param.requires_grad = True
        module.train()


    def prepare_fine_tuning(self, keep_modality:Modality) -> None:
        self._remove_modalities_except(keep_modality=keep_modality)
        self._unfreeze(self)

    def _remove_modalities_except(self, keep_modality:Modality) -> None:
        """
        Removes all modalities from the model except the ones specified.
        Useful when fine-tuning the model on downstream task
        involving only a subset of the supported modalities.
        """
        # comparison done on name basis, as on "enum" basis yields problems after serialization
        for modality in self.supported_modalities:
            modality_str = modality.name.lower()
            if modality_str != keep_modality:
                for block in self.blocks:
                    block.remove_modality(modality)

        if keep_modality != Modality.IMAGE:
            del self.patch_embed
            del self.cls_token
        else:
            del self.text_embeddings
