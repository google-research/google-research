# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2022 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

from pytorch_lightning.accelerators import accelerator
from opt import get_opts
import torch
from collections import defaultdict

from torch.utils.data import DataLoader
from datasets import dataset_dict

# models
from models.nerf import *
from models.rendering import *
from models.coverage import *
from utils.point_utils import scale_and_shift_points

# optimizer, scheduler, visualization
from utils import *

# losses
from losses import loss_dict

# metrics
from metrics import *

# pytorch-lightning
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin


class NeRFSystem(LightningModule):
  """NeRF System."""

  def __init__(self, hparams):
    super().__init__()
    self.save_hyperparameters(hparams)

    self.color_loss = loss_dict['color'](coef=1.)
    self.coverage_loss = loss_dict['coverage'](coef=hparams.coverage_pen_weight)

    if hparams.with_semantics:
      self.semantics_loss = loss_dict['semantics'](coef=0.02)
      self.seg_palette = make_palette(hparams.N_classes)

    self.embedding_xyz = Embedding(hparams.N_emb_xyz)
    self.embedding_dir = Embedding(hparams.N_emb_dir)
    self.embeddings = {'xyz': self.embedding_xyz, 'dir': self.embedding_dir}

    coverage_func_class = coverage_funcs[hparams.coverage_type]

    self.nerf_coarse = Nerflets(
        n=hparams.N_nerflets,
        D=hparams.N_mlp_depth,
        W=hparams.N_mlp_width,
        in_channels_xyz=6 * hparams.N_emb_xyz + 3,
        in_channels_dir=6 * hparams.N_emb_dir + 3,
        with_semantics=hparams.with_semantics,
        n_classes=hparams.N_classes,
        topk=hparams.K_nerflets)
    self.coverage_coarse = coverage_func_class(
        n=hparams.N_nerflets,
        with_mlp=hparams.coverage_with_mlp,
        softmax_temp=hparams.coverage_type_rbf_softmax_temp,
        weight_min=hparams.coverage_type_rbf_weight_min)
    self.models = {'coarse': self.nerf_coarse}
    self.coverage_models = {'coarse': self.coverage_coarse}
    self.all_trainable_models = [self.nerf_coarse, self.coverage_coarse]

    if hparams.with_bg_nerf:
      self.bg_nerf = BgNeRF(
          D=3,
          W=32,
          in_channels_dir=6 * hparams.N_emb_dir + 3,
          with_semantics=hparams.with_semantics,
          n_classes=hparams.N_classes)
      self.models['bg'] = self.bg_nerf
      load_ckpt(self.bg_nerf, hparams.weight_path, 'bg_nerf')
      self.all_trainable_models += [self.bg_nerf]

    if hparams.N_importance > 0:
      self.nerf_fine = self.nerf_coarse
      self.coverage_fine = self.coverage_coarse
      self.models['fine'] = self.nerf_fine
      self.coverage_models['fine'] = self.coverage_fine
      load_ckpt(self.nerf_fine, hparams.weight_path, 'nerf_fine')
      load_ckpt(self.coverage_fine, hparams.weight_path, 'coverage_fine')
    else:
      load_ckpt(self.nerf_coarse, hparams.weight_path, 'nerf_coarse')
      load_ckpt(self.coverage_coarse, hparams.weight_path, 'coverage_coarse')

  def forward(self, rays):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    results = defaultdict(list)
    for i in range(0, B, self.hparams.chunk):
      rendered_ray_chunks = \
          render_rays(self.models,
                      self.coverage_models,
                      self.embeddings,
                      rays[i:i + self.hparams.chunk],
                      self.hparams.N_samples,
                      self.hparams.use_disp,
                      self.hparams.perturb,
                      self.hparams.noise_std,
                      self.hparams.N_importance,
                      self.hparams.chunk,  # chunk size is effective in val mode
                      self.train_dataset.white_back,
                      with_semantics=self.hparams.with_semantics,
                      point_transform_func=self.xyz_transformation,
                      topk=self.hparams.K_nerflets)

      for k, v in rendered_ray_chunks.items():
        results[k] += [v]

    for k, v in results.items():
      results[k] = torch.cat(v, 0)
    return results

  def setup(self, stage):
    dataset = dataset_dict[self.hparams.dataset_name]
    kwargs = {
        'root_dir': self.hparams.root_dir,
        'img_wh': tuple(self.hparams.img_wh)
    }
    if self.hparams.dataset_name == 'llff':
      kwargs['spheric_poses'] = self.hparams.spheric_poses
      kwargs['val_num'] = self.hparams.num_gpus
    if self.hparams.dataset_name == 'scannet':
      kwargs['ref_loc_file'] = self.hparams.nerflets_loc_ref_mesh
    self.train_dataset = dataset(split='train', **kwargs)
    self.val_dataset = dataset(split='val', **kwargs)

    if self.hparams.normalize_mlp_inputs:
      assert np.allclose(self.train_dataset.scene_box,
                         self.val_dataset.scene_box)
      scene_box = torch.tensor(self.train_dataset.scene_box)
      self.xyz_transformation = lambda x: \
          scale_and_shift_points(scene_box, x)
    else:
      self.xyz_transformation = None

    if self.hparams.dataset_name == 'scannet':
      self.coverage_coarse.setup_loc(self.train_dataset.ref_points,
                                     self.hparams.freeze_nerflets_loc)
    else:
      self.coverage_coarse.setup_loc(None, self.hparams.freeze_nerflets_loc)

  def configure_optimizers(self):
    self.optimizer = get_optimizer(self.hparams, self.all_trainable_models)
    scheduler = get_scheduler(self.hparams, self.optimizer)
    return [self.optimizer], [scheduler]

  def train_dataloader(self):
    return DataLoader(
        self.train_dataset,
        shuffle=True,
        num_workers=4,
        batch_size=self.hparams.batch_size,
        pin_memory=True)

  def val_dataloader(self):
    return DataLoader(
        self.val_dataset,
        shuffle=False,
        num_workers=4,
        batch_size=1,  # validate one image (H*W rays) at a time
        pin_memory=True)

  def training_step(self, batch, batch_nb):
    if self.global_step % self.hparams.coverage_log_freq == 0:
      sif_filename = f'sif_{self.global_step:06d}.txt'
      sif_dir = os.path.join(self.logger.log_dir, 'sif')
      os.makedirs(sif_dir, exist_ok=True)
      self.coverage_coarse.dumps(os.path.join(sif_dir, sif_filename))
      if 'fine' in self.coverage_models:
        sif_dir = os.path.join(self.logger.log_dir, 'sif_fine')
        os.makedirs(sif_dir, exist_ok=True)
        self.coverage_fine.dumps(os.path.join(sif_dir, sif_filename))

    rays, rgbs = batch['rays'], batch['rgbs']
    results = self(rays)
    color_loss = self.color_loss(results, rgbs)
    coverage_loss = self.coverage_loss(results)
    semantics_loss = self.semantics_loss(results, batch['sems']) \
        if self.hparams.with_semantics else 0.
    loss = color_loss + coverage_loss + semantics_loss

    with torch.no_grad():
      typ = 'fine' if 'rgb_fine' in results else 'coarse'
      psnr_ = psnr(results[f'rgb_{typ}'], rgbs)

    self.log('lr', get_learning_rate(self.optimizer))
    self.log('train/loss', loss)
    self.log('train/color_loss', color_loss)
    self.log('train/coverage_loss', coverage_loss)
    self.log('train/semantics_loss', semantics_loss)
    self.log('train/psnr', psnr_, prog_bar=True)

    return loss

  def validation_step(self, batch, batch_nb):
    rays, rgbs, sems, mask = batch['rays'], batch['rgbs'], batch['sems'], batch[
        'valid_mask']
    rays = rays.squeeze()  # (H*W, 3)
    rgbs = rgbs.squeeze()  # (H*W, 3)
    mask = mask.squeeze()
    results = self(rays)
    log = {'val_color_loss': self.color_loss(results, rgbs)}
    typ = 'fine' if 'rgb_fine' in results else 'coarse'

    if batch_nb == 0:
      W, H = self.hparams.img_wh
      img = results[f'rgb_{typ}'].view(H, W, 3).permute(2, 0,
                                                        1).cpu()  # (3, H, W)
      img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
      depth = visualize_depth(results[f'depth_{typ}'].view(H, W))  # (3, H, W)
      stack = torch.stack([img_gt, img, depth])  # (3, 3, H, W)
      self.logger.experiment.add_images('val/GT_pred_depth', stack,
                                        self.global_step)

      if self.hparams.with_semantics:
        sems = sems.squeeze()  # (H*W,)
        sem_logits = results[f'sem_logits_{typ}'].view(
            H, W, self.hparams.N_classes).cpu()
        sem = sem_logits.max(dim=2)[1]
        sem_gt = sems.view(H, W).cpu().long()
        sem = self.seg_palette[sem.view(-1)].view(H, W, 3).permute(2, 0, 1)
        sem_gt = self.seg_palette[sem_gt.view(-1)].view(H, W,
                                                        3).permute(2, 0, 1)
        stack = torch.stack([sem_gt, sem])  # (2, 3, H, W)
        self.logger.experiment.add_images('val/semGT_sem', stack,
                                          self.global_step)

    # psnr_ = psnr(results[f'rgb_{typ}'], rgbs, valid_mask=mask)
    psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
    log['val_psnr'] = psnr_

    return log

  def validation_epoch_end(self, outputs):
    mean_color_loss = torch.stack([x['val_color_loss'] for x in outputs]).mean()
    mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()

    self.log('val/color_loss', mean_color_loss)
    self.log('val/psnr', mean_psnr, prog_bar=True)


def main(hparams):
  system = NeRFSystem(hparams)
  ckpt_cb = ModelCheckpoint(
      dirpath=f'logs/{hparams.exp_name}/ckpts/',
      filename='{epoch:d}',
      monitor='val/psnr',
      mode='max',
      save_top_k=5)
  pbar = TQDMProgressBar(refresh_rate=1)
  callbacks = [ckpt_cb, pbar]

  logger = TensorBoardLogger(
      save_dir='logs', name=hparams.exp_name, default_hp_metric=False)

  trainer = Trainer(
      max_epochs=hparams.num_epochs,
      callbacks=callbacks,
      resume_from_checkpoint=hparams.ckpt_path,
      logger=logger,
      enable_model_summary=False,
      accelerator='auto',
      devices=hparams.num_gpus,
      num_sanity_val_steps=1,
      benchmark=True,
      profiler='simple' if hparams.num_gpus == 1 else None,
      strategy=DDPPlugin(
          find_unused_parameters=False) if hparams.num_gpus > 1 else None)

  trainer.fit(system)


if __name__ == '__main__':
  hparams = get_opts()
  main(hparams)
