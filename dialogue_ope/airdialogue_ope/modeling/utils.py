# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

from transformers.modeling_utils import *
import copy
import torch
import torch.nn.functional as F

REGFUNC = {
    "square": lambda x: x**2,
    "square_cut5": lambda x: F.relu(x**2 - 5),
    "square_cut100": lambda x: F.relu(x**2 - 100),
    "abs_cut10": lambda x: F.relu(x.abs() - 10),
    "abs_cut20": lambda x: F.relu(x.abs() - 20),
}


class PreTrainedModels(PreTrainedModel):

  @classmethod
  def from_pretrained(cls, pretrained_model_name_or_path, *model_args,
                      **kwargs):
    config = kwargs.pop("config", None)
    state_dict = kwargs.pop("state_dict", None)
    cache_dir = kwargs.pop("cache_dir", None)
    from_tf = kwargs.pop("from_tf", False)
    force_download = kwargs.pop("force_download", False)
    resume_download = kwargs.pop("resume_download", False)
    proxies = kwargs.pop("proxies", None)
    output_loading_info = kwargs.pop("output_loading_info", False)
    local_files_only = kwargs.pop("local_files_only", False)
    use_cdn = kwargs.pop("use_cdn", True)

    # Load config if we don't provide a configuration
    if not isinstance(config, PretrainedConfig):
      config_path = config if config is not None else pretrained_model_name_or_path
      config, model_kwargs = cls.config_class.from_pretrained(
          config_path,
          *model_args,
          cache_dir=cache_dir,
          return_unused_kwargs=True,
          force_download=force_download,
          resume_download=resume_download,
          proxies=proxies,
          local_files_only=local_files_only,
          **kwargs,
      )
    else:
      model_kwargs = kwargs

    # Load model
    if pretrained_model_name_or_path is not None:
      if os.path.isdir(pretrained_model_name_or_path):
        if from_tf and os.path.isfile(
            os.path.join(pretrained_model_name_or_path,
                         TF_WEIGHTS_NAME + ".index")):
          # Load from a TF 1.0 checkpoint
          archive_file = os.path.join(pretrained_model_name_or_path,
                                      TF_WEIGHTS_NAME + ".index")
        elif from_tf and os.path.isfile(
            os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)):
          # Load from a TF 2.0 checkpoint
          archive_file = os.path.join(pretrained_model_name_or_path,
                                      TF2_WEIGHTS_NAME)
        elif os.path.isfile(
            os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)):
          # Load from a PyTorch checkpoint
          archive_file = os.path.join(pretrained_model_name_or_path,
                                      WEIGHTS_NAME)
        else:
          raise EnvironmentError(
              "Error no file named {} found in directory {} or `from_tf` set to False"
              .format(
                  [WEIGHTS_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME + ".index"],
                  pretrained_model_name_or_path,
              ))
      elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(
          pretrained_model_name_or_path):
        archive_file = pretrained_model_name_or_path
      elif os.path.isfile(pretrained_model_name_or_path + ".index"):
        assert (
            from_tf
        ), ("We found a TensorFlow checkpoint at {}, please set from_tf to True"
            " to load from this checkpoint").format(
            pretrained_model_name_or_path + ".index")
        archive_file = pretrained_model_name_or_path + ".index"
      else:
        archive_file = hf_bucket_url(
            pretrained_model_name_or_path,
            filename=(TF2_WEIGHTS_NAME if from_tf else WEIGHTS_NAME),
            use_cdn=use_cdn,
        )

      try:
        # Load from URL or cache if already cached
        resolved_archive_file = cached_path(
            archive_file,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
        )
        if resolved_archive_file is None:
          raise EnvironmentError
      except EnvironmentError:
        msg = (
            f"Can't load weights for '{pretrained_model_name_or_path}'. Make "
            f"sure that:\n\n- '{pretrained_model_name_or_path}' is a correct "
            f"model identifier listed on 'https://huggingface.co/models'\n\n- "
            f"or '{pretrained_model_name_or_path}' is the correct path to a "
            f"directory containing a file named one of {WEIGHTS_NAME}, "
            f"{TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME}.\n\n"
        )
        raise EnvironmentError(msg)

      if resolved_archive_file == archive_file:
        logger.info("loading weights file {}".format(archive_file))
      else:
        logger.info("loading weights file {} from cache at {}".format(
            archive_file, resolved_archive_file))
    else:
      resolved_archive_file = None

    # Instantiate model.
    model = cls(config, *model_args, **model_kwargs)

    if state_dict is None and not from_tf:
      try:
        state_dict = torch.load(resolved_archive_file, map_location="cpu")
      except Exception:
        raise OSError(
            "Unable to load weights from pytorch checkpoint file. "
            "If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True. "
        )

    missing_keys = []
    unexpected_keys = []
    error_msgs = []

    if from_tf:
      if resolved_archive_file.endswith(".index"):
        # Load from a TensorFlow 1.X checkpoint - provided by original authors
        model = cls.load_tf_weights(
            model, config, resolved_archive_file[:-6])  # Remove the '.index'
      else:
        # Load from our TensorFlow 2.0 checkpoints
        try:
          from transformers import load_tf2_checkpoint_in_pytorch_model

          model = load_tf2_checkpoint_in_pytorch_model(
              model, resolved_archive_file, allow_missing_keys=True)
        except ImportError:
          logger.error(
              "Loading a TensorFlow model in PyTorch, requires both PyTorch and TensorFlow to be installed. Please see "
              "https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions."
          )
          raise
    else:
      # Convert old format to new format if needed from a PyTorch state_dict
      has_all_sub_modules = all(
          any(s.startswith(_prefix)
              for s in state_dict.keys())
          for _prefix in model.base_model_prefixs)
      has_prefix_module = any(
          s.startswith(model.base_model_prefix) for s in state_dict.keys())

      old_keys = list(state_dict.keys())
      for key in old_keys:
        new_key = key

        if "gamma" in key:
          new_key = new_key.replace("gamma", "weight")
        if "beta" in key:
          new_key = new_key.replace("beta", "bias")

        _state = state_dict.pop(key)
        if has_all_sub_modules:
          state_dict[new_key] = _state
        elif not has_prefix_module:
          for _prefix in model.base_model_prefixs:
            _key = _prefix + "." + new_key
            state_dict[_key] = _state
        else:
          if new_key.startswith(model.base_model_prefix):
            for _prefix in model.base_model_prefixs:
              _key = _prefix + new_key[len(model.base_model_prefix):]
              state_dict[_key] = _state
          else:
            state_dict[new_key] = _state
      if hasattr(model, "hack_pretrained_state_dict"):
        state_dict = model.hack_pretrained_state_dict(state_dict)

      # copy state_dict so _load_from_state_dict can modify it
      metadata = getattr(state_dict, "_metadata", None)
      state_dict = state_dict.copy()
      if metadata is not None:
        state_dict._metadata = metadata

      # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
      # so we need to apply the function recursively.
      def load(module, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            True,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        for name, child in module._modules.items():
          if child is not None:
            load(child, prefix + name + ".")

      # Make sure we are able to load base models as well as derived models (with heads)
      load(model)
      load = None

      if len(missing_keys) > 0:
        logger.info(
            "Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
      if len(unexpected_keys) > 0:
        logger.info("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
      if len(error_msgs) > 0:
        raise RuntimeError(
            "Error(s) in loading state_dict for {}:\n\t{}".format(
                model.__class__.__name__, "\n\t".join(error_msgs)))
    model.tie_weights(
    )  # make sure token embedding weights are still tied if needed

    # Set model in evaluation mode to deactivate DropOut modules by default
    model.eval()

    if output_loading_info:
      loading_info = {
          "missing_keys": missing_keys,
          "unexpected_keys": unexpected_keys,
          "error_msgs": error_msgs,
      }
      return model, loading_info

    if hasattr(config, "xla_device") and config.xla_device:
      import torch_xla.core.xla_model as xm

      model = xm.send_cpu_data_to_device(model, xm.xla_device())
      model.to(xm.xla_device())

    return model
