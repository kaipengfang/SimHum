# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, model, restore_path):
        super().__init__()
        self._model = model
        if restore_path:
            # model restoration display
            print()
            print("\033[1;33m💾 Model Restoration:\033[0m")
            print(f"   📂 Loading weights from: \033[1;32m{restore_path}\033[0m")
            print()
            state_dict = torch.load(restore_path, map_location="cpu")
            state_dict = (
                state_dict["features"]
                if "features" in state_dict
                else state_dict["model"]
            )
            self.load_state_dict(state_dict)

    @property
    def embed_dim(self):
        raise NotImplementedError
