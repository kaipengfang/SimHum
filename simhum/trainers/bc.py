# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from simhum.trainers.base import BaseTrainer


class BehaviorCloning(BaseTrainer):
    def training_step(self, batch, global_step):
        (imgs, obs), actions, mask = batch
       
        imgs = {k: v.to(self.device_id) for k, v in imgs.items()}
        obs = {k: v.to(self.device_id) for k, v in obs.items()}
        actions, mask = [ar.to(self.device_id) for ar in (actions, mask)]
        
        ac_flat = actions.reshape((actions.shape[0], -1))
        mask_flat = mask.reshape((mask.shape[0], -1))
        loss = self.model(imgs, obs, ac_flat, mask_flat)
        if isinstance(loss, dict):
            self.log("robot_loss", global_step, loss['robot_loss'].item())
            self.log("human_robot_loss", global_step, loss['human_robot_loss'].item())
            self.log("human_hands_loss", global_step, loss['human_hands_loss'].item())
            self.log("bc_loss", global_step, loss['loss'].item())
            # print("robot_loss", loss['robot_loss'].item())
            # print("human_robot_loss", loss['human_robot_loss'].item())
            # print("human_hands_loss", loss['human_hands_loss'].item())
            # print("bc_loss", loss['loss'].item())
            loss = loss['loss']
        else :
            self.log("bc_loss", global_step, loss.item())
        if self.is_train:
            self.log("lr", global_step, self.lr)
        return loss
