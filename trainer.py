from pymic.loss.seg.deep_sup import match_prediction_and_gt_shape
import copy
from torch import nn
from torch.nn import CrossEntropyLoss
from monai.losses import DiceLoss, LogHausdorffDTLoss, FocalLoss, DiceCELoss, DiceFocalLoss, TverskyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
import datetime
import time
from torch.optim import Adam
from monai.networks.utils import one_hot
import torch
import numpy as np
from torch.cuda import memory_allocated
from metric import get_classwise_dice, get_multi_class_metric, dc
from scipy.ndimage import zoom
from tensorboardX import SummaryWriter
from os.path import join
from pymic.util.ramps import get_rampup_ratio
from torch.nn import functional as F
import gc

loss_dict = {
    "dc": DiceLoss(softmax=True),
    "focal": FocalLoss(),
    "ce": CrossEntropyLoss(),
    "bd": LogHausdorffDTLoss(softmax=True)
}


def sharpening(P, T=0.1):
    T = 1.0 / T
    P_sharpen = P**T / (P**T + (1 - P)**T)
    return P_sharpen


class BaseTrainer:

    def __init__(self, config, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.device = self.config["Training"]["device"]
        self.logger = kwargs["logger"]
        self.additional_param = kwargs
        self.model = self.build_model()
        self.max_val_scalar = None
        self.criterion = self.build_criterion()
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_sched(self.optimizer)
        self.glob_it = 0
        self.best_model_wts = None
        self.train_iter = None
        self.summ_writer = SummaryWriter(
            join(config["Training"]["output_dir"], "tensorboard"))

    def build_model(self):
        print(f"before creating model: {memory_allocated()}")
        from model import Unet
        from model import initialize_weights
        model = Unet(self.config["Network"]).to(self.device)
        model.apply(lambda param: initialize_weights(param, 1))
        print(f"after creating model: {memory_allocated()}")
        return model

    def build_optimizer(self):
        return Adam(self.model.parameters(),
                    lr=self.config["Training"]["lr"],
                    weight_decay=self.config["Training"]["weight_decay"])

    def build_criterion(self):
        # return LogHausdorffDTLoss(softmax=True)
        return DiceLoss(softmax=True)

    def build_sched(self, optimizer):
        return ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=self.config["Training"]["lr_gamma"],
            patience=self.config["Training"]["ReduceLROnPlateau_patience"])

    def write_scalars(self, train_scalars, valid_scalars, lr_value, glob_it):
        loss_scalar = {
            'train': train_scalars['loss'],
            'valid': valid_scalars['loss']
        }
        dice_scalar = {
            'train': train_scalars['avg_fg_dice'],
            'valid': valid_scalars['avg_fg_dice']
        }
        self.summ_writer.add_scalars('loss', loss_scalar, glob_it)
        self.summ_writer.add_scalars('dice', dice_scalar, glob_it)
        self.summ_writer.add_scalars('lr', {"lr": lr_value}, glob_it)
        class_num = self.config['Network']['class_num']
        for c in range(class_num):
            cls_dice_scalar = {
                'train': train_scalars['class_dice'][c],
                'valid': valid_scalars['class_dice'][c]
            }
            self.summ_writer.add_scalars(f'class_{c}_dice', cls_dice_scalar,
                                         glob_it)

        train_dice = "[" + ' '.join("{0:.4f}".format(x)
                                    for x in train_scalars['class_dice']) + "]"
        self.logger.info(
            f"train loss {train_scalars['loss']:.4f}, avg foreground dice {train_scalars['avg_fg_dice']:.4f} {train_dice}"
        )

        valid_dice = "[" + ' '.join("{0:.4f}".format(x)
                                    for x in valid_scalars['class_dice']) + "]"
        self.logger.info(
            f"valid loss {valid_scalars['loss']:.4f}, avg foreground dice {valid_scalars['avg_fg_dice']:.4f} {valid_dice}"
        )

    def batch_forward(self, img, mask, to_onehot_y=False):
        output, mul_pred = self.model(img)

        if len(output) == 1:
            output = output[0]
        elif len(output) > 1:
            output = torch.stack(output).mean(0)
        else:
            raise RuntimeError

        if to_onehot_y:
            mask = one_hot(mask, self.config["Network"]["class_num"])
        loss = self.criterion(output, mask)

        if self.config["Network"][
                "deep_supervision"] != "normal" or not self.model.training:
            return output, loss

        # for deep supervision
        deepsup_loss = 0
        for chunked_pred in mul_pred:
            pred, mask = match_prediction_and_gt_shape(chunked_pred, mask, 0)
            deepsup_loss += self.criterion(pred, mask)

        deepsup_loss = deepsup_loss / len(mul_pred)
        loss += deepsup_loss

        return output, loss

    def training(self, dataloader):
        trainloader = dataloader["labeled"]
        iter_valid = self.config["Training"]["iter_valid"]
        class_num = self.config["Network"]["class_num"]
        self.model.train()
        train_loss = 0
        train_dice_list = []

        for it in range(iter_valid):
            try:
                img, mask = next(self.train_iter)
            except StopIteration:
                self.train_iter = iter(trainloader)
                img, mask = next(self.train_iter)
            print(f"it: {it}")
            img, mask = img.to(self.device), mask.to(self.device)
            print(f"load img: {memory_allocated()}")
            onehot_mask = one_hot(mask, class_num)

            self.optimizer.zero_grad()

            output, loss = self.batch_forward(img, onehot_mask)
            print(f"before backward: {memory_allocated()}")
            loss.backward()
            print(f"after backward: {memory_allocated()}")
            self.optimizer.step()
            train_loss += loss.item()
            preds = output.argmax(1).unsqueeze(1)
            bin_mask = one_hot(preds, class_num)
            soft_y = onehot_mask.permute(0, 2, 3, 1).reshape((-1, class_num))
            predict = bin_mask.permute(0, 2, 3, 1).reshape((-1, class_num))
            dice_tesnsor = get_classwise_dice(predict, soft_y).cpu().numpy()
            train_dice_list.append(dice_tesnsor)
        train_avg_loss = train_loss / iter_valid
        train_cls_dice = np.asarray(train_dice_list).mean(axis=0)
        train_avg_dice = train_cls_dice[1:].mean()
        train_scalers = {
            'loss': train_avg_loss,
            'avg_fg_dice': train_avg_dice,
            'class_dice': train_cls_dice
        }
        return train_scalers

    @torch.no_grad()
    def validation2d(self, dataloader):
        self.model.eval()

        valid_loader = dataloader["valid"]
        class_num = self.config["Network"]["class_num"]

        dice_his, valid_loss = [], []
        for _, (img, mask) in enumerate(valid_loader):
            batch_slices, mask_slices = img.to(self.device), mask.to(
                self.device)
            output, loss = self.batch_forward(batch_slices,
                                              mask_slices,
                                              to_onehot_y=True)
            batch_pred_mask = output.argmax(axis=1)

            dice = dc(batch_pred_mask,
                      mask_slices.squeeze(1),
                      class_num,
                      include_backgroud=True)
            valid_loss.append(loss.item())
            dice_his.append(dice.cpu().numpy().tolist())

        valid_avg_loss = np.asarray(valid_loss).mean()

        valid_cls_dice = np.asarray(dice_his).mean(axis=0)
        valid_avg_dice = valid_cls_dice[1:].mean()

        valid_scalers = {
            'loss': valid_avg_loss,
            'avg_fg_dice': valid_avg_dice,
            'class_dice': valid_cls_dice
        }
        return valid_scalers

    @torch.no_grad()
    def validation(self, dataloader):
        self.model.eval()

        valid_loader = dataloader["valid"]
        class_num = self.config["Network"]["class_num"]
        batch_size = self.config["Dataset"]["batch_size"]
        input_size = self.config["Dataset"]["input_size"]

        dice_his, valid_loss = [], []

        for idx, (img, mask) in enumerate(valid_loader):
            img, mask = img[0], mask[0]
            h, w = img.shape[-2], img.shape[-1]
            batch_pred = []
            volume_loss = 0
            zoomed_img = zoom(img, (1, 1, input_size / h, input_size / w),
                              order=1,
                              mode='nearest')
            zoomed_mask = zoom(mask, (1, 1, input_size / h, input_size / w),
                               order=0,
                               mode='nearest')

            for batch in range(0, img.shape[0], batch_size):
                last = batch + batch_size
                last = last if last < img.shape[0] else None
                batch_slices, mask_slices = zoomed_img[
                    batch:last], zoomed_mask[batch:last]

                batch_slices = torch.tensor(batch_slices, device=self.device)
                mask_slices = torch.tensor(mask_slices, device=self.device)

                output, loss = self.batch_forward(batch_slices,
                                                  mask_slices,
                                                  to_onehot_y=True)
                volume_loss += loss.item()
                batch_pred.append(output.cpu().numpy())

            pred_volume = np.concatenate(batch_pred)
            pred_volume = zoom(pred_volume,
                               (1, 1, h / input_size, w / input_size),
                               order=1,
                               mode='nearest')
            del batch_pred
            batch_pred_mask = pred_volume.argmax(axis=1)
            dice, _, _ = get_multi_class_metric(
                batch_pred_mask,
                np.asarray(mask.squeeze(1)),
                class_num,
                include_backgroud=True,
            )
            valid_loss.append(volume_loss)
            dice_his.append(dice)
        valid_avg_loss = np.asarray(valid_loss).mean()

        valid_cls_dice = np.asarray(dice_his).mean(axis=0)
        valid_avg_dice = valid_cls_dice[1:].mean()

        valid_scalers = {
            'loss': valid_avg_loss,
            'avg_fg_dice': valid_avg_dice,
            'class_dice': valid_cls_dice
        }
        return valid_scalers

    def finish(self):
        self.summ_writer.flush()
        self.summ_writer.close()

    def train(self, dataloader):
        iter_max = self.config["Training"]["iter_max"]
        iter_valid = self.config["Training"]["iter_valid"]
        early_stop = self.config["Training"]["early_stop_patience"]

        self.max_val_scalar = None
        max_performance_it = 0

        self.train_iter = iter(dataloader["labeled"])
        start_it = self.glob_it
        for it in range(0, iter_max, iter_valid):

            lr_value = self.optimizer.param_groups[0]['lr']
            t0 = time.time()
            train_scalars = self.training(dataloader)
            t1 = time.time()
            if self.config["Dataset"].get("name", "ACDC") == "ISIC":
                valid_scalars = self.validation2d(dataloader)
            else:
                valid_scalars = self.validation(dataloader)
            t2 = time.time()

            self.scheduler.step(valid_scalars["avg_fg_dice"])

            self.glob_it += iter_valid

            self.logger.info(
                f"\n{str(datetime.datetime.now())[:-7]} iteration {self.glob_it}"
            )
            self.logger.info(f"learning rate {lr_value}")
            self.logger.info(
                f"training/validation time:{t1 - t0:.4f}/{t2 - t1:.4f}")

            self.write_scalars(train_scalars, valid_scalars, lr_value,
                               self.glob_it)

            if self.max_val_scalar is None or valid_scalars[
                    "avg_fg_dice"] > self.max_val_scalar["avg_fg_dice"]:
                max_performance_it = self.glob_it
                self.max_val_scalar = valid_scalars
                self.best_model_wts = {
                    'model_state_dict':
                    copy.deepcopy(self.model.state_dict()),
                    'optimizer_state_dict':
                    copy.deepcopy(self.optimizer.state_dict()),
                }

            if self.glob_it - start_it - max_performance_it > early_stop:
                self.logger.info("The training is early stopped")
                break
            # torch.cuda.empty_cache()
            # gc.collect()
        # best
        save_path = f"{self.config['Training']['checkpoint_dir']}/best{self.max_val_scalar['avg_fg_dice']:.4f}.pt"
        torch.save(self.best_model_wts, save_path)

        # latest
        save_path = f"{self.config['Training']['checkpoint_dir']}/g{self.glob_it}_l{self.glob_it - start_it}_latest{valid_scalars['avg_fg_dice']:.4f}.pt"
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(save_dict, save_path)
        return self.max_val_scalar


class URPCTrainer(BaseTrainer):

    def train(self, dataloader):
        self.unlab_it = iter(dataloader["unlabeled"])
        return super().train(dataloader)

    def write_scalars(self, train_scalars, valid_scalars, lr_value, glob_it):
        loss_scalar = {
            'train': train_scalars['loss'],
            'valid': valid_scalars['loss']
        }
        loss_sup_scalar = {'train': train_scalars['loss_sup']}
        loss_upsup_scalar = {'train': train_scalars['loss_reg']}
        dice_scalar = {
            'train': train_scalars['avg_fg_dice'],
            'valid': valid_scalars['avg_fg_dice']
        }
        self.summ_writer.add_scalars('loss', loss_scalar, glob_it)
        self.summ_writer.add_scalars('loss_sup', loss_sup_scalar, glob_it)
        self.summ_writer.add_scalars('loss_reg', loss_upsup_scalar, glob_it)
        self.summ_writer.add_scalars('lr', {"lr": lr_value}, glob_it)
        self.summ_writer.add_scalars('dice', dice_scalar, glob_it)
        class_num = self.config['Network']['class_num']
        for c in range(class_num):
            cls_dice_scalar = {
                'train': train_scalars['class_dice'][c],
                'valid': valid_scalars['class_dice'][c]
            }
            self.summ_writer.add_scalars('class_{0:}_dice'.format(c),
                                         cls_dice_scalar, glob_it)
        self.logger.info('train loss {0:.4f}, avg foreground dice {1:.4f} '.format(
            train_scalars['loss'], train_scalars['avg_fg_dice']) + "[" + \
                         ' '.join("{0:.4f}".format(x) for x in train_scalars['class_dice']) + "]")
        self.logger.info('valid loss {0:.4f}, avg foreground dice {1:.4f} '.format(
            valid_scalars['loss'], valid_scalars['avg_fg_dice']) + "[" + \
                         ' '.join("{0:.4f}".format(x) for x in valid_scalars['class_dice']) + "]")

    def training(self, dataloader):
        trainloader, unlbloader = dataloader["labeled"], dataloader[
            "unlabeled"]
        iter_valid = self.config["Training"]["iter_valid"]
        rampup_mode = self.config["Training"]["rampup_mode"]
        class_num = self.config["Network"]["class_num"]
        ramp_start = self.config["Training"]["rampup_start"]
        ramp_end = self.config["Training"]["rampup_end"]
        regularize_w = self.config["Training"]["regularize_w"]
        self.model.train()
        train_loss = 0
        train_loss_sup = 0
        train_loss_reg = 0
        train_dice_list = []
        for it in range(iter_valid):
            try:
                imglb, masklb = next(self.train_iter)
            except StopIteration:
                self.train_iter = iter(trainloader)
                imglb, masklb = next(self.train_iter)

            try:
                imgub, _ = next(self.unlab_it)
            except StopIteration:
                self.unlab_it = iter(unlbloader)
                imgub, _ = next(self.unlab_it)

            imglb_l = len(imglb)
            img = torch.cat([imglb, imgub], dim=0)
            img, masklb = img.to(self.device), masklb.to(self.device)
            onehot_mask = one_hot(masklb, class_num)

            self.optimizer.zero_grad()

            output, mul_pred = self.model(img)

            output = output[0]
            labeled_output, unlabeled_output = output[:imglb_l], output[
                imglb_l:]

            for i in range(0, len(mul_pred)):
                mul_pred[i] = nn.functional.interpolate(
                    mul_pred[i], output[0].shape[1:])
            labeled_aux, unlabeled_aux = zip(*((aux[:imglb_l], aux[imglb_l:])
                                               for aux in mul_pred))

            # supervised loss and deep supervision
            aux_preds = torch.cat([labeled_output, *labeled_aux])
            labeled_mask = onehot_mask.repeat(len(labeled_aux) + 1, 1, 1, 1)
            loss_sup = self.criterion(aux_preds, labeled_mask)

            # Consistency loss
            stacked_unlabeled = torch.stack([unlabeled_output,
                                             *unlabeled_aux]).softmax(2)
            avg_pred = torch.mean(stacked_unlabeled, dim=0) * 0.99 + 0.005
            loss_reg = 0
            for aux in stacked_unlabeled:
                aux = aux * 0.99 + 0.005
                var = torch.sum(nn.functional.kl_div(aux.log(),
                                                     avg_pred,
                                                     reduction="none"),
                                dim=1,
                                keepdim=True)
                exp_var = torch.exp(-var)
                square_e = torch.square(avg_pred - aux)
                loss_i = torch.mean(square_e * exp_var) / \
                         (torch.mean(exp_var) + 1e-8) + torch.mean(var)
                loss_reg += loss_i
            loss_reg = loss_reg / len(stacked_unlabeled)

            alpha = get_rampup_ratio(
                self.glob_it, ramp_start, ramp_end,
                mode=rampup_mode) * regularize_w
            loss = loss_sup + alpha * loss_reg

            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            train_loss_sup = train_loss_sup + loss_sup.item()
            train_loss_reg = train_loss_reg + loss_reg.item()

            preds = output.detach()[:imglb_l].argmax(1, keepdims=True)
            bin_mask = one_hot(preds, class_num)
            soft_y = onehot_mask.permute(0, 2, 3, 1).reshape((-1, class_num))
            predict = bin_mask.permute(0, 2, 3, 1).reshape((-1, class_num))
            dice_tesnsor = get_classwise_dice(predict, soft_y).cpu().numpy()
            train_dice_list.append(dice_tesnsor)
        train_avg_loss = train_loss / iter_valid
        train_avg_loss_sup = train_loss_sup / iter_valid
        train_avg_loss_reg = train_loss_reg / iter_valid
        train_cls_dice = np.asarray(train_dice_list).mean(axis=0)
        train_avg_dice = train_cls_dice[1:].mean()
        train_scalers = {
            'loss': train_avg_loss,
            'loss_sup': train_avg_loss_sup,
            'loss_reg': train_avg_loss_reg,
            'avg_fg_dice': train_avg_dice,
            'class_dice': train_cls_dice
        }
        return train_scalers


class MUCTrainer(BaseTrainer):

    def __init__(self, config, **kwargs) -> None:
        assert isinstance(config["Network"]["norm_type"], list) or isinstance(
            config["Network"]["norm_type"], tuple), ""
        self.loss_name = config["Training"]["loss"]
        super().__init__(config, **kwargs)

    def build_criterion(self):
        return loss_dict[self.loss_name]

    def write_scalars(self, train_scalars, valid_scalars, lr_value, glob_it):
        loss_scalar = {
            'train': train_scalars['loss'],
            'valid': valid_scalars['loss']
        }
        loss_sup_scalar = {'train': train_scalars['loss_sup']}
        loss_upsup_scalar = {'train': train_scalars['loss_reg']}
        dice_scalar = {
            'train': train_scalars['avg_fg_dice'],
            'valid': valid_scalars['avg_fg_dice']
        }
        self.summ_writer.add_scalars('loss', loss_scalar, glob_it)
        self.summ_writer.add_scalars('loss_sup', loss_sup_scalar, glob_it)
        self.summ_writer.add_scalars('loss_reg', loss_upsup_scalar, glob_it)
        self.summ_writer.add_scalars('lr', {"lr": lr_value}, glob_it)
        self.summ_writer.add_scalars('dice', dice_scalar, glob_it)
        class_num = self.config['Network']['class_num']
        for c in range(class_num):
            cls_dice_scalar = {
                'train': train_scalars['class_dice'][c],
                'valid': valid_scalars['class_dice'][c]
            }
            self.summ_writer.add_scalars('class_{0:}_dice'.format(c),
                                         cls_dice_scalar, glob_it)
        self.logger.info(
            'train loss {0:.4f}, avg foreground dice {1:.4f} '.format(
                train_scalars['loss'], train_scalars['avg_fg_dice']) + "[" +
            ' '.join("{0:.4f}".format(x)
                     for x in train_scalars['class_dice']) + "]")
        self.logger.info(
            'valid loss {0:.4f}, avg foreground dice {1:.4f} '.format(
                valid_scalars['loss'], valid_scalars['avg_fg_dice']) + "[" +
            ' '.join("{0:.4f}".format(x)
                     for x in valid_scalars['class_dice']) + "]")

    def train(self, dataloader):
        self.unlab_it = iter(dataloader["unlabeled"])
        return super().train(dataloader)

    def batch_forward(self, img, mask, to_onehot_y=False):

        if self.model.training:
            raise NotImplementedError
        else:
            output = torch.stack(self.model(img)[0], dim=0).mean(0)
            if to_onehot_y:
                mask = one_hot(mask, self.config["Network"]["class_num"])
            loss = self.criterion(output, mask)
            return output, loss

    def training(self, dataloader):
        trainloader, unlbloader = dataloader["labeled"], dataloader[
            "unlabeled"]
        iter_valid = self.config["Training"]["iter_valid"]
        class_num = self.config["Network"]["class_num"]
        ramp_start = self.config["Training"]["rampup_start"]
        rampup_mode = self.config["Training"]["rampup_mode"]
        ramp_end = self.config["Training"]["rampup_end"]
        regularize_w = self.config["Training"]["regularize_w"]
        self.model.train()
        train_loss = 0
        train_loss_sup = 0
        train_loss_reg = 0
        train_dice_list = []
        for it in range(iter_valid):
            try:
                imglb, masklb = next(self.train_iter)
            except StopIteration:
                self.train_iter = iter(trainloader)
                imglb, masklb = next(self.train_iter)

            try:
                imgub, _ = next(self.unlab_it)
            except StopIteration:
                self.unlab_it = iter(unlbloader)
                imgub, _ = next(self.unlab_it)

            imglb_l = len(imglb)
            img = torch.cat([imglb, imgub], dim=0)
            img, masklb = img.to(self.device), masklb.to(self.device)
            onehot_mask = one_hot(masklb, class_num)

            self.optimizer.zero_grad()

            output, mul_pred = self.model(img)

            output = torch.stack(output)
            labeled_output, unlabeled_output = output[:, :
                                                      imglb_l], output[:,
                                                                       imglb_l:]
            # dicece loss for labeled data

            loss_sup = 0
            for _, p in enumerate(labeled_output):
                loss_sup += self.criterion(p, onehot_mask)
            loss_sup /= labeled_output.shape[0]

            # Consistency loss
            unlabeled_output = unlabeled_output.softmax(2)
            avg_pred = torch.mean(unlabeled_output, dim=0) * 0.99 + 0.005
            loss_reg = 0
            for aux in unlabeled_output:
                aux = aux * 0.99 + 0.005
                var = torch.sum(nn.functional.kl_div(aux.log(),
                                                     avg_pred,
                                                     reduction="none"),
                                dim=1,
                                keepdim=True)
                exp_var = torch.exp(-var)
                square_e = torch.square(avg_pred - aux)
                loss_i = torch.mean(square_e * exp_var) / \
                         (torch.mean(exp_var) + 1e-8) + torch.mean(var)
                # loss_i = torch.mean(square_e * exp_var) / \
                #          (torch.mean(exp_var) + 1e-8)
                loss_reg += loss_i
            loss_reg = loss_reg / len(unlabeled_output)
            alpha = get_rampup_ratio(
                self.glob_it, ramp_start, ramp_end,
                mode=rampup_mode) * regularize_w
            loss = loss_sup + alpha * loss_reg
            # loss = loss_sup
            # loss_reg = torch.Tensor([0])
            #for deep supervision
            deepsup_loss = 0
            for chunked_pred in mul_pred:
                pred, mask = match_prediction_and_gt_shape(
                    chunked_pred, onehot_mask, 0)
                deepsup_loss += self.criterion(pred[:imglb_l].softmax(1), mask)

            deepsup_loss = deepsup_loss / len(mul_pred)
            loss += deepsup_loss

            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            train_loss_sup = train_loss_sup + loss_sup.item()
            train_loss_reg = train_loss_reg + loss_reg.item()

            preds = output.detach().mean(0)[:imglb_l].argmax(1, keepdims=True)
            bin_mask = one_hot(preds, class_num)
            soft_y = onehot_mask.permute(0, 2, 3, 1).reshape((-1, class_num))
            predict = bin_mask.permute(0, 2, 3, 1).reshape((-1, class_num))
            dice_tesnsor = get_classwise_dice(predict, soft_y).cpu().numpy()
            train_dice_list.append(dice_tesnsor)
        train_avg_loss = train_loss / iter_valid
        train_avg_loss_sup = train_loss_sup / iter_valid
        train_avg_loss_reg = train_loss_reg / iter_valid
        train_cls_dice = np.asarray(train_dice_list).mean(axis=0)
        train_avg_dice = train_cls_dice[1:].mean()
        train_scalers = {
            'loss': train_avg_loss,
            'loss_sup': train_avg_loss_sup,
            'loss_reg': train_avg_loss_reg,
            'avg_fg_dice': train_avg_dice,
            'class_dice': train_cls_dice
        }
        return train_scalers


class CPSTrainer(BaseTrainer):

    def build_model(self):
        from model import BiNet
        net1 = super().build_model()
        net2 = super().build_model()
        return BiNet(net1, net2)

    def train(self, dataloader):
        self.unlab_it = iter(dataloader["unlabeled"])
        return super().train(dataloader)

    def training(self, dataloader):
        trainloader, unlbloader = dataloader["labeled"], dataloader[
            "unlabeled"]
        iter_valid = self.config["Training"]["iter_valid"]
        class_num = self.config["Network"]["class_num"]
        ramp_start = self.config["Training"]["rampup_start"]
        rampup_mode = self.config["Training"]["rampup_mode"]
        ramp_end = self.config["Training"]["rampup_end"]
        regularize_w = self.config["Training"]["regularize_w"]
        self.model.train()
        train_loss = 0
        train_loss_sup1, train_loss_pseudo_sup1 = 0, 0
        train_loss_sup2, train_loss_pseudo_sup2 = 0, 0
        train_dice_list = []
        for it in range(iter_valid):
            try:
                imglb, masklb = next(self.train_iter)
            except StopIteration:
                self.train_iter = iter(trainloader)
                imglb, masklb = next(self.train_iter)

            try:
                imgub, _ = next(self.unlab_it)
            except StopIteration:
                self.unlab_it = iter(unlbloader)
                imgub, _ = next(self.unlab_it)

            imglb_l = len(imglb)
            img = torch.cat([imglb, imgub], dim=0)
            img, masklb = img.to(self.device), masklb.to(self.device)
            onehot_mask = one_hot(masklb, class_num)

            self.optimizer.zero_grad()

            output1, output2 = self.model(img)
            output1_soft, output2_soft = output1, output2
            self.optimizer.zero_grad()

            loss_sup1 = self.criterion(output1_soft[:imglb_l], onehot_mask)
            loss_sup2 = self.criterion(output2_soft[:imglb_l], onehot_mask)

            net1_pseudo_label, net2_pseudo_label = one_hot(
                output1[imglb_l:].detach().argmax(1, keepdims=True),
                class_num), one_hot(
                    output2[imglb_l:].detach().argmax(1, keepdims=True),
                    class_num)

            #for cross pseudo supervision
            pse_sup1 = self.criterion(output1_soft[imglb_l:],
                                      net2_pseudo_label)
            pse_sup2 = self.criterion(output2_soft[imglb_l:],
                                      net1_pseudo_label)

            rampup_ratio = get_rampup_ratio(self.glob_it, ramp_start, ramp_end,
                                            rampup_mode)
            regular_w = regularize_w * rampup_ratio

            model1_loss = loss_sup1 + regular_w * pse_sup1
            model2_loss = loss_sup2 + regular_w * pse_sup2

            loss = model1_loss + model2_loss

            loss.backward()
            self.optimizer.step()

            train_loss = train_loss + loss.item()
            train_loss_sup1 = train_loss_sup1 + loss_sup1.item()
            train_loss_sup2 = train_loss_sup2 + loss_sup2.item()
            train_loss_pseudo_sup1 = train_loss_pseudo_sup1 + pse_sup1.item()
            train_loss_pseudo_sup2 = train_loss_pseudo_sup2 + pse_sup2.item()

            preds = output1_soft.detach()[:imglb_l].argmax(1, keepdims=True)
            bin_mask = one_hot(preds, class_num)
            soft_y = onehot_mask.permute(0, 2, 3, 1).reshape((-1, class_num))
            predict = bin_mask.permute(0, 2, 3, 1).reshape((-1, class_num))
            dice_tesnsor = get_classwise_dice(predict, soft_y).cpu().numpy()
            train_dice_list.append(dice_tesnsor)

        train_avg_loss = train_loss / iter_valid
        train_avg_loss_sup1 = train_loss_sup1 / iter_valid
        train_avg_loss_sup2 = train_loss_sup2 / iter_valid
        train_avg_loss_pse_sup1 = train_loss_pseudo_sup1 / iter_valid
        train_avg_loss_pse_sup2 = train_loss_pseudo_sup2 / iter_valid
        train_cls_dice = np.asarray(train_dice_list).mean(axis=0)
        train_avg_dice = train_cls_dice[1:].mean()

        train_scalers = {
            'loss': train_avg_loss,
            'loss_sup1': train_avg_loss_sup1,
            'loss_sup2': train_avg_loss_sup2,
            'loss_pse_sup1': train_avg_loss_pse_sup1,
            'loss_pse_sup2': train_avg_loss_pse_sup2,
            'regular_w': regular_w,
            'avg_fg_dice': train_avg_dice,
            'class_dice': train_cls_dice
        }
        return train_scalers

    def write_scalars(self, train_scalars, valid_scalars, lr_value, glob_it):
        loss_scalar = {
            'train': train_scalars['loss'],
            'valid': valid_scalars['loss']
        }
        loss_sup_scalar = {
            'net1': train_scalars['loss_sup1'],
            'net2': train_scalars['loss_sup2']
        }
        loss_pse_sup_scalar = {
            'net1': train_scalars['loss_pse_sup1'],
            'net2': train_scalars['loss_pse_sup2']
        }
        dice_scalar = {
            'train': train_scalars['avg_fg_dice'],
            'valid': valid_scalars['avg_fg_dice']
        }
        self.summ_writer.add_scalars('loss', loss_scalar, glob_it)
        self.summ_writer.add_scalars('loss_sup', loss_sup_scalar, glob_it)
        self.summ_writer.add_scalars('loss_pseudo_sup', loss_pse_sup_scalar,
                                     glob_it)
        self.summ_writer.add_scalars('regular_w',
                                     {'regular_w': train_scalars['regular_w']},
                                     glob_it)
        self.summ_writer.add_scalars('lr', {"lr": lr_value}, glob_it)
        self.summ_writer.add_scalars('dice', dice_scalar, glob_it)
        class_num = self.config['Network']['class_num']
        for c in range(class_num):
            cls_dice_scalar = {'train':train_scalars['class_dice'][c], \
                'valid':valid_scalars['class_dice'][c]}
            self.summ_writer.add_scalars('class_{0:}_dice'.format(c),
                                         cls_dice_scalar, glob_it)

        self.logger.info('train loss {0:.4f}, avg dice {1:.4f} '.format(
            train_scalars['loss'], train_scalars['avg_fg_dice']) + "[" + \
            ' '.join("{0:.4f}".format(x) for x in train_scalars['class_dice']) + "]")
        self.logger.info('valid loss {0:.4f}, avg dice {1:.4f} '.format(
            valid_scalars['loss'], valid_scalars['avg_fg_dice']) + "[" + \
            ' '.join("{0:.4f}".format(x) for x in valid_scalars['class_dice']) + "]")

    def batch_forward(self, img, mask, to_onehot_y=False):
        if self.model.training:
            raise NotImplementedError
        else:
            output, _ = self.model(img)
            if to_onehot_y:
                mask = one_hot(mask, self.config["Network"]["class_num"])

            loss = self.criterion(output[0], mask)
            return output[0], loss


class MeanTeacherTrainer(BaseTrainer):

    def __init__(self, config, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.ema_param = config["Training"]["ema_decay"]

    def batch_forward(self, img, mask, to_onehot_y=False):
        if self.model.training:
            raise NotImplementedError
        else:
            output, _ = self.model(img)
            if to_onehot_y:
                mask = one_hot(mask, self.config["Network"]["class_num"])

            loss = self.criterion(output[0], mask)
            return output[0], loss

    def build_model(self):
        self.net_ema = super().build_model()
        return super().build_model()

    def ema_update(self):
        iter_valid = self.config["Training"]["iter_valid"]
        alpha = min(1 - 1 / (self.glob_it / iter_valid + 1), self.ema_param)
        for ema_param, param in zip(self.net_ema.parameters(),
                                    self.model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

    def training(self, dataloader):
        trainloader, unlbloader = dataloader["labeled"], dataloader[
            "unlabeled"]
        iter_valid = self.config["Training"]["iter_valid"]
        class_num = self.config["Network"]["class_num"]
        ramp_start = self.config["Training"]["rampup_start"]
        rampup_mode = self.config["Training"]["rampup_mode"]
        ramp_end = self.config["Training"]["rampup_end"]
        regularize_w = self.config["Training"]["regularize_w"]
        self.model.train()
        train_loss = 0
        train_loss_sup = 0
        train_loss_reg = 0
        train_dice_list = []
        for it in range(iter_valid):
            try:
                imglb, masklb = next(self.train_iter)
            except StopIteration:
                self.train_iter = iter(trainloader)
                imglb, masklb = next(self.train_iter)

            try:
                imgub, _ = next(self.unlab_it)
            except StopIteration:
                self.unlab_it = iter(unlbloader)
                imgub, _ = next(self.unlab_it)

            imglb_l = len(imglb)
            img = torch.cat([imglb, imgub], dim=0)
            img, masklb = img.to(self.device), masklb.to(self.device)
            onehot_mask = one_hot(masklb, class_num)

            self.optimizer.zero_grad()

            outputs, _ = self.model(img)
            outputs_soft = outputs[0]
            labed_out, unlabed_out = outputs_soft[:imglb_l], outputs_soft[
                imglb_l:]
            loss_sup = self.criterion(labed_out, onehot_mask)
            unlabed_soft = unlabed_out.softmax(1)

            noise = torch.clamp(torch.randn_like(imgub) * 0.1, -0.2, 0.2)
            inputs_ema = imgub + noise
            inputs_ema = inputs_ema.to(self.device)
            with torch.no_grad():
                outputs_ema, _ = self.net_ema(inputs_ema)
                p1_ema_soft = outputs_ema[0].softmax(1)

            rampup_ratio = get_rampup_ratio(self.glob_it, ramp_start, ramp_end,
                                            rampup_mode)
            regular_w = regularize_w * rampup_ratio

            loss_reg = F.mse_loss(unlabed_soft, p1_ema_soft)
            loss = loss_sup + regular_w * loss_reg

            loss.backward()
            self.optimizer.step()

            self.ema_update()
            train_loss = train_loss + loss.item()
            train_loss_sup = train_loss_sup + loss_sup.item()
            train_loss_reg = train_loss_reg + loss_reg.item()

            preds = labed_out.detach().argmax(1, keepdims=True)
            bin_mask = one_hot(preds, class_num)
            soft_y = onehot_mask.permute(0, 2, 3, 1).reshape((-1, class_num))
            predict = bin_mask.permute(0, 2, 3, 1).reshape((-1, class_num))
            dice_tesnsor = get_classwise_dice(predict, soft_y).cpu().numpy()
            train_dice_list.append(dice_tesnsor)

        train_avg_loss = train_loss / iter_valid
        train_avg_loss_sup = train_loss_sup / iter_valid
        train_avg_loss_reg = train_loss_reg / iter_valid
        train_cls_dice = np.asarray(train_dice_list).mean(axis=0)
        train_avg_dice = train_cls_dice[1:].mean()
        train_scalers = {
            'loss': train_avg_loss,
            'loss_sup': train_avg_loss_sup,
            'loss_reg': train_avg_loss_reg,
            'regular_w': regular_w,
            'avg_fg_dice': train_avg_dice,
            'class_dice': train_cls_dice
        }
        return train_scalers

    def train(self, dataloader):
        self.unlab_it = iter(dataloader["unlabeled"])
        return super().train(dataloader)


class PseudoMUCTrainer(BaseTrainer):

    def __init__(self, config, **kwargs) -> None:
        assert isinstance(config["Network"]["norm_type"], list) or isinstance(
            config["Network"]["norm_type"], tuple), ""
        super().__init__(config, **kwargs)

    def build_criterion(self):
        return DiceCELoss(softmax=True)

    def write_scalars(self, train_scalars, valid_scalars, lr_value, glob_it):
        loss_scalar = {
            'train': train_scalars['loss'],
            'valid': valid_scalars['loss']
        }
        loss_sup_scalar = {'train': train_scalars['loss_sup']}
        loss_upsup_scalar = {'train': train_scalars['loss_reg']}
        dice_scalar = {
            'train': train_scalars['avg_fg_dice'],
            'valid': valid_scalars['avg_fg_dice']
        }
        self.summ_writer.add_scalars('loss', loss_scalar, glob_it)
        self.summ_writer.add_scalars('loss_sup', loss_sup_scalar, glob_it)
        self.summ_writer.add_scalars('loss_reg', loss_upsup_scalar, glob_it)
        self.summ_writer.add_scalars('lr', {"lr": lr_value}, glob_it)
        self.summ_writer.add_scalars('dice', dice_scalar, glob_it)
        class_num = self.config['Network']['class_num']
        for c in range(class_num):
            cls_dice_scalar = {
                'train': train_scalars['class_dice'][c],
                'valid': valid_scalars['class_dice'][c]
            }
            self.summ_writer.add_scalars('class_{0:}_dice'.format(c),
                                         cls_dice_scalar, glob_it)
        self.logger.info(
            'train loss {0:.4f}, avg foreground dice {1:.4f} '.format(
                train_scalars['loss'], train_scalars['avg_fg_dice']) + "[" +
            ' '.join("{0:.4f}".format(x)
                     for x in train_scalars['class_dice']) + "]")
        self.logger.info(
            'valid loss {0:.4f}, avg foreground dice {1:.4f} '.format(
                valid_scalars['loss'], valid_scalars['avg_fg_dice']) + "[" +
            ' '.join("{0:.4f}".format(x)
                     for x in valid_scalars['class_dice']) + "]")

    def train(self, dataloader):
        self.unlab_it = iter(dataloader["unlabeled"])
        return super().train(dataloader)

    def batch_forward(self, img, mask, to_onehot_y=False):

        if self.model.training:
            raise NotImplementedError
        else:
            output = torch.stack(self.model(img)[0], dim=0).mean(0)
            if to_onehot_y:
                mask = one_hot(mask, self.config["Network"]["class_num"])
            loss = self.criterion(output, mask)
            return output, loss

    def training(self, dataloader):
        trainloader, unlbloader = dataloader["labeled"], dataloader[
            "unlabeled"]
        iter_valid = self.config["Training"]["iter_valid"]
        class_num = self.config["Network"]["class_num"]
        ramp_start = self.config["Training"]["rampup_start"]
        rampup_mode = self.config["Training"]["rampup_mode"]
        ramp_end = self.config["Training"]["rampup_end"]
        regularize_w = self.config["Training"]["regularize_w"]
        self.model.train()
        train_loss = 0
        train_loss_sup = 0
        train_loss_reg = 0
        train_dice_list = []
        for it in range(iter_valid):
            try:
                imglb, masklb = next(self.train_iter)
            except StopIteration:
                self.train_iter = iter(trainloader)
                imglb, masklb = next(self.train_iter)

            try:
                imgub, _ = next(self.unlab_it)
            except StopIteration:
                self.unlab_it = iter(unlbloader)
                imgub, _ = next(self.unlab_it)

            imglb_l = len(imglb)
            img = torch.cat([imglb, imgub], dim=0)
            img, masklb = img.to(self.device), masklb.to(self.device)
            onehot_mask = one_hot(masklb, class_num)

            self.optimizer.zero_grad()

            output, mul_pred = self.model(img)

            output = torch.stack(output)
            labeled_output, unlabeled_output = output[:, :
                                                      imglb_l], output[:,
                                                                       imglb_l:]
            # dicece loss for labeled data

            loss_sup = 0
            for _, p in enumerate(labeled_output):
                loss_sup += self.criterion(p, onehot_mask)
            loss_sup /= labeled_output.shape[0]

            unlabeled_output_soft = unlabeled_output.softmax(2)
            # Consistency loss
            avg_pred = torch.mean(unlabeled_output_soft, dim=0) * 0.99 + 0.005
            loss_reg = 0
            for aux in unlabeled_output_soft:
                aux = aux * 0.99 + 0.005
                var = torch.sum(nn.functional.kl_div(aux.log(),
                                                     avg_pred,
                                                     reduction="none"),
                                dim=1,
                                keepdim=True)
                exp_var = torch.exp(-var)
                square_e = torch.square(avg_pred - aux)
                loss_i = torch.mean(square_e * exp_var) / \
                         (torch.mean(exp_var) + 1e-8) + torch.mean(var)
                loss_reg += loss_i
            loss_reg = loss_reg / len(unlabeled_output_soft)

            #for pseudo supervision
            pseudo_label = unlabeled_output.argmax(2)
            pseudo_label = pseudo_label[torch.randperm(
                pseudo_label.shape[0])].detach()

            pseudo_label = torch.concat(list(pseudo_label))
            pseudo_label = one_hot(pseudo_label.unsqueeze(1), class_num)

            unlab_output_concat = torch.concat(list(unlabeled_output))
            loss_pseudo = self.criterion(unlab_output_concat, pseudo_label)

            alpha = get_rampup_ratio(
                self.glob_it, ramp_start, ramp_end,
                mode=rampup_mode) * regularize_w
            loss = loss_sup + alpha * loss_reg + alpha * loss_pseudo
            #for deep supervision
            deepsup_loss = 0
            for chunked_pred in mul_pred:
                pred, mask = match_prediction_and_gt_shape(
                    chunked_pred, onehot_mask, 0)
                deepsup_loss += self.criterion(pred[:imglb_l].softmax(1), mask)

            deepsup_loss = deepsup_loss / len(mul_pred)
            loss += deepsup_loss

            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            train_loss_sup = train_loss_sup + loss_sup.item()
            train_loss_reg = train_loss_reg + loss_reg.item()

            preds = output.detach().mean(0)[:imglb_l].argmax(1, keepdims=True)
            bin_mask = one_hot(preds, class_num)
            soft_y = onehot_mask.permute(0, 2, 3, 1).reshape((-1, class_num))
            predict = bin_mask.permute(0, 2, 3, 1).reshape((-1, class_num))
            dice_tesnsor = get_classwise_dice(predict, soft_y).cpu().numpy()
            train_dice_list.append(dice_tesnsor)
        train_avg_loss = train_loss / iter_valid
        train_avg_loss_sup = train_loss_sup / iter_valid
        train_avg_loss_reg = train_loss_reg / iter_valid
        train_cls_dice = np.asarray(train_dice_list).mean(axis=0)
        train_avg_dice = train_cls_dice[1:].mean()
        train_scalers = {
            'loss': train_avg_loss,
            'loss_sup': train_avg_loss_sup,
            'loss_reg': train_avg_loss_reg,
            'avg_fg_dice': train_avg_dice,
            'class_dice': train_cls_dice
        }
        return train_scalers


class MultiLossMUCTrainer(BaseTrainer):

    def __init__(self, config, **kwargs) -> None:
        assert isinstance(config["Network"]["norm_type"], list) or isinstance(
            config["Network"]["norm_type"], tuple), ""
        super().__init__(config, **kwargs)

    def build_criterion(self):
        self.dicece = DiceCELoss(softmax=True)
        return list(loss_dict.values())
        # return [
        #     DiceCELoss(softmax=True),
        #     CrossEntropyLoss(),
        #     DiceFocalLoss(softmax=True),
        #     DiceLoss(softmax=True),
        #     CrossEntropyLoss(),
        #     DiceFocalLoss(softmax=True),
        # ]

    def write_scalars(self, train_scalars, valid_scalars, lr_value, glob_it):
        loss_scalar = {
            'train': train_scalars['loss'],
            'valid': valid_scalars['loss']
        }
        loss_sup_scalar = {'train': train_scalars['loss_sup']}
        loss_upsup_scalar = {'train': train_scalars['loss_reg']}
        dice_scalar = {
            'train': train_scalars['avg_fg_dice'],
            'valid': valid_scalars['avg_fg_dice']
        }
        self.summ_writer.add_scalars('loss', loss_scalar, glob_it)
        self.summ_writer.add_scalars('loss_sup', loss_sup_scalar, glob_it)
        self.summ_writer.add_scalars('loss_reg', loss_upsup_scalar, glob_it)
        self.summ_writer.add_scalars('lr', {"lr": lr_value}, glob_it)
        self.summ_writer.add_scalars('dice', dice_scalar, glob_it)
        class_num = self.config['Network']['class_num']
        for c in range(class_num):
            cls_dice_scalar = {
                'train': train_scalars['class_dice'][c],
                'valid': valid_scalars['class_dice'][c]
            }
            self.summ_writer.add_scalars('class_{0:}_dice'.format(c),
                                         cls_dice_scalar, glob_it)
        self.logger.info(
            'train loss {0:.4f}, avg foreground dice {1:.4f} '.format(
                train_scalars['loss'], train_scalars['avg_fg_dice']) + "[" +
            ' '.join("{0:.4f}".format(x)
                     for x in train_scalars['class_dice']) + "]")
        self.logger.info(
            'valid loss {0:.4f}, avg foreground dice {1:.4f} '.format(
                valid_scalars['loss'], valid_scalars['avg_fg_dice']) + "[" +
            ' '.join("{0:.4f}".format(x)
                     for x in valid_scalars['class_dice']) + "]")

    def train(self, dataloader):
        self.unlab_it = iter(dataloader["unlabeled"])
        return super().train(dataloader)

    def batch_forward(self, img, mask, to_onehot_y=False):

        if self.model.training:
            raise NotImplementedError
        else:
            output = torch.stack(self.model(img)[0], dim=0).mean(0)
            if to_onehot_y:
                mask = one_hot(mask, self.config["Network"]["class_num"])
            loss = self.dicece(output, mask)
            return output, loss

    def training(self, dataloader):
        trainloader, unlbloader = dataloader["labeled"], dataloader[
            "unlabeled"]
        iter_valid = self.config["Training"]["iter_valid"]
        class_num = self.config["Network"]["class_num"]
        ramp_start = self.config["Training"]["rampup_start"]
        rampup_mode = self.config["Training"]["rampup_mode"]
        ramp_end = self.config["Training"]["rampup_end"]
        regularize_w = self.config["Training"]["regularize_w"]
        self.model.train()
        train_loss = 0
        train_loss_sup = 0
        train_loss_reg = 0
        train_dice_list = []
        for it in range(iter_valid):
            try:
                imglb, masklb = next(self.train_iter)
            except StopIteration:
                self.train_iter = iter(trainloader)
                imglb, masklb = next(self.train_iter)

            try:
                imgub, _ = next(self.unlab_it)
            except StopIteration:
                self.unlab_it = iter(unlbloader)
                imgub, _ = next(self.unlab_it)

            imglb_l = len(imglb)
            img = torch.cat([imglb, imgub], dim=0)
            img, masklb = img.to(self.device), masklb.to(self.device)
            onehot_mask = one_hot(masklb, class_num)

            self.optimizer.zero_grad()

            output, mul_pred = self.model(img)

            output = torch.stack(output)
            labeled_output, unlabeled_output = output[:, :
                                                      imglb_l], output[:,
                                                                       imglb_l:]
            unlabeled_output = unlabeled_output.softmax(1)
            # dicece loss for labeled data

            loss_sup = 0
            for i, p in enumerate(labeled_output):
                loss_sup += self.criterion[i](p, onehot_mask)
            loss_sup /= labeled_output.shape[0]

            # Consistency loss
            avg_pred = torch.mean(unlabeled_output, dim=0) * 0.99 + 0.005
            loss_reg = 0
            for aux in unlabeled_output:
                aux = aux * 0.99 + 0.005
                var = torch.sum(nn.functional.kl_div(aux.log(),
                                                     avg_pred,
                                                     reduction="none"),
                                dim=1,
                                keepdim=True)
                exp_var = torch.exp(-var)
                square_e = torch.square(avg_pred - aux)
                loss_i = torch.mean(square_e * exp_var) / \
                         (torch.mean(exp_var) + 1e-8) + torch.mean(var)
                loss_reg += loss_i
            loss_reg = loss_reg / len(unlabeled_output)
            alpha = get_rampup_ratio(
                self.glob_it, ramp_start, ramp_end,
                mode=rampup_mode) * regularize_w
            loss = loss_sup + alpha * loss_reg
            #for deep supervision
            deepsup_loss = 0
            for chunked_pred in mul_pred:
                pred, mask = match_prediction_and_gt_shape(
                    chunked_pred, onehot_mask, 0)
                deepsup_loss += self.dicece(pred[:imglb_l], mask)

            deepsup_loss = deepsup_loss / len(mul_pred)
            loss += deepsup_loss

            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            train_loss_sup = train_loss_sup + loss_sup.item()
            train_loss_reg = train_loss_reg + loss_reg.item()

            preds = output.detach().mean(0)[:imglb_l].argmax(1, keepdims=True)
            bin_mask = one_hot(preds, class_num)
            soft_y = onehot_mask.permute(0, 2, 3, 1).reshape((-1, class_num))
            predict = bin_mask.permute(0, 2, 3, 1).reshape((-1, class_num))
            dice_tesnsor = get_classwise_dice(predict, soft_y).cpu().numpy()
            train_dice_list.append(dice_tesnsor)
        train_avg_loss = train_loss / iter_valid
        train_avg_loss_sup = train_loss_sup / iter_valid
        train_avg_loss_reg = train_loss_reg / iter_valid
        train_cls_dice = np.asarray(train_dice_list).mean(axis=0)
        train_avg_dice = train_cls_dice[1:].mean()
        train_scalers = {
            'loss': train_avg_loss,
            'loss_sup': train_avg_loss_sup,
            'loss_reg': train_avg_loss_reg,
            'avg_fg_dice': train_avg_dice,
            'class_dice': train_cls_dice
        }
        return train_scalers


class MCNetTrainer(BaseTrainer):

    def build_model(self):
        from model import MCNet
        from model import initialize_weights

        model = MCNet(self.config["Network"]).to(self.device)

        model.apply(lambda param: initialize_weights(param, 1))
        print(f"total params: {model.count_params()}")
        return model

    def batch_forward(self, img, mask, to_onehot_y=False):

        if self.model.training:
            raise NotImplementedError
        else:
            output = self.model(img)
            if to_onehot_y:
                mask = one_hot(mask, self.config["Network"]["class_num"])
            loss = self.criterion(output, mask)
            return output, loss

    def training(self, dataloader):
        trainloader, unlbloader = dataloader["labeled"], dataloader[
            "unlabeled"]
        iter_valid = self.config["Training"]["iter_valid"]
        class_num = self.config["Network"]["class_num"]
        temperature = self.config["Network"]["temperature"]
        ramp_start = self.config["Training"]["rampup_start"]
        rampup_mode = self.config["Training"]["rampup_mode"]
        ramp_end = self.config["Training"]["rampup_end"]
        regularize_w = self.config["Training"]["regularize_w"]
        self.model.train()
        train_loss = 0
        train_loss_sup = 0
        train_loss_reg = 0
        train_dice_list = []
        for it in range(iter_valid):
            try:
                imglb, masklb = next(self.train_iter)
            except StopIteration:
                self.train_iter = iter(trainloader)
                imglb, masklb = next(self.train_iter)

            try:
                imgub, _ = next(self.unlab_it)
            except StopIteration:
                self.unlab_it = iter(unlbloader)
                imgub, _ = next(self.unlab_it)

            imglb_l = len(imglb)
            img = torch.cat([imglb, imgub], dim=0)
            img, masklb = img.to(self.device), masklb.to(self.device)
            onehot_mask = one_hot(masklb, class_num)

            self.optimizer.zero_grad()

            outputs = self.model(img)
            p = outputs[0][:imglb_l]

            num_outputs = len(outputs)

            p_ori = torch.zeros((num_outputs, ) + outputs[0].shape)
            y_psu = torch.zeros((num_outputs, ) + outputs[0].shape)

            loss_sup = 0
            for idx in range(num_outputs):
                p0i = outputs[idx][:imglb_l]
                loss_sup += self.criterion(p0i, onehot_mask)

                # get pseudo labels
                p_i = F.softmax(outputs[idx], dim=1)
                p_ori[idx] = p_i
                y_psu[idx] = sharpening(p_i, temperature)
            loss_reg = 0.0
            for i in range(num_outputs):
                for j in range(num_outputs):
                    if i != j:
                        loss_reg += F.mse_loss(p_ori[i],
                                               y_psu[j],
                                               reduction='mean')

            alpha = get_rampup_ratio(
                self.glob_it, ramp_start, ramp_end,
                mode=rampup_mode) * regularize_w
            loss = loss_sup + alpha * loss_reg
            loss.backward()
            self.optimizer.step()

            train_loss = train_loss + loss.item()
            train_loss_sup = train_loss_sup + loss_sup.item()
            train_loss_reg = train_loss_reg + loss_reg.item()

            preds = p.detach().argmax(1, keepdims=True)
            bin_mask = one_hot(preds, class_num)
            soft_y = onehot_mask.permute(0, 2, 3, 1).reshape((-1, class_num))
            predict = bin_mask.permute(0, 2, 3, 1).reshape((-1, class_num))
            dice_tesnsor = get_classwise_dice(predict, soft_y).cpu().numpy()
            train_dice_list.append(dice_tesnsor)
        train_avg_loss = train_loss / iter_valid
        train_avg_loss_sup = train_loss_sup / iter_valid
        train_avg_loss_reg = train_loss_reg / iter_valid
        train_cls_dice = np.asarray(train_dice_list).mean(axis=0)
        train_avg_dice = train_cls_dice[1:].mean()
        train_scalers = {
            'loss': train_avg_loss,
            'loss_sup': train_avg_loss_sup,
            'loss_reg': train_avg_loss_reg,
            'regular_w': alpha,
            'avg_fg_dice': train_avg_dice,
            'class_dice': train_cls_dice
        }
        return train_scalers

    def train(self, dataloader):
        self.unlab_it = iter(dataloader["unlabeled"])
        return super().train(dataloader)
