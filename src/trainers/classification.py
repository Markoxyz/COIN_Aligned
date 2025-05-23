import torch
from easydict import EasyDict as edict
from torchmetrics import F1Score, MetricCollection, Precision, Recall
from tqdm import tqdm

from src.datasets import get_dataloaders
from src.datasets.augmentations import get_transforms

# from src.datasets.tsm_synth_dataset import get_totalsegmentor_dataloaders
from src.models.classifier import ClassificationModel
from src.trainers.trainer import BaseTrainer
from src.utils.generic_utils import save_model
from flashtorch.utils import apply_transforms, load_image
from flashtorch.utils import apply_transforms, load_image
from flashtorch.saliency import Backprop

class ClassificationTrainer(BaseTrainer):
    def __init__(self, opt: edict, model: ClassificationModel, continue_path: str = None) -> None:
        super().__init__(opt, model, continue_path)

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.task = 'binary' if opt.model.n_classes == 1 else 'multiclass'
        
        self.apply_tanh_to_train = True if 'tanh' in self.opt.dataset.augs else False
        print(f'Apply tanh to train: {self.apply_tanh_to_train}')

        metric_list = [
            Precision(num_classes=opt.model.n_classes, task=self.task, average='macro'),
            Recall(num_classes=opt.model.n_classes, task=self.task, average='macro'),
            F1Score(num_classes=opt.model.n_classes, task=self.task, average='macro'),
        ]
        self.train_metrics = MetricCollection(metric_list).to(self.device)
        self.val_metrics = self.train_metrics.clone()
        
        self.tanh_f1_metric = F1Score(num_classes=opt.model.n_classes, task=self.task, average='macro').to(self.device) ###########

    def get_dataloaders(self) -> tuple:
        transforms = get_transforms(self.opt.dataset)
        return get_dataloaders(self.opt.dataset, transforms)
        # return get_totalsegmentor_dataloaders(transforms, self.opt.dataset)

    def restore_state(self):
        latest_ckpt = max(self.ckpt_dir.glob('*.pth'), key=lambda p: int(p.name.replace('.pth', '').split('_')[1]))
        load_ckpt = latest_ckpt if self.ckpt_name is None else (self.ckpt_dir / self.ckpt_name)
        state = torch.load(load_ckpt)
        self.batches_done = state['step']
        self.current_epoch = state['epoch']
        self.model.load_state_dict(state['model'], strict=True)
        self.model.optimizer.load_state_dict(state['optimizers'][0])
        self.logger.info(f"Restored checkpoint {latest_ckpt} ({state['date']})")

    def save_state(self, latest = False, best = False) -> str:
        if best:
            self.logger.info('Saving best checkpoint...')
            return save_model(self.opt, self.model, (self.model.optimizer,), self.batches_done, -2, self.ckpt_dir)
         
        if latest:
            self.logger.info('Saving latest checkpoint...')
            return save_model(self.opt, self.model, (self.model.optimizer,), self.batches_done, -1, self.ckpt_dir)

        return save_model(self.opt, self.model, (self.model.optimizer,), self.batches_done, self.current_epoch, self.ckpt_dir)

    def training_epoch(self, loader: torch.utils.data.DataLoader) -> None:
        self.model.train()
        self.train_metrics.reset()
        losses = []

        epoch_steps = self.opt.get('epoch_steps')
        epoch_length = epoch_steps or len(loader)
        avg_pos_to_neg_ratio = 0.0
        with tqdm(enumerate(loader), desc=f'Training epoch: {self.current_epoch}', leave=False, total=epoch_length) as prog:
            for i, batch in prog:
                if i == epoch_steps:
                    break
                batch = {k: batch[k].to(self.device) for k in {'image', 'masks', 'label'}}
                
                if self.apply_tanh_to_train:
                    # Apply tanh to input images
                    batch['image'] = torch.tanh(batch['image'])
                
                outs = self.model(batch, training=True, global_step=self.batches_done)
                
                avg_pos_to_neg_ratio += batch['label'].sum() / batch['label'].shape[0]
                self.batches_done = self.current_epoch * len(loader) + i

                self.train_metrics.update(outs['preds'], batch['label'])
                losses.append(outs['loss'])

                sample_step = self.batches_done % self.opt.sample_interval == 0
                if sample_step:
                    prog.set_postfix_str('training_loss={:.5f}'.format(outs['loss'].item()), refresh=True)
        self.logger.info('[Average positives/negatives ratio in batch: %f]' % round(avg_pos_to_neg_ratio.item() / epoch_length, 3))
        epoch_stats = {'loss': torch.mean(torch.tensor(losses)), **self.train_metrics.compute()}
        self.logger.log(epoch_stats, self.current_epoch, 'train')
        self.logger.info(
            '[Finished training epoch %d/%d] [Epoch loss: %f] [Epoch F1 score: %f]'
            % (self.current_epoch, self.opt.n_epochs, epoch_stats['loss'], epoch_stats[f'{self.task.title()}F1Score'])
        )
        return epoch_stats

    @torch.no_grad()
    def validation_epoch(self, loader: torch.utils.data.DataLoader) -> None:
        self.model.eval()
        self.val_metrics.reset()
        self.tanh_f1_metric.reset() #####
        losses = []
        num_pos = 0
        
        for _, batch in tqdm(enumerate(loader), desc=f'Validation epoch: {self.current_epoch}', leave=False, total=len(loader)):
            batch = {k: batch[k].to(self.device) for k in {'image', 'masks', 'label'}}
            # print('val', batch['label'].sum())
            num_pos += batch['label'].sum()
            outs = self.model(batch, training=False)
            
            # Apply tanh to input images
            tanh_images = torch.tanh(batch['image'])
            tanh_batch = batch.copy()
            tanh_batch['image'] = tanh_images

            # Forward pass
            outs2 = self.model(tanh_batch, training=False)
            preds = outs2['preds']

            predicted_labels = (preds > 0.5).long()

            # Update tanh-based F1 metric
            self.tanh_f1_metric.update(predicted_labels, batch['label'])
            
            self.val_metrics.update(outs['preds'], batch['label'])
            losses.append(outs['loss'])
        epoch_stats = {'loss': torch.mean(torch.tensor(losses)), **self.val_metrics.compute()}
        self.logger.log(epoch_stats, self.current_epoch, 'val')
        self.logger.info('[Number of positive samples in validation: %d]' % num_pos.item())
        self.logger.info(
            '[Finished validation epoch %d/%d] [Epoch loss: %f] [Epoch F1 score: %f]'
            % (self.current_epoch, self.opt.n_epochs, epoch_stats['loss'], epoch_stats[f'{self.task.title()}F1Score'])
        )
        
        tanh_f1_score = self.tanh_f1_metric.compute()
        self.logger.info(f'[Tanh input F1 score: {tanh_f1_score:.4f}]')
        
        return epoch_stats
