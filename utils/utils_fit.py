import torch
from tqdm import tqdm
from utils.utils import get_lr

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# 训练一个epoch
def fit_one_epoch(model_train, model, loss_history, optimizer, criterion, epoch, epoch_step, gen, Epoch, anchors, cfg, cuda):
    total_r_loss = 0
    total_conf_loss = 0
    total_landmark_loss = 0
    total_c_loss = 0

    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            # 训练一个batch
            if iteration >= epoch_step:
                break
            images, targets = batch[0], batch[1]
            if len(images)==0:
                continue
            with torch.no_grad():
                if cuda:
                    images = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
                else:
                    images = torch.from_numpy(images).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
            #----------------------#
            #   清零梯度
            #----------------------#
            optimizer.zero_grad()
            #----------------------#
            #   前向传播
            #----------------------#
            out = model_train(images)
            #----------------------#
            #   计算损失
            #----------------------#
            r_loss, conf_loss, landm_loss, c_loss = criterion(out, anchors, targets)
            loss = cfg['loc_weight'] * r_loss + conf_loss + landm_loss + c_loss

            loss.backward()
            optimizer.step()
            
            total_conf_loss += conf_loss.item()
            total_r_loss += cfg['loc_weight'] * r_loss.item()
            total_landmark_loss += landm_loss.item()
            total_c_loss += c_loss.item()
            
            pbar.set_postfix(**{'Conf Loss'         : total_conf_loss / (iteration + 1),
                                'Regression Loss'   : total_r_loss / (iteration + 1), 
                                'LandMark Loss'     : total_landmark_loss / (iteration + 1),
                                'Class Loss'        : total_c_loss / (iteration + 1),
                                'lr'                : get_lr(optimizer)})
            pbar.update(1)

    print('Saving state, iter:', str(epoch+1))
    torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f.pth'%((epoch+1),(total_conf_loss + total_r_loss + total_landmark_loss + total_c_loss)/(epoch_step+1)))
    loss_history.append_loss((total_conf_loss + total_r_loss + total_landmark_loss)/(epoch_step+1))
