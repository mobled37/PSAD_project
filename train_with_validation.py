import torch


from torch import nn
from torch.utils.data import DataLoader
from custom_dataset_fixed3 import PSAD_Dataset
from resnet1d import ResNet1D
from torch.utils.tensorboard import SummaryWriter
import tqdm
import wandb

from torchmetrics import F1Score, Accuracy, Recall

def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader

def train_single_epoch(model, data_loader, loss_fn, optimiser, writer, global_step, device):
    model.train().to(device)
    prog_bar2 = tqdm.tqdm(desc=f'training in progress',
                          total=len(data_loader),
                          position=1,
                          leave=True)

    # f1 = F1Score().to(device)
    accuracy = Accuracy().to(device)
    # recall = Recall().to(device)
    acc_sum, loss_sum = 0, 0

    # for inputs, targets in data_loader:
    for idx, batch in enumerate(data_loader):
        inputs, targets = batch[0].to(device), batch[1].squeeze(1).to(device)

        # calculate loss
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        # F1 Score
        predictions = predictions.detach().cpu()
        pred_binary = predictions > .5
        targets = targets.detach().cpu()

        target_long = targets.type(torch.LongTensor).to(device)
        predictions_long = pred_binary.type(torch.LongTensor).to(device)

        target_long = torch.reshape(target_long, (-1,)).to(device)
        predictions_long = torch.reshape(predictions_long, (-1,)).to(device)

        # score
        # f1_sc = f1(predictions_long, target_long)
        accuracy_sc = accuracy(predictions_long, target_long).to(device)
        # recall_sc = recall(predictions_long, target_long).to(device)
        # biaccuracy_sc = binary_acc((predictions_long, target_long)).to(device)
        acc_sum += accuracy_sc
        # f1_sum += f1_sc
        # rec_sum += recall_sc

        # sum loss
        loss_sum += loss.item()

        # backpropagate loss and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        # logger
        prog_bar2.update()
        global_step += 1

    metrics = {
        'accuracy': acc_sum / len(data_loader),
        'loss': loss_sum / len(data_loader),
        # 'recall': rec_sum / len(data_loader),
        # 'F1 score': f1_sum / len(data_loader)
        # 'bi_acc': biaccuracy_sc
    }

    print(f"Loss: {loss_sum / len(data_loader)}")

    writer.add_scalar('Loss/train', loss.item(), global_step)
    return metrics, global_step

def train(model, data_loader, loss_fn, optimiser, device, epochs):
    writer = SummaryWriter()
    global_step = 0
    for i in range(epochs):
        print(f"Epoch {i + 1}")
        metrics, step = train_single_epoch(model, data_loader, loss_fn, optimiser,
                           device=device, writer=writer, global_step=global_step)
        global_step = step
        wandb.log(metrics, step=global_step)
        print("-------------------")
    print('Finished Training')

def validation_single_epoch(model, data_loader, loss_fn, writer, global_step, device):
    """
    Is that model.eval() needs .to(device) ?
    """
    model.eval().to(device)
    prog_bar2 = tqdm.tqdm(desc=f'validation in progress',
                          total=len(data_loader),
                          position=1,
                          leave=True)

    # f1 = F1Score().to(device)
    accuracy = Accuracy().to(device)
    # recall = Recall().to(device)
    acc_sum, loss_sum = 0, 0

    # for inputs, targets in data_loader:
    for idx, batch in enumerate(data_loader):
        inputs, targets = batch[0].to(device), batch[1].squeeze(1).to(device)

        """
        Score section
        I don`t touch here because validation needs same logics
        """

        # calculate loss
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        # detach cpu cause we work on gpu
        predictions = predictions.detach().cpu()
        targets = targets.detach().cpu()

        # set prediction to binary
        pred_binary = predictions > .5

        target_long = targets.type(torch.LongTensor).to(device)
        predictions_long = pred_binary.type(torch.LongTensor).to(device)

        target_long = torch.reshape(target_long, (-1,)).to(device)
        predictions_long = torch.reshape(predictions_long, (-1,)).to(device)

        # score
        # f1_sc = f1(predictions_long, target_long)
        accuracy_sc = accuracy(predictions_long, target_long).to(device)
        # recall_sc = recall(predictions_long, target_long).to(device)

        # for mean the score
        acc_sum += accuracy_sc
        # f1_sum += f1_sc
        # rec_sum += recall_sc

        # sum loss
        loss_sum += loss.item()

        """
        I just learn validation process don`t need backpropagation
        """
        # backpropagate loss and update weights
        # loss.backward()

        # logger
        prog_bar2.update()
        global_step += 1

    metrics = {
        'val_accuracy': acc_sum / len(data_loader),
        'val_loss': loss_sum / len(data_loader),
        # 'val_recall': rec_sum / len(data_loader),
        # 'val_F1 score': f1_sum / len(data_loader)
    }

    print(f"Loss: {loss_sum / len(data_loader)}")
    writer.add_scalar('Loss/train', loss.item(), global_step)
    return metrics, global_step

def validation(model, data_loader, loss_fn, device, epochs):
    """
    Just delete optimiser, but IDK this is right
    And I think we just delete that SummaryWriter() because we don`t use torchsummary
    """
    writer = SummaryWriter()
    global_step = 0
    for i in range(epochs):
        print(f"Epoch {i + 1}")
        metrics, step = validation_single_epoch(model, data_loader, loss_fn,
                           device=device, writer=writer, global_step=global_step)
        global_step = step
        wandb.log(metrics, step=global_step)
        print("-------------------")
    print('Finished Validation')

if __name__ == "__main__":

    '''
        in test mode change :
        BATCH_SIZE, EPOCHS, scheduler milestones

        check cpu system ram 
        n_blocks

        check gpu ram
        BATCH_SIZE, n_blocks 

        I WANT TO KNOW n_blocks MEAN!!!!

        '''

    # Hyperparameter
    BATCH_SIZE = 128
    EPOCHS = 50
    LEARNING_RATE = 0.001

    FILENAME_DIR = '/content/drive/MyDrive/PSAD/sample_metadata/metadata.json'
    AUDIO_DIR = '/content/drive/MyDrive/PSAD/sample_save'

    # FILENAME_DIR = '/Users/valleotb/Desktop/Valleotb/sample_metadata/metadata.json'
    # AUDIO_DIR = '/Users/valleotb/Desktop/Valleotb/sample_save'

    # device setting
    if torch.cuda.is_available():
        device = "cuda"

    else:
        device = "cpu"
    print(f'Using {device} device')

    # multiprocessing
    torch.multiprocessing.set_start_method('spawn')

    # dataset setting
    usd = PSAD_Dataset(
        audio_folder_dir=AUDIO_DIR,
        metadata_dir=FILENAME_DIR,
        device=device,
        load_first=True
    )

    # data seperation
    train_dataset, val_dataset = torch.utils.data.random_split(usd, [40000, len(usd)-40000])

    # create a data loader for train / validation
    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=8)
    validation_data_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=8)

    # wandb initialize
    wandb.init()

    # construct model and assign it to device
    cnn = ResNet1D(
        in_channels=1,
        base_filters=4,
        kernel_size=64,
        n_classes=10,
        stride=16,
        groups=1,
        n_block=16  # system RAM과 관련이 생겨버림 (4 이상하면 터지는듯)
    ).to(device)

    # wandb logger
    wandb.watch(cnn)

    # instantiate loss function + optimiser
    loss_fn = nn.BCELoss()
    optimiser = torch.optim.Adam(cnn.parameters(),
                                 lr=LEARNING_RATE)

    # scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimiser, milestones=[20, 40], gamma=0.1)

    # train model
    train(cnn, train_data_loader, loss_fn, optimiser, device, EPOCHS)
    torch.save(cnn.state_dict(), "/content/drive/MyDrive/psad_resnet_checkpoints/psad_resnet.pth")

    validation(cnn, validation_data_loader, loss_fn, device, EPOCHS)
    torch.save(cnn.state_dict(), "/content/drive/MyDrive/psad_resnet_checkpoints/psad_resnet_validation.pth")
    print("Model Trained and Stored at psad_resnet.pth")