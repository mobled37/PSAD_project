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

def train_single_epoch(model, data_loader, loss_fn, optimiser, global_step, device, val=False):

    if val:
        model.eval().to(device)
    else:
        model.train().to(device)

    train_type = 'train' if not val else 'val'
    prog_bar2 = tqdm.tqdm(desc=f'{train_type} in progress',
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
        # if val:
        #     with torch.no_grad:
        #         predictions = model(inputs)
        # else:
        #     predictions = model(inputs)
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
        accuracy_sc = accuracy(predictions_long, target_long).to(device)
        acc_sum += accuracy_sc

        # sum loss
        loss_sum += loss.item()

        if not val:
            # backpropagate loss and update weights
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            global_step += 1

        # logger
        prog_bar2.update()
    metrics = {
        f'{train_type}_accuracy': acc_sum / len(data_loader),
        f'{train_type}_loss': loss_sum / len(data_loader),
    }
    print(f"Loss: {loss_sum / len(data_loader)}")

    # writer.add_scalar('Loss/train', loss.item(), global_step)
    return metrics, global_step

def train(model, train_loader, val_loader, loss_fn, optimiser, device, epochs):
    # writer = SummaryWriter()
    global_step = 0
    for i in range(epochs):
        print(f"Epoch {i + 1}")
        metrics, step = train_single_epoch(model, train_loader, loss_fn, optimiser,
                           device=device, global_step=global_step)
        wandb.log(metrics, step=global_step)
        metrics, _ = train_single_epoch(model, val_loader, loss_fn, optimiser,
                                        device, global_step, val=True)
        global_step = step
        print("-------------------")
    print('Finished Training')

if __name__ == "__main__":

    """
    in test mode change :
    BATCH_SIZE, EPOCHS, scheduler milestones
    
    check cpu system ram 
    base_filters
    
    check gpu ram
    BATCH_SIZE, base_filters
    
    performance check
    kernel_size, stride, n_blocks
    
    I WANT TO KNOW n_blocks MEAN!!!!
    """

    # Hyperparameter
    BATCH_SIZE = 64
    EPOCHS = 50
    LEARNING_RATE = 0.001

    FILENAME_DIR = '/content/drive/MyDrive/PSAD/sample_metadata/metadata.json'
    AUDIO_DIR = '/content/drive/MyDrive/PSAD/sample_save'

    # For Test Mode
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
    train_data_percent = int(0.8 * len(usd))
    train_dataset, val_dataset = torch.utils.data.random_split(usd, [train_data_percent, len(usd) - train_data_percent])

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
        n_block=16  # n_blocks 의미 알아낼것
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
    train(model=cnn,
          train_loader=train_data_loader,
          val_loader=validation_data_loader,
          loss_fn=loss_fn,
          optimiser=optimiser,
          device=device,
          epochs=EPOCHS)
    torch.save(cnn.state_dict(), "/content/drive/MyDrive/psad_resnet_checkpoints/psad_resnet.pth")

    # validation(cnn, validation_data_loader, loss_fn, device, EPOCHS)
    # torch.save(cnn.state_dict(), "/content/drive/MyDrive/psad_resnet_checkpoints/psad_resnet_validation.pth")
    print("Model Trained and Stored at psad_resnet.pth")