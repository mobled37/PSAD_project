import torch
import torchaudio

from torch import nn
from torch.utils.data import DataLoader
from custom_dataset_fixed3 import PSAD_Dataset
from resnet1d import ResNet1D
from torch.utils.tensorboard import SummaryWriter
import tqdm
import wandb
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torchmetrics


# BATCH_SIZE = 2
# EPOCHS = 2
# LEARNING_RATE = 0.001
#
# ANNOTATIONS_FILE = "abcd"
# AUDIO_DIR = "/Users/valleotb/Downloads/UrbanSound8K/audio"
# SAMPLE_RATE = 22050
# NUM_SAMPLES = 22050

def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader

def train_single_epoch(model, data_loader, loss_fn, optimiser, writer, global_step, device):
    prog_bar2 = tqdm.tqdm(desc=f'training in progress',
                          total=len(data_loader),
                          position=1,
                          leave=True)
    metric = torchmetrics.F1().to(device)
    # for inputs, targets in data_loader:
    for idx, batch in enumerate(data_loader):
        inputs, targets = batch[0].to(device), batch[1].squeeze(1).to(device)

        # calculate loss
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        '''
        f1 score 다루는 칸 
        '''
        predictions = predictions.detach().cpu()
        targets = targets.detach().cpu()

        # accuracy_sc = accuracy_score(targets, predictions)
        # precision_sc = precision_score(targets, predictions)
        # f1_sc = f1_score(targets, predictions)
        target_long = targets.type(torch.LongTensor).to(device)
        predictions_long = (predictions > .5).type(torch.LongTensor).to(device)

        target_long = torch.reshape(target_long, (-1,)).to(device)
        predictions_long = torch.reshape(predictions_long, (-1,)).to(device)
        f1 = metric(predictions_long, target_long)

        # backpropagate loss and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        global_step += 1
        prog_bar2.update()

    wandb.log({
        'loss': loss.item(),
        'F1 score': metric.compute()
        # 'accuracy_sc': accuracy_sc,
        # 'precision_sc': precision_sc,
        # 'f1_sc': f1_sc
    }, step=global_step)



    print(f"Loss: {loss.item()}")

    writer.add_scalar('Loss/train', loss.item(), global_step)

def train(model, data_loader, loss_fn, optimiser, device, epochs):
    writer = SummaryWriter()
    global_step = 0
    for i in range(epochs):
        print(f"Epoch {i + 1}")
        # tk0 = tqdm(data_loader, total=int(len(data_loader)))
        # counter = 0
        train_single_epoch(model, data_loader, loss_fn, optimiser,
                           device=device, writer=writer, global_step=global_step)
        print("-------------------")
    print('Finished Training')

if __name__ == "__main__":
    BATCH_SIZE = 64
    EPOCHS = 3
    LEARNING_RATE = 0.001

    FILENAME_DIR = '/content/drive/MyDrive/PSAD/sample_metadata/metadata.json'
    AUDIO_DIR = '/content/drive/MyDrive/PSAD/sample_save'

    # FILENAME_DIR = '/Users/valleotb/Desktop/Valleotb/sample_metadata/metadata.json'
    # AUDIO_DIR = '/Users/valleotb/Desktop/Valleotb/sample_save'

    if torch.cuda.is_available():
        device = "cuda"

    else:
        device = "cpu"
    print(f'Using {device} device')
    torch.multiprocessing.set_start_method('spawn')



    usd = PSAD_Dataset(
        audio_folder_dir=AUDIO_DIR,
        metadata_dir=FILENAME_DIR,
        device=device,
        load_first=True
    )
    # create a data loader for the train set
    train_data_loader = DataLoader(usd, batch_size=BATCH_SIZE, num_workers=8)


    wandb.init()
    # construct model and assign it to device
    cnn = ResNet1D(
        in_channels=1,
        base_filters=4,
        kernel_size=16,
        n_classes=10,
        stride=2,
        groups=1,
        n_block=4
    ).to(device)
    wandb.watch(cnn)

    # instantiate loss function + optimiser
    loss_fn = nn.BCELoss()
    optimiser = torch.optim.Adam(cnn.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(cnn, train_data_loader, loss_fn, optimiser, device, EPOCHS)

    torch.save(cnn.state_dict(), "checkpoints/psad_resnet.pth")
    print("Model Trained and Stored at cnn.pth")