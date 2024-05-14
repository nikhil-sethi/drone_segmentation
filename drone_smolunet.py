from torch import nn
import torch
import torchvision
from torchvision import transforms
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
from functools import partial
from torchvision.transforms import functional as F, InterpolationMode
import matplotlib.pyplot as plt
import enum

# globals
device =  torch.device("cuda")
np.random.seed(1)
torch.manual_seed(5)

colormap = np.array([
    [155.,38.,182.],
    [14.,135.,204.],
    [124.,252.,0.],
    [255.,20.,147.],
    [169.,169.,169.]])


def encoder_level(in_channels, out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            # nn.Conv2d(out_channels, out_channels, 3),
            # nn.ReLU()
        ).to(device)

def upconv(channels):
    return nn.Sequential(nn.ConvTranspose2d(2*channels, channels, kernel_size=2, stride=2),
                                 nn.BatchNorm2d(channels),
                                 nn.ReLU()
                                 ).to(device)

def decoder_level(channels):
    # crop = torchvision.transforms.Resize((size, size))
    # upconv =  # upsamples img H & W and deconvolutes (decreases channels because of stride=2) 
    # upconv = nn.ConvTranspose2d(2*channels, channels, kernel_size=2, stride=2).to(device)
    # decode_layer =  

    # mix_input =  # concatenate along the channels
    return encoder_level(2*channels, channels)

# def crop(x, channels, size):
#     return torchvision.transforms.Resize((channels, size, size))(x)

# model
class droneSegmenter(nn.Module):
    """Unet for semantic segmentation"""
    def __init__(self) -> None:
        super().__init__()

        # Encoder
        self.encoder_1 = encoder_level(3, 16)
        self.encoder_2 = encoder_level(16, 32)
        self.encoder_3 = encoder_level(32, 64)
        self.encoder_4 = encoder_level(64, 128)

        # bottleneck
        self.bottleneck = encoder_level(128, 256)
        
        # decoder
        self.decoder_4 = encoder_level(256, 128)
        self.upconv_4 = upconv(128)
        self.decoder_3 = encoder_level(128, 64)
        self.upconv_3 = upconv(64)
        self.decoder_2 = encoder_level(64, 32)
        self.upconv_2 = upconv(32)
        self.decoder_1 = encoder_level(32, 16)
        self.upconv_1 = upconv(16)

        # output        
        self.output = nn.Conv2d(16, out_channels=5, kernel_size=1)

        self.to(device)

    def forward(self, x):
        # ENCODER
        encoder_out1 = self.encoder_1(x)
        encoder_out2 = self.encoder_2(torch.max_pool2d(encoder_out1, (2,2)))
        encoder_out3 = self.encoder_3(torch.max_pool2d(encoder_out2, (2,2)))
        encoder_out4 = self.encoder_4(torch.max_pool2d(encoder_out3, (2,2)))
        # print(encoder_out1.shape)

        bottleneck_out = self.bottleneck(torch.max_pool2d(encoder_out4, (2,2))) # 1024 x
        
        # print(bottleneck_out.shape, encoder_out4.shape)
        # DECODER
        
        # decoder_in4 = (bottleneck_out, encoder_out4) # 512

        decoder_in4 = torch.cat([self.upconv_4(bottleneck_out), encoder_out4], 1)
        decoder_out4 = self.decoder_4(decoder_in4)
        decoder_in3 = torch.cat([self.upconv_3(decoder_out4), encoder_out3], 1) # 256
        decoder_out3 = self.decoder_3(decoder_in3)
        decoder_in2 = torch.cat([self.upconv_2(decoder_out3), encoder_out2], 1)
        decoder_out2 = self.decoder_2(decoder_in2)
        decoder_in1 = torch.cat([self.upconv_1(decoder_out2), encoder_out1], 1)
        decoder_out1 = self.decoder_1(decoder_in1)

        return self.output(decoder_out1)

# dataset
class droneDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.masks = os.listdir(mask_dir)

        
        self.img_transform = transforms.Compose([
                            transforms.Resize((256, 256)),  # TODO Check this out Resize images to a fixed size
                            transforms.ToTensor(),       
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # image net normalization
                            ])   
        
        # self.mask_transform = transforms.Compose([
        #                     transforms.Resize((388, 388)),  # TODO Check this out Resize images to a fixed size
        #                     # transforms.ToTensor(),       
        #                     ])   
    

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.images[idx])
        mask_name = os.path.join(self.mask_dir, self.masks[idx])
        image = Image.open(img_name).convert("RGB")
        mask_id = Image.open(mask_name)

        if self.transform:
            image = self.img_transform(image).to(device)
            mask_id = F.resize(mask_id, (256,256), interpolation=InterpolationMode.NEAREST)
            mask_id = F.pil_to_tensor(mask_id).squeeze(0).long().to(device)
            # print(mask_id.shape)
            # mask = torch.zeros_like(image)
            # taken from classes_dict.txt
            # mask[:,mask_id==0/255] = torch.tensor([[155.,38.,182.]]).T/255
            # mask[:,mask_id==1/255] = torch.tensor([[14.,135.,204.]]).T/255
            # mask[:,mask_id==2/255] = torch.tensor([[124.,252.,0.]]).T/255
            # mask[:,mask_id==3/255] = torch.tensor([[255.,20.,147.]]).T/255
            # mask[:,mask_id==4/255] = torch.tensor([[169.,169.,169.]]).T/255
            # print(image.shape)
        return image, mask_id

def load_data():

    transform = transforms.Compose([
    # transforms.Resize((572, 572)),  # TODO Check this out Resize images to a fixed size
    transforms.ToTensor(),       
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # image net normalization
    ])

    dataset = droneDataset("/root/datasets/semantic_drone/classes_dataset/classes_dataset/original_images", 
                        "/root/datasets/semantic_drone/classes_dataset/classes_dataset/label_images_semantic", transform=transform)


    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [0.6, 0.2, 0.2])
    batch_size = 8
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader

def calc_class_weights(dataloader):
    weights = torch.zeros(5).to(device)
    for _, target in dataloader:
        
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=5).permute(0,3,1,2)
        
        batch_weight = torch.mean(target_one_hot.float(), dim=[0,2,3])#.float().mean(dim=0)
        # print(batch_weight)
        weights +=batch_weight

    weights = weights/(len(dataloader))
    inverse_weights = 1-weights
    return inverse_weights

def plot(img, gt, pred):
    """Just for testing a particular datapoint"""

    fig, axes = plt.subplots(1,3)
    add_imgs_to_ax(axes, img, gt, pred)
    plt.show()

def add_imgs_to_ax(axes, img, gt, pred):
    axes[0].imshow(img.permute(1,2,0).cpu())

    gt_rgb = colormap[gt.cpu()]
    axes[1].imshow(gt_rgb.astype(int)) # mask
    
    pred_rgb = colormap[pred.cpu()]
    axes[2].imshow(pred_rgb.astype(int)) # mask

def plot_mult(imgs, gts, preds):
    fig, axes = plt.subplots(len(imgs),3)

    for i in range(5): add_imgs_to_ax(axes[i], imgs[i], gts[i], preds[i])
    
    plt.show()


def logit_to_mask(logits):
    """Assuming logits = channels x h x w"""
    mask_id = torch.argmax(logits.softmax(dim=1), dim=1)
    
    return mask_id

def class_acc_batch(pred, target):
    pred_one_hot = torch.nn.functional.one_hot(pred, num_classes=5).permute(0,3,1,2)
    target_one_hot = torch.nn.functional.one_hot(target, num_classes=5).permute(0,3,1,2)
 

    TP_1h = pred_one_hot & target_one_hot 
    FP_1h = (pred_one_hot | target_one_hot) - target_one_hot
    FN_1h = (pred_one_hot | target_one_hot) - pred_one_hot

    TP = torch.sum(TP_1h, dim=[2,3])
    FP = torch.sum(FP_1h, dim=[2,3])
    FN = torch.sum(FN_1h, dim=[2,3])

    batch_acc = torch.nan_to_num(TP/(TP+FP+FN))
    
    return batch_acc

# parameters
lr = 0.001

# create the model and optimizer
# model = droneSegmenter()

def train(model, optimizer, dataloader, loss_fn):
    # start training in loop
    n_epochs = 40
    model.train()
    
    for epoch in range(n_epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        epoch_class_accuracy = torch.zeros(5).to(device)
        for idx, (data, target) in enumerate(dataloader): 
            optimizer.zero_grad()

            out = model.forward(data)

            # target_one_hot = torch.nn.functional.one_hot(target, num_classes=10)
            # print(out.shape, target.shape)
            loss = loss_fn(out, target)
            accuracy = (torch.argmax(out, 1)==target).float().mean()

            pred = logit_to_mask(out)
            # print(out.shape, pred.shape)
            class_acc = class_acc_batch(pred, target).mean(dim=0)
            loss.backward()
            optimizer.step()
            epoch_loss +=loss.item()
            epoch_accuracy += accuracy.item()
            epoch_class_accuracy += class_acc

            # print(loss)
        print("Epoch train loss: ", epoch_loss/len(dataloader), "Epoch accuracy: ", epoch_accuracy/len(dataloader), "Epoch class accuracy: ", epoch_class_accuracy/len(dataloader))

def val(model, dataloader, loss_fn):
    model.eval()
    val_loss = 0
    mean_accuracy = 0 # calcualted pixel wise comparison to target
    for (data, target) in dataloader:
        with torch.no_grad():
            out = model.forward(data)

            # target_one_hot = torch.nn.functional.one_hot(target, num_classes=10)
            # print(out.shape, target.shape)
            loss = loss_fn(out, target)
            accuracy = (torch.argmax(out, 1)==target).float().mean()
            val_loss += loss.item()
            mean_accuracy += accuracy.item()
    print("Mean validation loss: ", val_loss/len(dataloader), "Mean accuracy: ", mean_accuracy/len(dataloader))

if __name__=="__main__":
    # train()
    train_dataloader, val_dataloader, test_dataloader = load_data()
    weights = calc_class_weights(train_dataloader)
    print(weights)
    data, label = next(iter(val_dataloader))

    # plot_test(train_dataloader)
    model = droneSegmenter()
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    train(model, optim, train_dataloader, loss_fn)
    val(model, val_dataloader, loss_fn)

    torch.save(model, 'model_unweighted.pt')
    
    out = model.forward(data)
    preds = logit_to_mask(out)
    idx = 7
    plot(data[idx], label[idx], preds[idx])