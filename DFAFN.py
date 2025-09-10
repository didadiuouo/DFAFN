
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
import os
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from models.network import MCGF
from sklearn.decomposition import PCA
import torchvision
import torch.nn.functional as F
import random
from sklearn.decomposition import KernelPCA
from scipy.stats import norm

from sklearn.ensemble import RandomForestClassifier
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
# SEED = 1
# SEED = 0

class SparseAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        encoded = self.activation(self.encoder(x))
        decoded = self.activation(self.decoder(encoded))
        return encoded, decoded

    def l1_regularization(self):
        l1_norm = torch.sum(torch.abs(self.encoder.weight))
        return l1_norm

class MultiModalDataset(Dataset):
    def __init__(self, dataframe, image_dir, text_transform=None, image_transform=None):
        self.data = dataframe
        self.image_dir = image_dir
        self.text_transform = text_transform
        self.image_transform = image_transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        patient_id = self.data.iloc[idx]['Patient ID']
        image_path_1 = os.path.join(self.image_dir, f"{patient_id}.jpg")
        label = self.data.iloc[idx]['Label']

        if os.path.exists(image_path_1):
            image_1 = cv2.imread(image_path_1)
            if self.image_transform:
                image_1 = self.image_transform(image_1)
        else:
            image_1 = torch.zeros((3, 224, 224), dtype=torch.float)

        text_features = self.data.iloc[idx, 1:-1].values.astype('float32')

        if self.text_transform:
            text_features = self.text_transform(text_features)
        if len(text_features) < D_pca:
            text_features = F.pad(text_features, (0, D_pca - len(text_features)))
        else:
            text_features = text_features[:D_pca]

        return {'image_1': image_1, 'text': text_features}, label

class DFAFN(nn.Module):
    def __init__(self, num_classes, image_embedding_size, text_embedding_size):
        super(DFAFN, self).__init__()


        resnet = torchvision.models.resnet18(pretrained=True)
        self.image_encoder = nn.Sequential(
            *list(resnet.children())[:-2],  # 去掉最后两层全连接层
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.image_fc = nn.Linear(resnet.fc.in_features, image_embedding_size)

        self.image_autoencoder = SparseAutoencoder(128, image_embedding_size)


        self.text_conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, padding=1)
        self.text_conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.text_fc = nn.Linear(64 * D_pca, 256)

        self.text_autoencoder = SparseAutoencoder(256, text_embedding_size)

        self.cross = MCGF(64)
        self.mlp_head = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 128))
        # 融合层
        self.fc1 = nn.Linear(image_embedding_size + text_embedding_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(p=0.3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_image, x_text):

        x_image = self.image_encoder(x_image)
        x_image = x_image.view(x_image.size(0), -1)
        # x_image = self.dropout(x_image)
        x_image = self.image_fc(x_image)

        x_text = x_text.unsqueeze(1)
        x_text = F.relu(self.text_conv1(x_text))
        x_text = F.relu(self.text_conv2(x_text))
        x_text = x_text.view(x_text.size(0), -1)
        x_text = self.text_fc(x_text)

        x_text, _ = self.text_autoencoder(x_text)

        x_image = x_image.unsqueeze(1)
        x_text = x_text.unsqueeze(1)
        x_1 = self.cross(x_image, x_text)
        x_2 = self.cross(x_text, x_image)
        x1, x2 = map(lambda t: t[:, 0], (x_1, x_2))
        x = self.mlp_head(x1) + self.mlp_head(x2)  # 100*3  # x = torch.cat((x_image, x_text), dim=1)
        # x = torch.cat((x_image, x_text), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = self.dropout(x)
        # x = self.dropout(F.relu(self.fc1(x)))
        # x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        x = self.softmax(x)
        return x

# auc_max = 0.797597657767233
# best_SEED=168    res50 best_SEED= 20
# auc_max= 0.8007650989445837 best_learning_rate= 3.5e-05
auc_max = 0
# best_SEED=20
SEED = 168
num_epochs = 100
# best_num_epochs = 50
D_pca = 700
learning_rate = 3.5e-05
values = np.arange(0, 0.0001, 0.000001)

best_auc = 0
best_model_path = "best_model.pth"
pretrian = False


for D_pca in range(700, 701):
    # learning_rate = 0.000042
    set_seed(SEED)

    sparse_lambda = 0.01
    scaler = StandardScaler()

    # pca = PCA(n_components=D_pca)
    kpca = KernelPCA(n_components=D_pca, kernel='poly', gamma=0.1)

    excel_data_ori = pd.read_excel(r'./data/Newfeature-603.xlsx', sheet_name='Tm_data')

    excel_data_norm = scaler.fit_transform(excel_data_ori.iloc[:, 1:])
    # excel_data_norm = excel_data_ori.iloc[:, 1:]
    excel_data = kpca.fit_transform(excel_data_norm)

    excel_data = pd.DataFrame(excel_data, columns=[f'PCA{i}' for i in range(1, excel_data.shape[1] + 1)])
    # excel_data['Patient ID'] = excel_data_ori['Patient ID']
    excel_data.insert(0, "Patient ID", excel_data_ori['Patient ID'])
    image_dir = r'./data/zuidaqinewwaikuo/'
    # image_dir = r"F:\5.Classification\radiomics+cnn\radiomics+cnn_(crossattention)_608\picture\zuidaqi"


    label_encoder = LabelEncoder()


    labels = pd.read_excel(r'./data/Newfeature-603.xlsx',
                           header=0, sheet_name="480label")
    label_mapping = dict(zip(labels['namelist'], labels['labels']))

    excel_data['Label'] = excel_data['Patient ID'].apply(lambda x: x.split('_')[0]).map(label_mapping)
    # feature = excel_data.iloc[:, 1:-1]
    # label = excel_data['labels']
    excel_data['Label'] = label_encoder.fit_transform(excel_data['Label'])
    accuracies = []
    aucs = []
    recalls = []
    F1_scores = []
    Precisions = []


    model = DFAFN(num_classes=3, image_embedding_size=64, text_embedding_size=64)

    if pretrian:

        model.load_state_dict(torch.load(best_model_path))
        model.to(device)


        model.eval()
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # model.to(device)


    for split_num in range(5):
        print('split_num=', split_num)
        train_data, test_data = train_test_split(excel_data, test_size=0.2, random_state=np.random.randint(100000))
        model.__init__(num_classes=3, image_embedding_size=64, text_embedding_size=64)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        model.to(device)

        image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),

        ])
        text_transform = None

        train_dataset = MultiModalDataset(train_data, image_dir, text_transform=text_transform,
                                          image_transform=image_transform)
        test_dataset = MultiModalDataset(test_data, image_dir, text_transform=text_transform,
                                         image_transform=image_transform)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            all_labels = []
            all_preds = []

            for batch in train_loader:
                images, texts, labels = batch[0]['image_1'].to(device), batch[0]['text'].to(device), batch[1].to(device)

                optimizer.zero_grad()
                outputs = model(images, texts)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss = criterion(outputs, labels)
                # sparse_loss = model.image_autoencoder.l1_regularization() + model.text_autoencoder.l1_regularization()
                sparse_loss = model.text_autoencoder.l1_regularization()
                loss = loss + sparse_lambda * sparse_loss
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(outputs.detach().cpu().numpy())
                # one_batch_auc = roc_auc_score(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy(), multi_class='ovr')
                # print(f'Training one batch AUC: {one_batch_auc :.4f}')
                # auc += roc_auc_score(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy(), multi_class='ovr')
            scheduler.step()
            epoch_loss = running_loss / len(train_dataset)

            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.8f}')
            accuracy = correct / total

            auc = roc_auc_score(all_labels, all_preds, multi_class='ovr')
            print(f'Training Accuracy: {accuracy * 100:.2f}%,Training AUC: {auc :.4f}')

        model.eval()
        correct = 0
        total = 0
        auc_test = 0
        batch_cnt_test = 0
        with torch.no_grad():
            all_labels = []
            all_preds = []
            for batch in test_loader:
                images, texts, labels = batch[0]['image_1'].to(device), batch[0]['text'].to(device), batch[1].to(device)

                outputs = model(images, texts)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(outputs.detach().cpu().numpy())
                # auc_test += roc_auc_score(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy(), multi_class='ovr')
                # auc = roc_auc_score(labels, outputs, multi_class='ovr')
                batch_cnt_test += 1
                # one_batch_auc = roc_auc_score(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy(), multi_class='ovr')
                # print(f'Validation one batch AUC: {one_batch_auc :.4f}')
        accuracy_val = correct / total
        auc_val = roc_auc_score(all_labels, all_preds, multi_class='ovr')
        recall = recall_score(all_labels, np.argmax(all_preds, axis=1), average='macro')
        precision = precision_score(all_labels, np.argmax(all_preds, axis=1), average='macro', zero_division=1)
        f1 = f1_score(all_labels, np.argmax(all_preds, axis=1), average='macro')

        # auc_test = auc_test/batch_cnt_test
        # print(f'Test Accuracy: {accuracy:.4f}')
        print(f'val Accuracy: {accuracy_val * 100:.4f}%')
        print(f'val AUC: {auc_val :.4f}')
        print(f"val recall: {recall * 100:.4f}%")
        print(f"val Precision: {precision * 100:.4f}%")
        print(f"val F1-Score: {f1 * 100:.4f}%")

        accuracies.append(accuracy_val)
        aucs.append(auc_val)
        recalls.append(recall)
        F1_scores.append(f1)
        Precisions.append(precision)

    auc_interval = norm.interval(0.95, loc=np.mean(aucs), scale=np.std(aucs) / np.sqrt(len(aucs)))
    accuracy_interval = norm.interval(0.95, loc=np.mean(accuracies), scale=np.std(accuracies) / np.sqrt(len(accuracies)))
    recalls_interval = norm.interval(0.95, loc=np.mean(recalls), scale=np.std(recalls) / np.sqrt(len(recalls)))
    F1_scores_interval = norm.interval(0.95, loc=np.mean(F1_scores), scale=np.std(F1_scores) / np.sqrt(len(F1_scores)))
    Precisions_interval = norm.interval(0.95, loc=np.mean(Precisions), scale=np.std(Precisions) / np.sqrt(len(Precisions)))
    print('learning_rate =', learning_rate)
    print(f"Average auc: {np.mean(aucs):.4f}±{np.std(aucs)/np.sqrt(len(aucs)):.4f} 95% CI: {auc_interval}")
    print(f"Average accuracy: {np.mean(accuracies):.4f}±{np.std(accuracies)/np.sqrt(len(accuracies)):.4f} 95% CI: {accuracy_interval}")
    print(f"Average recalls: {np.mean(recalls):.4f}±{np.std(recalls)/np.sqrt(len(recalls)):.4f} 95% CI: {recalls_interval}")
    print(f"Average F1_score: {np.mean(F1_scores):.4f}±{np.std(F1_scores)/np.sqrt(len(F1_scores)):.4f}  95% CI: {F1_scores_interval}")
    print(f"Average Precisions: {np.mean(Precisions):.4f}±{np.std(Precisions)/np.sqrt(len(Precisions)):.4f}  95% CI: {Precisions_interval}")

    if np.mean(aucs) > auc_max:
        best_learning_rate = learning_rate
        auc_max = np.mean(aucs)

    print('auc_max=', auc_max, 'best_learning_rate=', best_learning_rate)

    if auc_val > best_auc:
        best_auc = auc_val
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved with AUC: {best_auc:.4f}")

