import torch
import numpy as np
import random
from sklearn import metrics
from tqdm import tqdm

use_cuda=torch.cuda.is_available()
device=torch.device("cuda:2" if use_cuda else "cpu")
if use_cuda:
    print('hello')

SEED = 1234
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if use_cuda:
    torch.cuda.manual_seed(SEED)

bs=16
d_hidden=256
d_output=2
n_layers=2
bidirectional=True
dropout=0.25
model_name='1'
max_epochs=10
require_improvement=3

from pathlib import Path

def read_imdb_split(split_dir):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir/label_dir).iterdir():
            texts.append(text_file.read_text())
            labels.append(0 if label_dir is "neg" else 1)

    return texts, labels

train_texts, train_labels = read_imdb_split('./data/aclImdb/train')
test_texts, test_labels = read_imdb_split('./data/aclImdb/test')

from sklearn.model_selection import train_test_split
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)

from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = IMDbDataset(train_encodings, train_labels)
val_dataset = IMDbDataset(val_encodings, val_labels)
test_dataset = IMDbDataset(test_encodings, test_labels)

'''
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, AdamW

device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
model.to(device)
model.train()

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
trainloader=torch.utils.data.DataLoader(train_dataset,batch_size=bs,shuffle=True)
testloader=torch.utils.data.DataLoader(val_dataset,batch_size=bs,shuffle=True)
valloader=torch.utils.data.DataLoader(test_dataset,batch_size=bs,shuffle=True)

optim = AdamW(model.parameters(), lr=5e-5)

for epoch in range(3):
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        print(loss)
        print(outputs[1])
        loss.backward()
        optim.step()

model.eval()
'''

trainloader=torch.utils.data.DataLoader(train_dataset,batch_size=bs,shuffle=True)
valloader=torch.utils.data.DataLoader(val_dataset,batch_size=bs,shuffle=True)
testloader=torch.utils.data.DataLoader(test_dataset,batch_size=bs,shuffle=True)


from transformers import DistilBertForSequenceClassification,AdamW
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
print(model)

import time
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train(model, train_iter, dev_iter, test_iter):
    if use_cuda:
        model.to(device)
    model.train()
    #optimizer = optim.Adam(model.parameters())
    optimizer = AdamW(model.parameters(), lr=5e-5)
    #criterion = nn.BCEWithLogitsLoss()

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    #writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d.%H.%M', time.localtime())+'_'+which_data+'_'+which_model+'_'+which_task+'_'+exp_number)

    for epoch in range(max_epochs):
        start_time = time.time()
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        train_loss=0
        # scheduler.step() # 学习率衰减
        for batch in tqdm(train_iter):
            optimizer.zero_grad()
            x=batch['input_ids'].to(device)
            m=batch['attention_mask'].to(device)
            y=batch['labels'].to(device)
            outputs = model(x,attention_mask=m, labels=y)
            loss=outputs[0]
            logits=outputs[1]
            loss.backward()
            optimizer.step()
            #训练集的准确率
            y=y.data.cpu().numpy()
            preds = torch.max(logits, 1)[1].cpu().numpy()
            predict_all=np.append(predict_all,preds)
            labels_all=np.append(labels_all,y)
            train_loss+=loss
        train_loss/=len(train_iter)   #train_loss
        train_acc = metrics.accuracy_score(labels_all, predict_all)


        #验证集
        dev_acc, dev_loss = evaluate(model, dev_iter)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        if dev_loss < dev_best_loss:
            dev_best_loss = dev_loss
            improve = '*'
            last_improve=epoch
            torch.save(model.state_dict(),model_name+'.pth')
        else:
            improve = ''
        msg = 'Epoch: {0:>3},  Epoch Time: {1}m {2}s,  Train Loss: {3:>5.2},  Train Acc: {4:>6.2%},  Val Loss: {5:>5.2},  Val Acc: {6:>6.2%} {7}'
        print(msg.format(epoch+1,epoch_mins, epoch_secs,train_loss, train_acc, dev_loss, dev_acc, improve))
        #writer.add_scalar("loss/train", loss.item(), total_batch)
        #writer.add_scalar("loss/dev", dev_loss, total_batch)
        #writer.add_scalar("acc/train", train_acc, total_batch)
        #writer.add_scalar("acc/dev", dev_acc, total_batch)

        if epoch - last_improve > require_improvement:
            # 验证集loss超过1epoch没下降，结束训练
            print("No optimization for a long time, auto-stopping...")
            break
    #writer.close()
    #训练跑完了，使用最佳模型测试
    model.load_state_dict(torch.load(model_name+'.pth'))
    test(model, test_iter)

def evaluate(model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for batch in tqdm(data_iter):
            x=batch['input_ids'].to(device)
            m=batch['attention_mask'].to(device)
            y=batch['labels'].to(device)
            outputs = model(x,attention_mask=m, labels=y)
            loss=outputs[0]
            logits=outputs[1]
            #loss = F.cross_entropy(outputs, y)
            #loss=criterion(outputs,labels)
            loss_total += loss
            y = y.data.cpu().numpy()
            predic = torch.max(logits, 1)[1].cpu().numpy()
            #predic=torch.round(torch.sigmoid(outputs)).cpu().numpy()
            labels_all = np.append(labels_all, y)
            predict_all = np.append(predict_all, predic)
    model.train()
    acc = metrics.accuracy_score(labels_all, predict_all)

    if test:
        report = metrics.classification_report(labels_all, predict_all, labels=[0,1],target_names=['pos','neg'], digits=4,output_dict=True)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion

    return acc, loss_total / len(data_iter)


def test(model, test_iter):
    test_acc, test_loss, test_report, test_confusion = evaluate(model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)

#original
train(model,trainloader,valloader,testloader)
