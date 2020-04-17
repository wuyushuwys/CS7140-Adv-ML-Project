from utils.model import ECG_Model, ECG_Model, GSR_Model, Dataset
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from torch import optim
import torch
from sklearn.model_selection import train_test_split
from utils.utils import f1_score, model_test, tokenize, plot_confusion_matrix
from sklearn.metrics import confusion_matrix
# load data

data_dir = 'split_data'


gsr_data = np.load(data_dir+'/GSR_DATA.npy')
ecg_data = np.load(data_dir+'/ECG_DATA.npy')
eeg_data = np.load(data_dir+'/EEG_DATA.npy')
ground_truth = np.load(data_dir+'/ground_truth.npy')


# processing eeg data to 2d-timeframes
num, time_frames, channels = eeg_data.shape
eeg_topo_data = np.zeros((num, time_frames, 9, 9))
for i, data in enumerate(eeg_data):
    eeg_topo_data[i, :, 1, 3] = data[:, 0]
    eeg_topo_data[i, :, 2, 0] = data[:, 1]
    eeg_topo_data[i, :, 2, 2] = data[:, 2]
    eeg_topo_data[i, :, 3, 1] = data[:, 3]
    eeg_topo_data[i, :, 4, 0] = data[:, 4]
    eeg_topo_data[i, :, 6 ,0] = data[:, 5]
    eeg_topo_data[i, :, 8, 3] = data[:, 6]
    eeg_topo_data[i, :, 8, 5] = data[:, 7]
    eeg_topo_data[i, :, 6, 8] = data[:, 8]
    eeg_topo_data[i, :, 4, 8] = data[:, 9]
    eeg_topo_data[i, :, 3, 7] = data[:, 10]
    eeg_topo_data[i, :, 2, 6] = data[:, 11]
    eeg_topo_data[i, :, 2, 8] = data[:, 12]
    eeg_topo_data[i, :, 1, 5] = data[:, 13]
    

ground_truth_labels = torch.zeros(ground_truth.shape[0])
labels = []
for i, data in enumerate(ground_truth):
    ground_truth_labels[i], name = tokenize(data)
    labels.append(name)
    

print(sorted(Counter(list(labels)).items()))


# create dataset, split dataset to train_set and test_set
dataset = Dataset(eeg_topo_data, ecg_data, gsr_data, ground_truth_labels)
test_split = 0.3
num_data = len(dataset)
num_test = int(num_data*test_split)
num_train = num_data - num_test
train_set, test_set = torch.utils.data.random_split(dataset, [num_train, num_test])

train_data = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
test_data = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2)


def run(model, board_name, data_index, train_data, test_data):
    tb_file = glob('./runs/'board_name+'/*')
    for f in tb_file:
        os.remove(f)
    writer = SummaryWriter(board_name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    model.to(device)
    total_epoch = 250
    old_test_loss = 0
    for epoch in range(1, total_epoch+1):
        start = time()
        for i, data in enumerate(train_data):
            model.train()
            # get the inputs; data is a list of [eeg, ecg, gsr, labels]
            inputs = data[data_index].to(device)
            labels = data[-1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        train_loss, train_f1, train_accuracy = model_test(model, train_data, device, criterion, data_index)
        test_loss, test_f1, test_accuracy = model_test(model, test_data, device, criterion, data_index)
        writer.add_scalar('Test/Loss', test_loss, epoch)
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Test/F1', test_f1, epoch)
        writer.add_scalar('Train/F1', train_f1, epoch)
        writer.add_scalar('Test/Accuracy', test_accuracy, epoch)
        writer.add_scalar('Train/Accuracy', train_accuracy, epoch)
        if test_loss < old_test_loss:
            torch.save(model.state_dict(), "./model/"+board_name.split('/')[-1]+'.pt')
        if epoch == 1:
            print("### Epoch [{:04d}/{}]\t train loss: {:.4f}\t train F1:{:.4f}\t train acc:{:.4f}\t test loss:{:.4f}\t test F1:{:.4f}\t test acc:{:.4f}"
                  .format(epoch, total_epoch, train_loss, train_f1,train_accuracy,  test_loss, test_f1, test_accuracy))

        if abs(old_test_loss-test_loss) < 1e-4:
            print("Test loss converging\nEarly Stop")
            print("### Epoch [{:04d}/{}]\t train loss: {:.4f}\t train F1:{:.4f}\t train acc:{:.4f}\t test loss:{:.4f}\t test F1:{:.4f}\t test acc:{:.4f}"
                  .format(epoch, total_epoch, train_loss, train_f1,train_accuracy,  test_loss, test_f1, test_accuracy))
            break
        old_test_loss = test_loss
        if epoch % 10 == 0:
            print("### Epoch [{:04d}/{}]\t train loss: {:.4f}\t train F1:{:.4f}\t train acc:{:.4f}\t test loss:{:.4f}\t test F1:{:.4f}\t test acc:{:.4f}"
                  .format(epoch, total_epoch, train_loss, train_f1,train_accuracy,  test_loss, test_f1, test_accuracy))
            
        
    

boards_name = ['./runs/EEG_fusion', './runs/ECG_fusion', './runs/GSR_fusion']
models = [EEG_Model(), ECG_Model(), GSR_Model()]

for i, (board, model) in enumerate(zip(boards_name, models)):
    print(board_name.split('/')[-1])
    print(model)
    run(model, board_name, i, train_data, test_data)
    
    
    
########################### Fusion ##########################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model1 = models[0]
model2 = models[1]
model3 = models[2]

model1.eval()
model2.eval()
model3.eval()


eeg_test_acc, eeg_test_f1, eeg_outputs, y_true_label = model_test(model1, test_data, device, 0, flag='eval')
ecg_test_acc, ecg_test_f1, ecg_outputs, _ = model_test(model2, test_data, device, 1, flag='eval')
gsr_test_acc, gsr_test_f1, gsr_outputs, _ = model_test(model3, test_data, device, 2, flag='eval')


def classification_score(results_c):  # the result of channel c
    f_d = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            dist = np.sqrt(np.sum((results_c[i]-results_c[j])**2)) # euclidean 
            f_d[i,j] = 1/(np.sqrt(2*np.pi))* np.exp((-np.square(dist))/2)  # classification score(normal distribution)   
    return f_d  #(4*4)


def channel_scores(result_c, f_d):   # results:(1*4), f_d_ij:(4*4)  # For channel c
    GauPR_c = np.zeros((1,4))
    for i in range(4):
        GauPR_c = GauPR_c + result_c[i] * f_d[i,:] 
    return GauPR_c  # (1,4) 


def channel_reliability(GauPR_c):
    average_G = np.mean(GauPR_c)
    tmp = 0
    for j in range(4):
        tmp = tmp + np.square(GauPR_c[0,j]-average_G)
    score_c = np.sqrt(tmp/3)
    return score_c


def fusion(scores, GauPRs):  # score = [score_c1, ..._c2, ..._c3] , GauPRs = [[GauPR_c1],[_c2],[_c3]]
    fusion = np.zeros((1,4))
    for j in range(4):
        for c in range(3):
            fusion[0,j] = fusion[0,j] + GauPRs[c,j]*scores[0,c]
    final_label = np.argmax(fusion)
    max_weight = fusion[0,final_label]   # max-weight of label(not model)
    return final_label, max_weight    


def fusion_each(a,b,c): 
    a = np.array(list(a))
    b = np.array(list(b))
    c = np.array(list(c))
    fd1 = classification_score(a)
    fd2 = classification_score(b)
    fd3 = classification_score(c)

    g1 = channel_scores(a,fd1)
    g2 = channel_scores(b,fd2)
    g3 = channel_scores(c,fd3)

    s1 = channel_reliability(g1)
    s2 = channel_reliability(g2)
    s3 = channel_reliability(g3)

    scores = np.hstack((s1,s2,s3)).reshape(1,3)

    weights = np.vstack((g1,g2,g3))
    final_label, max_weight = fusion(scores,weights)
    return final_label, max_weight, scores


labels_test = []
weights_test = []
scores_test = []
for i in range(len(eeg_outputs)):
    final_label, max_weights, scores = fusion_each(eeg_outputs[i], ecg_outputs[i], gsr_outputs[i])
    labels_test.append(final_label)
    weights_test.append(max_weights)
    scores_test.append(scores)
    
accuracy  = (y_true_label == torch.tensor(labels_test)).sum().item()/len(y_true_label)
f1_score(y_true_label, torch.tensor(labels_test))

print("Acc:{}, F1:{}".format(accuracy, f1_score))


cm = confusion_matrix(labels_test, y_true_label.numpy())
target_name = ['HAHV', 'HALV', 'LAHV', 'LALV']
plot_confusion_matrix(cm, target_name, cmap='Blues', normalize=False)

