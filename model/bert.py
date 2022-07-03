from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
import numpy as np
import datetime
import argparse
import random
import torch
import torch.nn as nn
import time
import os
import pickle
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from transformers import BertForSequenceClassification
from kobert_tokenizer import KoBERTTokenizer

from transformers import AdamW, get_linear_schedule_with_warmup

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def read_file(args):
    # 대아터 가져오기
    print("\n*** reading file... ***")
    train_data_path = os.path.join(args.file_path,args.train_file)
    valid_data_path = os.path.join(args.file_path,args.valid_file)
    test_data_path = os.path.join(args.file_path,args.test_file)

    train_data = pd.read_csv(train_data_path)
    train_data = remove_nan(train_data)
    valid_data = pd.read_csv(valid_data_path)
    valid_data = remove_nan(valid_data)
    test_data = pd.read_csv(test_data_path)
    test_data = remove_nan(test_data)

    print(" *** finished ***")

    x_train = train_data['text']
    y_train = train_data[[args.label]] - 1
    x_valid = valid_data['text']
    y_valid = valid_data[[args.label]] - 1
    x_test = test_data['text']
    y_test = test_data[[args.label]] - 1

    return x_train, y_train, x_valid, y_valid, x_test, y_test

def remove_nan(data):
    print("\n*** precessing Nan values ***")
    data = data.replace(np.nan, '', regex=True)
    print("*** finished ***")
    return data

def prepare_data(x_train, y_train, x_valid, y_valid, x_test, y_test, args):
    # load tokenizer
    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')

    max_length = args.max_len

    train_inputs = []
    train_token_type = []
    train_masks = []

    # 토크나이징
    print("*** Tokenize Train Data ***")
    for i in tqdm(range(len(x_train))):
        tokenized_inputs = tokenizer.encode_plus(text=x_train.iloc[i],padding='max_length',truncation=True,max_length=max_length)
        train_inputs.append(tokenized_inputs.input_ids)
        train_token_type.append(tokenized_inputs.token_type_ids)
        train_masks.append(tokenized_inputs.attention_mask)
    print("*** finished ***")

    train_labels = list(y_train[args.label].iloc[:])

    print("len(train_input)", len(train_inputs))
    print("len(train_token_type)", len(train_token_type))
    print("len(train_attention_mask)", len(train_masks))
    print("len(train_labels)", len(train_labels))

    train_inputs = torch.tensor(train_inputs)
    train_token_type = torch.tensor(train_token_type)
    train_masks = torch.tensor(train_masks)
    train_labels = torch.tensor(train_labels)

    valid_inputs = []
    valid_token_type = []
    valid_masks = []

    print("*** Tokenize Validation Data ***")
    for i in tqdm(range(len(x_valid))):
        tokenized_inputs = tokenizer.encode_plus(text=x_valid.iloc[i],padding='max_length',truncation=True,max_length=max_length)
        valid_inputs.append(tokenized_inputs.input_ids)
        valid_token_type.append(tokenized_inputs.token_type_ids)
        valid_masks.append(tokenized_inputs.attention_mask)
    print("*** finished ***")

    valid_labels = list(y_valid[args.label].iloc[:])

    print("len(valid_input)", len(valid_inputs))
    print("len(valid_token_type)", len(valid_token_type))
    print("len(valid_attention_mask)", len(valid_masks))
    print("len(valid_labels)", len(valid_labels))

    valid_inputs = torch.tensor(valid_inputs)
    valid_token_type = torch.tensor(valid_token_type)
    valid_masks = torch.tensor(valid_masks)
    valid_labels = torch.tensor(valid_labels)


    train_data = TensorDataset(train_inputs, train_token_type, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

    validation_data = TensorDataset(valid_inputs, valid_token_type, valid_masks, valid_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=args.batch_size)

    test_inputs = []
    test_token_type = []
    test_masks = []

    print("*** Tokenize Test Data ***")
    for i in tqdm(range(len(x_test))):
        tokenized_inputs = tokenizer.encode_plus(text=x_test.iloc[i],padding='max_length',truncation=True,max_length=max_length)
        test_inputs.append(tokenized_inputs.input_ids)
        test_token_type.append(tokenized_inputs.token_type_ids)
        test_masks.append(tokenized_inputs.attention_mask)
    print("*** finished ***")

    test_labels = list(y_test[args.label].iloc[:])

    print("len(test_input)", len(test_inputs))
    print("len(test_token_type)", len(test_token_type))
    print("len(test_attention_mask)", len(test_masks))
    print("len(test_labels)", len(test_labels))

    test_inputs = torch.tensor(test_inputs)
    test_token_type = torch.tensor(test_token_type)
    test_masks = torch.tensor(test_masks)
    test_labels = torch.tensor(test_labels)

    test_data = TensorDataset(test_inputs, test_token_type, test_masks, test_labels)
    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.batch_size)
    print("======== DONE!!! ========")

    return train_dataloader, validation_dataloader, test_dataloader

def prepare_test_data(x_test,y_test, args):

    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
    max_length = args.max_len

    test_inputs = []
    test_token_type = []
    test_masks = []

    print("*** Tokenize Test Data ***")
    for i in tqdm(range(len(x_test))):
        tokenized_inputs = tokenizer.encode_plus(text=x_test.iloc[i],padding='max_length',truncation=True,max_length=max_length)
        test_inputs.append(tokenized_inputs.input_ids)
        test_token_type.append(tokenized_inputs.token_type_ids)
        test_masks.append(tokenized_inputs.attention_mask)
    print("*** finished ***")

    test_labels = list(y_test[args.label].iloc[:])

    print("len(test_input)", len(test_inputs))
    print("len(test_token_type)", len(test_token_type))
    print("len(test_attention_mask)", len(test_masks))
    print("len(test_labels)", len(test_labels))

    test_inputs = torch.tensor(test_inputs)
    test_token_type = torch.tensor(test_token_type)
    test_masks = torch.tensor(test_masks)
    test_labels = torch.tensor(test_labels)

    test_data = TensorDataset(test_inputs, test_token_type, test_masks, test_labels)
    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.batch_size)
    print("======== DONE!!! ========")

    return test_dataloader

# accuracy function
def flat_metric(preds, labels):
    return accuracy_score(labels,preds), recall_score(labels, preds, average='weighted'), precision_score(labels, preds, average='weighted'), f1_score(labels, preds, average='weighted')

def flat_return(pred, labels):
    pred_flat = np.argmax(pred, axis=1).flatten() 
    labels_flat = labels.flatten() 
    return pred_flat.tolist(), labels_flat.tolist()

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def train(train_dataloader, validation_dataloader, args, device):

    # load model classification layer가 달린 bertmodel
    model = BertForSequenceClassification.from_pretrained("skt/kobert-base-v1", num_labels=5)
    
    # load parameter to GPU가 1대인 경우 아래 코드를 사용
    model.cuda()

    # gpu가 여러대인 경우에는 아래의 코드를 사용
    #if torch.cuda.device_count() > 1:
    #    model = nn.DataParallel(model) # 
    #model.to(device)

    # 가장 좋은 정확도 저장을 위해 변수 선언
    best_acc = 0

    optimizer = AdamW(model.parameters(),
                    lr = args.learning_rate, # 학습률
                    eps = args.epsilon # 0으로 나누는 것을 방지하기 위한 epsilon 값
                    )

    total_steps = len(train_dataloader) * args.epoch * 1/torch.cuda.device_count()
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = int(total_steps*0.2),
                                                num_training_steps = total_steps)

    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    model.zero_grad()

    for epoch_i in range(0, args.epoch):
        
        # ========================================
        #               Training
        # ========================================
        
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, args.epoch))
        print('Training...')

        t0 = time.time()
        total_loss = 0

        model.train()
            
        for step, batch in tqdm(enumerate(train_dataloader)):
            if step % 500 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            batch = tuple(t.to(device) for t in batch)
            
            b_input_ids, b_input_token, b_input_mask, b_labels = batch

            outputs = model(b_input_ids, 
                        token_type_ids=b_input_token, 
                        attention_mask=b_input_mask, 
                        labels=b_labels)
            
            loss = outputs['loss'].mean()
            if step%50==0 and not step==0:
                print("Step_Loss : ", loss)
            total_loss += loss.item()

            loss.backward() # have to use mean function if you are using multi-GPU
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            scheduler.step()

            model.zero_grad()

        avg_train_loss = total_loss / len(train_dataloader) # normalize loss          

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
            
        # ========================================
        #               Validation
        # ========================================

        print("")
        print("Running Validation...")

        t0 = time.time()
        model.eval()

        flat_labels, flat_preds = [], []

        for batch in validation_dataloader:
            batch = tuple(t.to(device) for t in batch)
            
            b_input_ids, b_input_token, b_input_mask, b_labels = batch
            with torch.no_grad():   
                outputs = model(b_input_ids, 
                                token_type_ids=b_input_token, 
                                attention_mask=b_input_mask)

            logits = outputs['logits']

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            
            flat_pred, flat_label = flat_return(logits, label_ids)
            flat_preds+=flat_pred
            flat_labels+=flat_label

        #avg_train_loss = eval_loss / len(validation_dataloader) # normalize loss          
        eval_acc, eval_recall, eval_pre, eval_f1 = flat_metric(flat_labels,flat_preds)
        #print("Valid Loss: {0:.2f}".format(eval_loss))
        print("Accuracy: {0:.4f}".format(eval_acc))
        print("Recall: {0:.4f}".format(eval_recall))
        print("Precision: {0:.4f}".format(eval_pre))
        print("F1 : {0:.4f}".format(eval_f1))

        if best_acc < (eval_acc):
            best_acc = eval_acc
            print("Saving model...")
            torch.save({'state_dict': model.state_dict()}, f'{args.checkpoint_name}.pth.tar')

    print("")
    print("Training complete!")

    return model

def test(test_dataloader, model, device):
    t0 = time.time()

    if model == None:
        model =  BertForSequenceClassification.from_pretrained("skt/kobert-base-v1", num_labels=5)
        checkpoint = torch.load(f'{args.checkpoint_name}.pth.tar')
        print("======== Load Model From Checkpoint ========")
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        print("======== DONE! ========")
        
    model.eval()

    flat_labels, flat_preds = [], []
    
    print("======== Start Testing!!! ========")
    for step, batch in enumerate(test_dataloader):
        if step % 100 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print(' Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(test_dataloader), elapsed))

        batch = tuple(t.to(device) for t in batch)
        
        b_input_ids, b_input_token, b_input_mask, b_labels = batch
        with torch.no_grad():     
            outputs = model(b_input_ids, 
                            token_type_ids=b_input_token, 
                            attention_mask=b_input_mask)
                

        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        
        flat_pred, flat_label = flat_return(logits, label_ids)
        flat_preds+=flat_pred
        flat_labels+=flat_label
    
    with open(f"{args.label}_bert_label.txt", "wb") as fp:
        pickle.dump(flat_labels, fp)
    with open(f"{args.label}_bert_pred.txt", "wb") as fp:
        pickle.dump(flat_preds, fp)

    eval_acc, eval_recall, eval_pre, eval_f1 = flat_metric(flat_labels,flat_preds)
    print("")
    print("======== Testing DONE!!! ========")
    print("Accuracy: {0:.4f}".format(eval_acc))
    print("Recall: {0:.4f}".format(eval_recall))
    print("Precision: {0:.4f}".format(eval_pre))
    print("F1 : {0:.4f}".format(eval_f1))
    print("Test took: {:}".format(format_time(time.time() - t0)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", default='dataset/1119', type=str)
    parser.add_argument("--checkpoint_name", required=True, type=str)
    parser.add_argument("--label", required=True, type=str)
    parser.add_argument("--max_len", default=256, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--gpu", default=1, type=int)
    parser.add_argument("--epoch", default=10, type=int)
    parser.add_argument("--learning_rate", default=1e-6, type=float)
    parser.add_argument("--epsilon", default=1e-8, type=float)
    parser.add_argument('--seed', type=int, default=12345, help="random seed for initialization")
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()
    args.train_file = 
    args.valid_file = 
    args.test_file = 

    print("======== Hparams for Model ========")
    print(args)
    set_seed(args)

    x_train, y_train, x_valid, y_valid, x_test, y_test = read_file(args)

    if torch.cuda.is_available():    
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
    else:
        device = torch.device("cpu")
        print('No GPU available, using the CPU instead.')

    if args.eval == True:
        test_dataloader = prepare_test_data(x_test, y_test, args)
        test(test_dataloader, None, device)

    else:
        train_dataloader, validation_dataloader, test_dataloader = prepare_data(x_train, y_train, x_valid, y_valid, x_test, y_test, args)
        model = train(train_dataloader, validation_dataloader, args, device)
        test(test_dataloader, model, device)
    exit()
