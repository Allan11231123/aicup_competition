import argparse
import os
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, get_linear_schedule_with_warmup
import re
import random
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torch.utils.data import Dataset
from datasets import load_dataset, Features, Value
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from islab.aicup import collate_batch_with_prompt_template
from islab.aicup import OpenDeidBatchSampler
from transformers import AutoConfig
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm,trange

def set_torch_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benckmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
def read_file(path):
    with open(path , 'r' , encoding = 'utf-8-sig') as fr:
        return fr.readlines()

def sample_text(model, tokenizer, text, n_words=20):
    model.eval()
    text = tokenizer.encode(text)
    inputs, past_key_values = torch.tensor([text]).to(device), None

    with torch.no_grad():
        for _ in range(n_words):
            out = model(inputs, past_key_values=past_key_values)
            logits = out.logits
            past_key_values = out.past_key_values
            log_probs = F.softmax(logits[:, -1], dim=-1)
            inputs = torch.multinomial(log_probs, 1)
            text.append(inputs.item())
            if tokenizer.decode(inputs.item()) == eos:
                break

    return tokenizer.decode(text)

def process_valid_data(test_txts , out_file):
    with open(out_file , 'w' , encoding = 'utf-8') as fw:
        for txt in test_txts:
            m_report = read_file(txt)
            boundary = 0
            # temp = ''.join(m_report)
            fid = txt.split('/')[-1].replace('.txt' , '')
            for idx,sent in enumerate(m_report):
                if sent.replace(' ' , '').replace('\n' , '').replace('\t' , '') != '':
                    sent = sent.replace('\t' , ' ')
                    fw.write(f"{fid}\t{boundary}\t{sent}\n")
                # else:
                #     print(f"{fid}\t{boundary}\t{sent}\n")
                #     assert 1==2
                boundary += len(sent)

def get_anno_format(sentence , infos , boundary,train_phi_category):
    anno_list = []
    lines = infos.split("\n")
    normalize_keys = ['DATE' , "TIME" , "DURATION" , "SET"]
    phi_dict = {}
    for line in lines:
        parts = line.split(":")
        if parts[0] not in train_phi_category or parts[1] == '':
            continue
        if len(parts) == 2:
            phi_dict[parts[0]] = parts[1].strip()
    for phi_key, phi_value in phi_dict.items():
        normalize_time = None
        if phi_key in normalize_keys:
            if '=>' in phi_value:
                temp_phi_values = phi_value.split('=>')
                phi_value = temp_phi_values[0]
                normalize_time = temp_phi_values[-1]
            else:
                normalize_time = phi_value
        try:
            matches = [(match.start(), match.end()) for match in re.finditer(phi_value, sentence)]
        except:
            continue
        for start, end in matches:
            if start == end:
                continue
            item_dict = {
                        'phi' : phi_key,
                        'st_idx' : start + int(boundary),
                        'ed_idx' : end + int(boundary),
                        'entity' : phi_value,
            }
            if normalize_time is not None:
                item_dict['normalize_time'] = normalize_time
            anno_list.append(item_dict)
    return anno_list

def aicup_predict(model, tokenizer,input, train_phi_category, template = "<|endoftext|> __CONTENT__\n\n####\n\n"):
    seeds = [template.replace("__CONTENT__", data['content']) for data in input]
    sep = tokenizer.sep_token
    eos = tokenizer.eos_token
    pad = tokenizer.pad_token
    pad_idx = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    """Generate text from a trained model."""
    model.eval()
    device = model.device
    texts = tokenizer(seeds, return_tensors = 'pt', padding=True).to(device)
    outputs = []
    #return
    with torch.cuda.amp.autocast():
        output_tokens = model.generate(**texts, max_new_tokens=400, pad_token_id = pad_idx,
                                        eos_token_id=tokenizer.convert_tokens_to_ids(eos))
        preds = tokenizer.batch_decode(output_tokens)
        for idx , pred in enumerate(preds):
            if "NULL" in pred:
                continue
            phi_infos = pred[pred.index(sep)+len(sep):].replace(pad, "").replace(eos, "").strip()
            annotations = get_anno_format(input[idx]['content'] , phi_infos , input[idx]['idx'],train_phi_category=train_phi_category)

            for annotation in annotations:
                if 'normalize_time' in annotation:
                    outputs.append(f'{input[idx]["fid"]}\t{annotation["phi"]}\t{annotation["st_idx"]}\t{annotation["ed_idx"]}\t{annotation["entity"]}\t{annotation["normalize_time"]}')
                else:
                    outputs.append(f'{input[idx]["fid"]}\t{annotation["phi"]}\t{annotation["st_idx"]}\t{annotation["ed_idx"]}\t{annotation["entity"]}')
    return outputs
    
def main():
    parser = argparse.ArgumentParser(description='This is a training program for aicup competition.')
    parser.add_argument('--input_path','-i',type=str,required=True)
    parser.add_argument('--model_name','-m',type=str,required=True)
    

    args = parser.parse_args()
    input_file = args.input_path
    model_name = args.model_name
    
    set_torch_seed()
    
    dataset = load_dataset("csv", data_files=input_file, delimiter='\t',
                       features = Features({
                              'fid': Value('string'), 'idx': Value('int64'),
                              'content': Value('string'), 'label': Value('string')}),
                              column_names=['fid', 'idx', 'content', 'label'], keep_default_na=False)
    train_data = list(dataset['train'])
    plm = "gpt2" #"EleutherAI/pythia-70m-deduped"
    bos = '<|endoftext|>'
    eos = '<|END|>'
    pad = '<|pad|>'
    sep ='\n\n####\n\n'
    special_tokens_dict = {'eos_token': eos, 'bos_token': bos, 'pad_token': pad, 'sep_token': sep}
    tokenizer = AutoTokenizer.from_pretrained(plm)
    tokenizer.padding_side = 'left'
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"{tokenizer.pad_token}: {tokenizer.pad_token_id}")
    
    BATCH_SIZE =4
    bucket_train_dataloader = DataLoader(train_data,
                                        batch_sampler=OpenDeidBatchSampler(train_data, BATCH_SIZE),
                                        collate_fn=lambda batch: collate_batch_with_prompt_template(batch, tokenizer),
                                        pin_memory=False)
    config = AutoConfig.from_pretrained(plm,
                                    bos_token_id=tokenizer.bos_token_id,
                                    eos_token_id=tokenizer.eos_token_id,
                                    pad_token_id=tokenizer.pad_token_id,
                                    sep_token_id=tokenizer.sep_token_id,
                                    output_hidden_states=False)

    model = AutoModelForCausalLM.from_pretrained(plm, config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    EPOCHS = 10 # CHANGE TO THE NUMBER OF EPOCHS YOU WANT
    optimizer = AdamW(model.parameters(),lr=3e-5) # YOU CAN ADJUST LEARNING RATE

    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    
    # 模型儲存路徑
    model_dir = f"/content/drive/MyDrive/台大課程資料/資訊檢索/資訊檢索比賽資料/bert_model/{model_name}"
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    min_loss = 9999

    global_step = 0
    total_loss = 0

    model.train()
    for _ in trange(EPOCHS, desc="Epoch"):
        model.train()
        total_loss = 0

        # Training loop
        predictions , true_labels = [], []

        for step, (seqs, labels, masks) in enumerate(bucket_train_dataloader):
            seqs = seqs.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            model.zero_grad()
            outputs = model(seqs, labels=labels, attention_mask=masks)
            logits = outputs.logits
            loss = outputs.loss
            loss = loss.mean()

            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        avg_train_loss = total_loss / len(bucket_train_dataloader)
        print("Average train loss: {}".format(avg_train_loss))
        torch.save(model.state_dict(), os.path.join(model_dir , 'BERT_Finial.pt'))
        if avg_train_loss < min_loss:
            min_loss = avg_train_loss
            torch.save(model.state_dict(), os.path.join(model_dir , 'BERT_best.pt'))
    model.load_state_dict(torch.load(os.path.join(model_dir , 'BERT_best.pt')))
    model = model.to(device)
    test_phase_path = "./Validation_Release"
    valid_out_file_path = './valid.tsv'
    test_txts = list(map(lambda x:os.path.join(test_phase_path , x) , os.listdir(test_phase_path)))
    test_txts = sorted(test_txts)
    valid_data = process_valid_data(test_txts , valid_out_file_path)
    valid_data = load_dataset("csv", data_files=valid_out_file_path, delimiter='\t',
                          features = Features({
                              'fid': Value('string'), 'idx': Value('int64'),
                              'content': Value('string'), 'label': Value('string')}),
                              column_names=['fid', 'idx', 'content', 'label'])
    valid_list= list(valid_data['train'])
    train_phi_category = ['PATIENT', 'DOCTOR', 'USERNAME',
             'PROFESSION',
             'ROOM', 'DEPARTMENT', 'HOSPITAL', 'ORGANIZATION', 'STREET', 'CITY', 'STATE', 'COUNTRY', 'ZIP', 'LOCATION-OTHER',
             'AGE',
             'DATE', 'TIME', 'DURATION', 'SET',
             'PHONE', 'FAX', 'EMAIL', 'URL', 'IPADDR',
             'SSN', 'MEDICALRECORD', 'HEALTHPLAN', 'ACCOUNT', 'LICENSE', 'VEHICLE', 'DEVICE', 'BIOID', 'IDNUM']
    BATCH_SIZE = 32

    with open(os.path.join(model_dir,"answer.txt"),'w',encoding='utf8') as f:
        for i in tqdm(range(0, len(valid_list), BATCH_SIZE)):
            with torch.no_grad():
                seeds = valid_list[i:i+BATCH_SIZE]
                outputs = aicup_predict(model, tokenizer, input=seeds,train_phi_category=train_phi_category)
                for o in outputs:
                    f.write(o)
                    f.write('\n')
if __name__ == '__main__':
    main()