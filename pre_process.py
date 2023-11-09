import os
import argparse


def read_file(path):
    with open(path,'r',encoding='utf-8-sig') as f:
        return f.readlines()
def process_annotation_file(lines):
    entity_dict={}
    for line in lines:
        items = line.strip('\n').split('\t')
        if len(items)==5:
            item_dict={
                'phi':items[1],
                'st_idx':int(items[2]),
                'ed_idx':int(items[3]),
                'entity':items[4]
            }
        elif len(items)==6:
            item_dict={
                'phi':items[1],
                'st_idx':int(items[2]),
                'ed_idx':int(items[3]),
                'entity':items[4],
                'normalized_time':items[5]
            }
        if items[0] not in entity_dict:
            entity_dict[items[0]]=[item_dict]
        else:
            entity_dict[items[0]].append(item_dict)
    return entity_dict

def process_medical_report(txt_name,medical_report_folder,annos_dict):
    file_name = txt_name+'.txt'
    sents = read_file(os.path.join(medical_report_folder,file_name))
    article = "".join(sents)

    bounary, item_idx, temp_seq, seq_pairs = 0, 0, "", []
    new_line_idx = 0
    for w_idx, word in enumerate(article):
        if word == '\n' :
            new_line_idx = w_idx+1
            if article[bounary:new_line_idx]=='\n':
                continue
            if temp_seq=="":
                temp_seq="PHI:NULL"
            sentence=article[bounary:new_line_idx].strip().replace('\t',' ')
            temp_seq=temp_seq.strip("\\n")
            seq_pair = f"{txt_name}\t{new_line_idx}\t{sentence}\t{temp_seq}"
            bounary=new_line_idx
            seq_pairs.append(seq_pair)
            temp_seq=""
        if w_idx == annos_dict[txt_name][item_idx]['st_idx']:
            phi_key=annos_dict[txt_name][item_idx]['phi']
            phi_value=annos_dict[txt_name][item_idx]['entity']
            if 'normalized_time' in annos_dict[txt_name][item_idx]:
                temp_seq+= f"{phi_key}:{phi_value}=>{annos_dict[txt_name][item_idx]['normalized_time']}\\n"
            else:
                temp_seq+= f"{phi_key}:{phi_value}\\n"
            if item_idx==len(annos_dict[txt_name])-1:
                continue
            item_idx+=1
    return seq_pairs

def generate_annotated_medical_report(anno_file_path,medical_report_folder,tsv_output_path):
    annos_lines= read_file(anno_file_path)
    annos_dict = process_annotation_file(annos_lines)
    txt_names=list(annos_dict.keys())

    all_seq_pairs=[]
    for txt_name in txt_names:
        all_seq_pairs.extend(process_medical_report(txt_name,medical_report_folder,annos_dict))
        print(f"report {txt_name} has been processed!")
    print("---All medical report have been processec!---")
    with open(tsv_output_path,'w',encoding='utf-8') as f:
        for seq_pair in all_seq_pairs:
            f.write(seq_pair)


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--annotation','-a',type=str,required=True)
    parser.add_argument('--report_folder','r',type=str,required=True)
    parser.add_argument('--output_path','-o',type=str,required=False,default="./train.tsv")

    args = parser.parse_args()
    annos_file_path = args.annotation 
    medical_report_folder = args.report_folder
    tsv_output_path = args.output_path
    generate_annotated_medical_report(annos_file_path,medical_report_folder,tsv_output_path)