

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

