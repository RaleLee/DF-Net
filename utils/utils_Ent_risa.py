import json
import torch
import torch.utils.data as data
import torch.nn as nn
from utils.config import *
import ast

from utils.utils_general import *


def read_langs(file_name, max_line=None):
    print(("Reading lines from {}".format(file_name)))
    data, context_arr, conv_arr, kb_arr, domain_dict, domain_l = [], [], [], [], {}, []
    max_resp_len = 0
    node_list = []

    with open(file_name, encoding='utf-8') as fin:
        cnt_lin, sample_counter, node_idx = 1, 1, 0
        for line in fin:
            line = line.strip()
            if line:
                # 处理domain
                if line.startswith("#"):
                    flag = 0  # 标记是否是domain
                    line = line.split()
                    for a in line:
                        if a == "#":  # 若是'#'则跳过
                            continue
                        if a.startswith("0"):  # 是domain的序号
                            domain_idx = int(a)
                            assert 3 >= domain_idx >= 0
                            # domain_l.append(domain_idx)
                            flag = 1
                            continue
                        if flag == 1:  # 是domain
                            domain_dict[domain_idx] = a
                            assert 8 >= domains[a] >= 0
                            # domain_l.append(domains[a])
                            flag = 0
                            node_list.append([a, domain_idx, node_idx])
                            node_idx += 1
                            continue
                        dialog_id = a  # 读取dialogue ID
                    domain_l = "全部"
                    continue
                # 处理每句话，每行是一个KB(entity, attribute, value)/一个query-answer对
                nid, line = line.split(' ', 1)
                # 处理answer-query对
                if '\t' in line:
                    # 将user/response/gold entity分开
                    u, r, u_seged, r_seged, gold_ent = line.split('\t')
                    # 生成user话中每个词的memory
                    gen_u = generate_memory(u_seged, "$u", str(nid))
                    context_arr += gen_u
                    conv_arr += gen_u
                    for tri in gen_u:
                        node_list.append([tri, node_idx])
                        node_idx += 1
                    # Get gold entity for each domain
                    # eval 能将字符串转为其原本的数据形式list/tuple/dict
                    # 这里的ast.literal_eval能安全的转换，即该字符串不合法时直接抛出异常
                    gold_ent = ast.literal_eval(gold_ent)
                    # ent_idx_restaurant, ent_idx_attraction, ent_idx_hotel = [], [], []
                    # if task_type == "restaurant":
                    #     ent_idx_restaurant = gold_ent
                    # elif task_type == "attraction":
                    #     ent_idx_attraction = gold_ent
                    # elif task_type == "hotel":
                    #     ent_idx_hotel = gold_ent
                    ent_index = list(set(gold_ent))

                    # Get local pointer position for each word in system response
                    ptr_index = []
                    for key in r_seged.split():
                        # 获取local指针，对于之前整理好的global指针，如果这个单词是backend的单词，那就记录它的位置
                        index = [loc for loc, val in enumerate(context_arr) if (val[0] == key and key in ent_index)]
                        # 这里取最大的index，如果没有，就取句子长度
                        if index:
                            index = max(index)
                        else:
                            index = len(context_arr)
                        ptr_index.append(index)

                    # Get global pointer labels for words in system response, the 1 in the end is for the NULL token
                    # 对于user+KB中的单词，如果是system response出现的单词或者是KB实体，就标为1，否则为0，然后添加了一个1对应句子末尾NULL
                    selector_index = [1 if (word_arr[0] in ent_index or word_arr[0] in r.split()) else 0 for word_arr in
                                      context_arr] + [1]
                    # 生成带sketch的回复
                    sketch_response, gold_sketch = generate_template(r_seged, gold_ent, kb_arr, domain_dict, node_list)
                    # if len(domain_label) < 3:
                    #     domain_label.append(RiSA_PAD_token)
                    # assert len(domain_label) == 3
                    # 把这段对话的所有内容放到一个dict中，然后加入到总数据
                    data_detail = {
                        'context_arr': list(context_arr + [['$$$$'] * MEM_TOKEN_SIZE]),  # $$$$ is NULL token
                        'response': r_seged,
                        'sketch_response': sketch_response,
                        'gold_sketch': gold_sketch,
                        'ptr_index': ptr_index + [len(context_arr)],
                        'selector_index': selector_index,
                        'ent_index': ent_index,
                        'conv_arr': list(conv_arr),
                        'kb_arr': list(kb_arr),
                        'id': int(sample_counter),
                        'ID': int(cnt_lin),
                        'domain': domain_l}
                    data.append(data_detail)
                    # 注意，在这里就按照turn来生成了多个对话，将gold response作为context历史一并加入
                    gen_r = generate_memory(r_seged, "$s", str(nid))
                    context_arr += gen_r
                    conv_arr += gen_r
                    for tri in gen_r:
                        node_list.append([tri, node_idx])
                        node_idx += 1
                    # 统计一下最长的回复长度
                    if max_resp_len < len(r_seged.split()):
                        max_resp_len = len(r_seged.split())
                    sample_counter += 1
                # 处理(entity, attribute, value)
                else:
                    r = line
                    kb_info = generate_memory(r, "", str(nid))
                    context_arr = kb_info + context_arr
                    kb_arr += kb_info
                    kb_info.extend([int(nid), node_idx])
                    node_list.append(kb_info)
                    node_idx += 1
            else:
                cnt_lin += 1
                context_arr, conv_arr, kb_arr, node_list, domain_dict= [], [], [], [], {}
                node_idx = 0
                if max_line and cnt_lin >= max_line:
                    break

    return data, max_resp_len


def generate_template(sentence, sent_ent, kb_arr, domain_dict, node_list):
    """
    Based on the system response and the provided entity table, the output is the sketch response.
    """
    dup_list = ["名称", "区域", "是否地铁直达", "电话号码", "地址", "评分", "价位", "制片国家/地区", "类型", "年代", "主演", "导演", "片名", "主演名单", "豆瓣评分",
                "出发地", "目的地", "日期", "到达时间", "票价", "天气"]
    domain = ""
    sketch_response = []
    gold_sketch = []
    if sent_ent == []:
        sketch_response = sentence.split()
    else:
        for word in sentence.split():
            if word not in sent_ent:
                sketch_response.append(word)
            else:
                ent_type = None
                for kb_item in kb_arr:
                    if word == kb_item[0]:
                        for k in node_list:
                            if k[0] == kb_item:
                                domain = domain_dict[k[1]]
                        ent_type = kb_item[1]
                        break
                if ent_type in dup_list:
                    ent_type = domain + '_' + ent_type
                assert ent_type is not None
                # sketch type就是带有@的实体属性种类标记
                sketch_response.append('@' + ent_type)
                gold_sketch.append('@' + ent_type)
    sketch_response = " ".join(sketch_response)
    return sketch_response, gold_sketch


def generate_memory(sent, speaker, time):
    sent_new = []
    sent_token = sent.split(' ')
    if speaker == "$u" or speaker == "$s":
        for idx, word in enumerate(sent_token):
            # 对于user/system的话，将每个单词拆开
            # list[单词，speaker种类，对话轮次，单词index，PAD]
            temp = [word, speaker, 'turn' + str(time), 'word' + str(idx)] + ["PAD"] * (MEM_TOKEN_SIZE - 4)
            sent_new.append(temp)
    else:
        # 处理(entity, attribute, value) 反转(value, attribute, entity, PAD)
        token_list, object_list = [], []
        for i in range(len(sent_token)):
            if i == 0 or i == 1:
                token_list.append(sent_token[i])
            else:
                object_list.append(sent_token[i])
        token_list.append(" ".join(object_list))
        sent_token = sent_token[::-1] + ["PAD"] * (MEM_TOKEN_SIZE - len(sent_token))
        sent_new.append(sent_token)
    return sent_new


def prepare_data_seq(batch_size=100, fast_test=False):
    file_train = 'data/RiSAWOZ/train.txt' if not fast_test else 'data/RiSAWOZ/dev.txt'
    file_dev = 'data/RiSAWOZ/dev.txt'
    file_test = 'data/RiSAWOZ/test.txt'
    # 读入数据，然后统计出最长的回复长度
    pair_train, train_max_len = read_langs(file_train, max_line=None)
    pair_dev, dev_max_len = read_langs(file_dev, max_line=None)
    pair_test, test_max_len = read_langs(file_test, max_line=None)
    max_resp_len = max(train_max_len, dev_max_len, test_max_len) + 1
    # Lang类提供词典，序列化与反序列化
    lang = Lang()
    # 完成数据集构造，返回的是dataloader
    train = get_seq(pair_train, lang, batch_size, True)
    dev = get_seq(pair_dev, lang, batch_size, False)
    test = get_seq(pair_test, lang, batch_size, False)

    print("Read %s sentence pairs train" % len(pair_train))
    print("Read %s sentence pairs dev" % len(pair_dev))
    print("Read %s sentence pairs test" % len(pair_test))
    print("Vocab_size: %s " % lang.n_words)
    print("Max. length of system response: %s " % max_resp_len)
    print("USE_CUDA={}".format(USE_CUDA))

    return train, dev, test, [], lang, max_resp_len


def get_data_seq(file_name, lang, max_len, batch_size=1):
    pair, _ = read_langs(file_name, max_line=None)
    # print(pair)
    d = get_seq(pair, lang, batch_size, False)
    return d
