from collections import defaultdict
import itertools
import math
import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
from models.components import MultiNonLinearClassifier, SelfAttention
from models.functions import getPretrainedLMHead
from utils.Focal_Loss import focal_loss
# from xerrors import cprint

from transformers.models.bert.modeling_bert import BertAttention, BertOutput, BertIntermediate


class RelAttnLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attention = BertAttention(config=self.config.model)
        self.crossattention = BertAttention(config=self.config.model)
        self.intermediate = BertIntermediate(config=self.config.model)
        self.output = BertOutput(config=self.config.model)

    def forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask=None):

        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1).unsqueeze(1)

        # hidden_states = self.attention(hidden_states, attention_mask.unsqueeze(1))[0] # for head
        hidden_states = self.crossattention(hidden_states=hidden_states,
                                            encoder_hidden_states=encoder_hidden_states,
                                            encoder_attention_mask=encoder_attention_mask)[0]

        intermediate_output = self.intermediate(hidden_states)
        hidden_states = self.output(intermediate_output, hidden_states)

        return hidden_states

# class RelAttnModel(nn.Module):

#     def __init__(self, config) -> None:
#         super().__init__()
#         self.config = config
#         self.layers = nn.ModuleList([
#             RelAttnLayer(self.config)
#             for _ in range(config.get("rel_attn_layer_num", 1))])

#     def forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask=None):

#         for layer in self.layers:
#             hidden_states = layer(hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask)

#         return hidden_states

class RelDecoder(nn.Module):

    def __init__(self, theta):
        super().__init__()
        self.config = theta.config


        self.rel_ids = theta.rel_ids
        self.ent_ids = theta.ent_ids
        self.tag_size = len(self.rel_ids)

        self.attn = nn.ModuleList([RelAttnLayer(self.config) for _ in range(self.config.ent_attn_layer_num)])

    def forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask=None):

        if isinstance(encoder_hidden_states, list):
            encoder_hidden_states = encoder_hidden_states[1:]
        else:
            encoder_hidden_states = [encoder_hidden_states] * len(self.attn)
        assert len(self.attn) == len(encoder_hidden_states), "encoder_hidden_states 长度不对"

        out = [hidden_states]
        for li, layer in enumerate(self.attn):
            hidden_states = layer(hidden_states, attention_mask, encoder_hidden_states[li], encoder_attention_mask)
            out.append(hidden_states)

        return out

class REModel(pl.LightningModule):

    def __init__(self, theta):
        super().__init__()
        self.log = theta.log
        self.config = theta.config
        self.rel_ids = theta.rel_ids
        self.ent_ids = theta.ent_ids

        config = self.config
        hidden_size = config.model.hidden_size

        rel_hidden_dim_in = hidden_size * 3
        if self.config.use_rel_mention or self.config.use_filter_in_rel:
            rel_hidden_dim_in = hidden_size * 4

        self.classifier = MultiNonLinearClassifier(
                            hidden_size=rel_hidden_dim_in,
                            tag_size=len(self.rel_ids),
                            layers_num=self.config.rel_mlp_layer_num)

        if self.config.use_rel_attn:
            self.decoder = RelDecoder(theta)

        if self.config.use_length_embedding:
            self.length_embedding = theta.length_embedding

        self.grt_count = 0
        self.hit_count = 0
        self.rel_count = 0
        self.cur_epoch = 0
        self.pre_mode = None
        self.filter_score = {} # p, r, f1
        self.rel_type_num = len(self.config.dataset.rels)
        self.ent_type_num = len(self.config.dataset.ents)
        self.entids2tag = {v: k for k, v in enumerate(self.ent_ids)}

        self.remain = 0
        self.total = 0
        self.remain_val = 0
        self.total_val = 0
        self.remain_gold = 0
        self.total_gold = 0

        self.statistic = defaultdict(float)

        self.loss_weight = torch.FloatTensor([config.get("na_rel_weight", 1)] + [1] * self.rel_type_num)

    def get_triples_label(self, triples, device, ent_groups, mode, epoch):
        """Get the labels of triples.

        Args:
            return_hit_rate: whether to return the hit rate of the triples.
        """

        triple_labels = torch.zeros(len(ent_groups), device=device, dtype=torch.long)

        if triples is not None and len(triples) == 0:
            return triple_labels

        for i, pair in enumerate(ent_groups):
            b = pair[0]
            # t: [sub_start, sub_end, obj_start, obj_end, rel, sub_type, obj_type]
            for t in triples[b]:
                if t[-1] == -1:
                    break

                if t[:4].tolist() == pair[1:5]:
                    triple_labels[i] = t[4] + 1
                    break

        gt_count = 0
        for triple in triples:
            for t in triple:
                if t[-1] == -1:
                    break
                gt_count += 1

        hit_count = (triple_labels != 0).sum().item()
        rel_count = len(triple_labels)

        if mode != "train" and mode == self.pre_mode:
            self.grt_count += gt_count
            self.hit_count += hit_count
            self.rel_count += rel_count

        return triple_labels

    def prepare(self, theta, hidden_state, batch, gold_entities=None, pred_entities=None, mode="train", return_loss=False):
        """Get hidden state for the 2nd stage: relation classification"""

        if mode == "train" or return_loss:
            entities = gold_entities
        else:
            entities = pred_entities

        ent_ids = theta.ent_ids
        device = hidden_state.device

        # when in predict mode, only input_ids is not None, others are None
        input_ids, _, pos, triples, _, _, _ = batch

        bsz, seq_len = input_ids.shape

        val_thres = self.config.use_thres_threshold

        if mode != 'train':
            cur_threshold = val_thres
        elif mode == "train" and self.config.use_train_threshold:
            metrics = theta.filter.train_metrics.get("f1", 1e-3)
            cur_threshold = 10000 ** (np.log(metrics**3)) * 0.01
        else:
            cur_threshold = -1 # 等同于没有阈值

        logits, filter_loss, map_dict = theta.filter(hidden_state, entities, triples, mode, current_epoch=theta.current_epoch)

        if not self.config.dry_test:
            labels = theta.filter.get_filter_label(entities, triples, logits, map_dict)

            # statistic log with some bug
            if mode == "train":
                for i in range(self.rel_type_num+1):
                    self.statistic[f"train_gold_label_{i}_count"] += (labels == i).sum().item()
                self.statistic["train_gold_label_count"] += len(labels)
            elif return_loss:
                for i in range(self.rel_type_num+1):
                    self.statistic[f"val_gold_label_{i}_count"] += (labels == i).sum().item()
                self.statistic["val_gold_label_count"] += len(labels)
            else:
                for i in range(self.rel_type_num+1):
                    self.statistic[f"val_pred_label_{i}_count"] += (labels == i).sum().item()
                self.statistic["val_pred_label_count"] += len(labels)

        max_len= self.config.max_seq_len or 300 #  if mode == "train" else 1024
        ent_groups = []
        bio_tags = []
        rel_input_ids = []
        rel_positional_ids = []
        # rel_positional_id2 = []
        rel_attention_mask = []
        ent_hidden_states = []
        span_mentions = [] # for ner stage hiddden states

        cls_token = theta.tokenizer.cls_token_id
        sep_token = theta.tokenizer.sep_token_id
        pad_token = theta.tokenizer.pad_token_id
        mask_token = theta.tokenizer.mask_token_id

        # 获取实体的数量
        ent_num = 0
        for entity in entities:
            ent_num += len(entity)

        for b, entity in enumerate(entities):

            if pos is not None:
                sent_s, sent_e = pos[b, 0], pos[b, 1]
                sent_len = sent_e - sent_s
                if sent_len > max_len:
                    max_len = sent_len + 15 # for min 5 entities
            else:
                sent_len = (input_ids[b] != pad_token).sum().item()
                sent_s, sent_e = 0, sent_len

            ids = [cls_token] + input_ids[b, sent_s:sent_e].tolist() + [sep_token]
            pos_ids = [b for b in range(sent_len+2)]
            masks = [1 for b in range(sent_len+2)]

            bio_ids = np.array([theta.ent_ids[0]] * (sent_len+2))
            for e in entity:
                bio_ids[e[0]-sent_s+1] = theta.ent_ids[e[2] + 1]
                bio_ids[e[0]-sent_s+2:e[1]-sent_s+1] = theta.ent_ids[e[2] + self.ent_type_num + 1]

            bio_ids = bio_ids.tolist()

            # if self.config.use_re_marker and len(entity) > 0:
            #     entity = sorted(entity, key=lambda x: x[0])
            #     ids_with_marker = ids[:entity[0][0]-sent_s+1]
            #     pos_ids_with_marker = pos_ids[:entity[0][0]-sent_s+1]
            #     bio_ids_with_marker = bio_ids[:entity[0][0]-sent_s+1]
            #     for i, e in enumerate(entity):
            #         ids_with_marker += [theta.ent_ids[e[2] + 1]] + ids[e[0]-sent_s+1:e[1]-sent_s+1] + [theta.ent_ids[e[2] + self.ent_type_num + 1]]
            #         pos_ids_with_marker += [pos_ids[e[0]-sent_s+1]] + pos_ids[e[0]-sent_s+1:e[1]-sent_s+1] + [pos_ids[e[1]-sent_s]]
            #         bio_ids_with_marker += [theta.ent_ids[e[2] + 1]] + bio_ids[e[0]-sent_s+1:e[1]-sent_s+1] + [theta.ent_ids[e[2] + self.ent_type_num + 1]]
            #         if i < len(entity) - 1:
            #             ids_with_marker += ids[e[1]-sent_s+1:entity[i+1][0]-sent_s+1]
            #             pos_ids_with_marker += pos_ids[e[1]-sent_s+1:entity[i+1][0]-sent_s+1]
            #             bio_ids_with_marker += bio_ids[e[1]-sent_s+1:entity[i+1][0]-sent_s+1]
            #
            #     ids_with_marker += ids[entity[-1][1]-sent_s+1:]
            #     pos_ids_with_marker += pos_ids[entity[-1][1]-sent_s+1:]
            #     bio_ids_with_marker += bio_ids[entity[-1][1]-sent_s+1:]
            #
            #     masks = [1 for b in range(len(ids_with_marker))]
            #     ids = ids_with_marker
            #     pos_ids = pos_ids_with_marker
            #     bio_ids = bio_ids_with_marker

            if mode != "predict" and not self.config.dry_test:
                pred_draft_ent_groups = theta.filter.get_draft_ent_groups(entities, b, map_dict, logits, mode)
                gold_draft_ent_groups = theta.filter.get_draft_ent_groups(entities, b, map_dict, labels, mode, pred=logits)

                count = len(gold_draft_ent_groups)
                pred_count = len([1 for e in pred_draft_ent_groups if e[2] > self.config.get("alb_test1", 0.5)])
                if mode == "train" and self.config.use_thres_train:
                    r = theta.filter.train_metrics.get("recall", 0)
                    f1 = theta.filter.train_metrics.get("f1", 0)
                    ent_count = len(entity)
                    ent_count_half = int(np.round(ent_count / 2))
                    pair_count = len(gold_draft_ent_groups)
                    pair_count_gold = len([1 for e in gold_draft_ent_groups if e[2] != 0])
                    pred_count_pos = len([1 for e in pred_draft_ent_groups if e[2] > val_thres])
                    pred_count_dynamic = int(pair_count - pair_count * r + pred_count * r)
                    strategies = {
                        "default": max(ent_count, pred_count_dynamic),
                        "1231": max(ent_count_half, pred_count_dynamic),
                        "val_pos": max(pred_count_pos, pred_count_dynamic),
                        "x2": pair_count_gold * 2 + 1,
                        "reverse": pair_count_gold + pair_count * f1,
                        "x2min": max(pair_count_gold * 2 + 1, pred_count_dynamic),  # 相较于使用 ent_count 这种方式，直接使用标签的数量显然更加合理
                        "x2Ent": max(pair_count_gold * 2 + ent_count, pred_count_dynamic),  # 相较于使用 ent_count 这种方式，直接使用标签的数量显然更加合理
                        "x2Pro": max(int((pair_count_gold * 2 + ent_count) / 2) + ent_count, pred_count_dynamic),  # 相较于使用 ent_count 这种方式，直接使用标签的数量显然更加合理
                        "x3min": max(pair_count_gold * 3 + 3, pred_count_dynamic),  # 相较于使用 ent_count 这种方式，直接使用标签的数量显然更加合理
                        "x4min": max(pair_count_gold * 4 + 4, pred_count_dynamic),  # 相较于使用 ent_count 这种方式，直接使用标签的数量显然更加合理
                        "x8min": max(pair_count_gold * 8 + 8, pred_count_dynamic),  # 相较于使用 ent_count 这种方式，直接使用标签的数量显然更加合理
                        "x2more": max(pair_count_gold * 2 + 2, pred_count_dynamic),  # 相较于使用 ent_count 这种方式，直接使用标签的数量显然更加合理
                        "x8max": min(max(ent_count_half, pred_count_dynamic), pair_count_gold * 8 + 1),  # 限制最大的值，免得 loss 异常
                        "x2minx8max": min(max(pair_count_gold * 2 + 1, pred_count_dynamic), pair_count_gold * 8 + 8), # 对照实验
                    }
                    count = int(strategies[self.config.use_filter_strategy or "default"])

                if mode == "train":
                    draft_ent_groups = gold_draft_ent_groups[:count]
                    if self.config.use_negative and self.config.use_negative.endswith("+") and mode == "train" and False:
                        pair_count = int((max_len - len(ids)) / 3 + 1)
                        draft_ent_groups = draft_ent_groups[:pair_count]
                        np.random.shuffle(draft_ent_groups)
                else:
                    draft_ent_groups = pred_draft_ent_groups

                # if self.config.use_shuffle_pairs and mode == "train":
                #     np.random.shuffle(draft_ent_groups)

            else:
                draft_ent_groups = theta.filter.get_draft_ent_groups(entities, b, map_dict, logits, mode)

            # if mode == "train" and return_loss:
            #     self.remain += len(draft_ent_groups)
            #     self.total += len(gold_draft_ent_groups)
            # elif mode == "dev" or mode == "test":
            #     self.remain_val += len(draft_ent_groups)
            #     self.total_val += len(gold_draft_ent_groups)

            marker_mask = 1
            for ent_pair in draft_ent_groups:
                sub_s, sub_e, sub_t = ent_pair[0][:3]
                obj_s, obj_e, obj_t = ent_pair[1][:3]
                score = ent_pair[2]

                if mode == "train":
                    filter_socre = ent_pair[3]
                else:
                    filter_socre = ent_pair[2]

                if mode == "test" and len(ids) + 5 > max_len:
                    max_len = len(ids) + 5
                    # print(f"实体对过多{len(ids)}, 超过最大长度{max_len}，已经扩展")

                if len(ids) + 5 <= max_len and filter_socre > cur_threshold:  # 当设置 filter_rate 为 0 时，仅包含正确的实体对
                    marker_mask += 1

                    ss_tid = theta.tag_ids[sub_t]
                    os_tid = theta.tag_ids[obj_t]
                    # se_tid = theta.tag_ids[sub_t + self.ent_type_num]
                    # oe_tid = theta.tag_ids[obj_t + self.ent_type_num]

                    mask_bio_id = theta.ent_ids[0]
                    ss_bio_id = theta.ent_ids[sub_t + 1]
                    os_bio_id = theta.ent_ids[obj_t + 1]
                    # se_bio_id = theta.ent_ids[sub_t + self.ent_type_num + 1]
                    # oe_bio_id = theta.ent_ids[obj_t + self.ent_type_num + 1]

                    ss_pid = sub_s - sent_s
                    os_pid = obj_s - sent_s
                    # se_pid = sub_e - sent_s + 2
                    # oe_pid = obj_e - sent_s + 2
                    if self.config.fix_pid:
                        ss_pid = sub_s - sent_s + 1
                        os_pid = obj_s - sent_s + 1
                        se_pid = sub_e - sent_s + 1
                        oe_pid = obj_e - sent_s + 1

                    mask_pos = os_pid

                    ids += [ss_tid, mask_token, os_tid]
                    pos_ids += [ss_pid, mask_pos, os_pid]
                    # pos_id2 += [mask_pos, se_pid, oe_pid]
                    bio_ids += [ss_bio_id, mask_bio_id, os_bio_id]
                    masks += [marker_mask] * 3

                    # if self.config.use_rel_opt1.endswith("+"):
                    #     ids += [se_tid, oe_tid]
                    #     pos_ids += [se_pid, oe_pid]
                    #     # pos_id2 += [se_pid, oe_pid]
                    #     bio_ids += [se_bio_id, oe_bio_id]
                    #     masks += [marker_mask] * 2

                    if self.config.use_rel_opt2 == "max":  # default head, options: head, tail, max, mean
                        raise NotImplementedError(f"rel_opt2: {self.config.use_rel_opt2} not implemented")
                        ent_hidden_states.append(torch.stack([
                            hidden_state[b, sub_s:sub_e].max(dim=0)[0],
                            hidden_state[b, obj_s:obj_e].max(dim=0)[0]])) # 最大池化
                    elif self.config.use_rel_opt2 == "mean":
                        _sub = hidden_state[b, sub_s:sub_e].mean(dim=0)
                        _obj = hidden_state[b, obj_s:obj_e].mean(dim=0)

                        if self.config.use_length_embedding:
                            _sub = _sub + self.length_embedding(torch.tensor(sub_e - sub_s, device=hidden_state.device))
                            _obj = _obj + self.length_embedding(torch.tensor(obj_e - obj_s, device=hidden_state.device))

                        ent_hidden_states.append(torch.stack([_sub, _obj]))

                        if self.config.use_filter_in_rel == "span_mention":
                            span_s, span_e = min(sub_s, obj_s), max(sub_e, obj_e)
                            span_mention = hidden_state[b, span_s:span_e].mean(dim=0)
                            span_mentions.append(span_mention)

                    elif self.config.use_rel_opt2 == "head" or not self.config.use_rel_opt2: # default head
                        raise NotImplementedError(f"rel_opt2: {self.config.use_rel_opt2} not implemented")
                        ent_hidden_states.append(torch.stack([
                            hidden_state[b, sub_s],
                            hidden_state[b, obj_s]])) # 取第一个 token
                    else:
                        raise NotImplementedError(f"rel_opt2: {self.config.use_rel_opt2} not implemented")

                    ent_groups.append([b, sub_s, sub_e, obj_s, obj_e, sub_t, obj_t, score])

            rel_input_ids.append(torch.tensor(ids))
            rel_positional_ids.append(torch.tensor(pos_ids))
            # rel_positional_id2.append(torch.tensor(pos_id2))
            rel_attention_mask.append(torch.tensor(masks)) # 不要放到 cuda 上
            bio_tags.append(torch.tensor(bio_ids))

            if mode == "train":
                self.remain += marker_mask - 1
                self.total += len(gold_draft_ent_groups)
                self.statistic["train_gold_count"] += len(gold_draft_ent_groups)
                self.statistic["train_gold_filtered_count"] += len(draft_ent_groups)
                self.statistic["train_gold_use_count"] += marker_mask - 1
                self.statistic["train_pred_count"] += pred_count
            elif mode == "dev" or (mode == "test" and not self.config.dry_test):
                if return_loss:
                    self.remain_gold += marker_mask - 1
                    self.total_gold += len(pred_draft_ent_groups)
                    self.statistic["val_gold_count"] += len(pred_draft_ent_groups)
                    self.statistic["val_gold_filtered_count"] += len([1 for e in pred_draft_ent_groups if e[2] > cur_threshold])
                    self.statistic["val_gold_use_count"] += marker_mask - 1
                else:
                    self.remain_val += marker_mask - 1
                    self.total_val += len(pred_draft_ent_groups)
                    self.statistic["val_pred_count"] += len(pred_draft_ent_groups)
                    self.statistic["val_pred_filtered_count"] += len([1 for e in pred_draft_ent_groups if e[2] > cur_threshold])
                    self.statistic["val_pred_use_count"] += marker_mask - 1

        rel_input_ids = nn.utils.rnn.pad_sequence(rel_input_ids, batch_first=True, padding_value=pad_token)
        rel_positional_ids = nn.utils.rnn.pad_sequence(rel_positional_ids, batch_first=True, padding_value=0)
        rel_attention_mask_pad = nn.utils.rnn.pad_sequence(rel_attention_mask, batch_first=True, padding_value=0)
        # rel_positional_id2 = nn.utils.rnn.pad_sequence(rel_positional_id2, batch_first=True, padding_value=0)
        bio_tags = nn.utils.rnn.pad_sequence(bio_tags, batch_first=True, padding_value=theta.ent_ids[0])

        # 2D attention mask
        padding_length = rel_input_ids.shape[1]
        rel_attention_mask_matrix = torch.zeros([bsz, padding_length, padding_length])

        for b, m in enumerate(rel_attention_mask):
            cur_len = len(m)
            matrix = []
            # 这里的 m.tolist() 会比之前要好，在计算上面
            for from_mask in m.tolist():
                matrix_i = []
                for to_mask in m.tolist():
                    # 每组实体只能看到自己的标记和句子中的文本
                    if to_mask == 1 or from_mask == to_mask:
                        matrix_i.append(1)
                    else:
                        matrix_i.append(0)

                matrix.append(matrix_i)
            rel_attention_mask_matrix[b, :cur_len, :cur_len] = torch.tensor(matrix)

        rel_attention_mask = rel_attention_mask_matrix.clone()

        rel_input_ids = rel_input_ids.to(device)
        rel_positional_ids = rel_positional_ids.to(device)
        # rel_positional_id2 = rel_positional_id2.to(device)
        rel_attention_mask = rel_attention_mask.to(device)
        bio_tags = bio_tags.to(device)
        assert rel_positional_ids.max() <= 512 and rel_positional_ids.min() >= 0, "positional ids error"
        # assert rel_positional_id2.max() <= 512 and rel_positional_id2.min() >= 0, "positional ids error"
        assert rel_input_ids.shape == rel_positional_ids.shape

        rel_hidden_states = []
        sent_ner_loss = torch.tensor(0.0).to(hidden_state.device)

        if len(ent_groups) != 0:
            # 1. ent hidden state
            ent_hidden_states = torch.stack(ent_hidden_states) # [ent_num, 2, hidden_size]

            rel_inputs_embeds = self.bert_embeddings_with_bio_tag_embedding(
                theta.plm_model.embeddings.word_embeddings,
                bio_tags=bio_tags,
                input_ids=rel_input_ids)

            # if self.config.use_bio_embed:
            #     rel_inputs_embeds = self.bert_embeddings_with_bio_tag_embedding(
            #         theta.plm_model.embeddings.word_embeddings,
            #         bio_tags=bio_tags,
            #         input_ids=rel_input_ids)
            # else:
            #     rel_inputs_embeds = None
            #     assert rel_input_ids is not None

            # 2. 重新计算 hidden state

            plm_model = theta.plm_model_for_re if self.config.use_two_plm else theta.plm_model
            if self.config.use_rel_prompt:
                batch_size = input_ids.shape[0]
                past_key_values = theta.get_prompt(stage="rel", batch_size=batch_size)
                prefix_attention_mask = torch.ones(batch_size, rel_attention_mask.shape[-1], theta.prompt_len).to(plm_model.device)
                rel_attention_mask = torch.cat((prefix_attention_mask, rel_attention_mask), dim=2)
            else:
                past_key_values = None

            outputs = plm_model(
                        input_ids=None, # rel_input_ids if not self.config.use_bio_embed else None, # torch.Size([8, 607])
                        inputs_embeds=rel_inputs_embeds,
                        attention_mask=rel_attention_mask, # torch.Size([8, 607, 607])
                        position_ids=rel_positional_ids, # torch.Size([8, 607]),
                        token_type_ids = torch.zeros(rel_input_ids.shape[:2], dtype=torch.long, device=device),
                        output_hidden_states=True,
                        past_key_values=past_key_values)
            rel_stage_hs = outputs.hidden_states[-1]

            # attention
            sent_mask = torch.where(rel_attention_mask_pad == 1, 1.0, 0.0).to(device)
            if self.config.use_rel_attn and False:
                max_sent_len = (rel_attention_mask_pad == 1).sum(dim=-1).max().item()
                sent_mask_attn = sent_mask[:, :max_sent_len]
                rel_sent_hs = rel_stage_hs * (rel_attention_mask_pad == 1).unsqueeze(-1).to(device)
                rel_sent_hs = rel_sent_hs[:, :max_sent_len, :]

                rel_triplet_hs = []
                rel_triplet_mask = []
                for bi in range(bsz):
                    rel_triplet_mask.append(rel_attention_mask_pad[bi, rel_attention_mask_pad[bi] > 1])
                    rel_triplet_hs.append(rel_stage_hs[bi, rel_attention_mask_pad[bi] > 1])

                rel_triplet_hs = nn.utils.rnn.pad_sequence(rel_triplet_hs, batch_first=True, padding_value=0)
                rel_triplet_mask = nn.utils.rnn.pad_sequence(rel_triplet_mask, batch_first=True, padding_value=0)

                mask_2d = rel_triplet_mask.unsqueeze(-1) * rel_triplet_mask.unsqueeze(-2)
                mask_mask = (rel_triplet_mask ** 2).unsqueeze(1)
                mask_2d_pos = (mask_2d > 0)
                rel_triplet_mask_2d = torch.where(mask_2d == mask_mask, 1.0, 0.0)
                rel_triplet_mask_2d = (rel_triplet_mask_2d * mask_2d_pos.float()).to(device)

                if self.config.use_rel_ner:
                    ner_labels = (bio_tags - theta.ent_ids[0])[:, :max_sent_len]
                    logits, sent_ner_loss, ner_hs = theta.ner_model(rel_sent_hs, labels=ner_labels, mask=sent_mask_attn, return_hs=True)
                    encoder_hidden_states = ner_hs
                else:
                    encoder_hidden_states = rel_sent_hs

                rel_triplet_hs = self.decoder(hidden_states=rel_triplet_hs,
                                                attention_mask=rel_triplet_mask_2d,
                                                encoder_hidden_states=encoder_hidden_states,
                                                encoder_attention_mask=sent_mask_attn)[-1]

                mask_pos = torch.where(rel_triplet_mask > 0)
                hs = rel_triplet_hs[mask_pos[0], mask_pos[1]] # copilot NewBee
                rel_hidden_states = hs[::3]
                sub_tag_hs = hs[1::3]
                obj_tag_hs = hs[2::3]

            else:
                # 找到 mask 的 Hidden State
                mask_pos = torch.where(rel_input_ids == theta.tokenizer.mask_token_id)
                rel_hidden_states = rel_stage_hs[mask_pos[0], mask_pos[1]] # copilot NewBee

                # sub_tag_hs = rel_stage_hs[mask_pos[0], mask_pos[1]+1]
                # obj_tag_hs = rel_stage_hs[mask_pos[0], mask_pos[1]+2]

                sub_tag_hs = rel_stage_hs[mask_pos[0], mask_pos[1]-1]
                obj_tag_hs = rel_stage_hs[mask_pos[0], mask_pos[1]+1]

                if self.config.use_rel_mention:
                    mentions = []
                    for _bi, _index in zip(mask_pos[0], mask_pos[1]):
                        sub_pos_id = rel_positional_ids[_bi, _index-1]
                        obj_pos_id = rel_positional_ids[_bi, _index+1]
                        mention_s = min(sub_pos_id, obj_pos_id)
                        mention_e = max(sub_pos_id, obj_pos_id)
                        mentions.append(rel_stage_hs[_bi, mention_s:mention_e+1].mean(dim=0))

                    mentions = torch.stack(mentions).to(device)

                    # 利用 sub obj 的对应来获取，目前的版本会重复获取 subject 或者 object

                if self.config.use_rel_ner and not self.config.dry_test:
                    if self.config.use_rel_ner == "no_mask":
                        sent_mask = torch.where(rel_attention_mask_pad > 0, 1, 0).to(device)
                    else:
                        sent_mask = torch.where(rel_attention_mask_pad == 1, 1, 0).to(device)

                    bio_labels = torch.tensor([[self.entids2tag[bbb] for bbb in bb] for bb in bio_tags.tolist()]).to(device)
                    logits, sent_ner_loss = theta.ner_model(rel_stage_hs, labels=bio_labels, mask=sent_mask, mode=mode)

            # if self.config.use_rel_opt1 is str and self.config.use_rel_opt1.endswith("+"):
            #     sub_tag_hs = rel_stage_hs[mask_pos[0], mask_pos[1]+1] + rel_stage_hs[mask_pos[0], mask_pos[1]+3]
            #     obj_tag_hs = rel_stage_hs[mask_pos[0], mask_pos[1]+2] + rel_stage_hs[mask_pos[0], mask_pos[1]+4]
            # else:
            #     sub_tag_hs = rel_stage_hs[mask_pos[0], mask_pos[1]+1]
            #     obj_tag_hs = rel_stage_hs[mask_pos[0], mask_pos[1]+2]

            # sub_tag_hs = rel_stage_hs[mask_pos[0], mask_pos[1]+1]
            # obj_tag_hs = rel_stage_hs[mask_pos[0], mask_pos[1]+2]

            # if self.config.use_rel_ner:
            #     sent_hs = rel_stage_hs
            #     sent_labels = bio_tags - theta.ent_ids[0]
            #     if self.config.use_rel_ner == "no_mask":
            #         sent_mask = torch.where(rel_attention_mask_pad > 0, 1.0, 0.0).to(device)
            #     # elif self.config.use_rel_ner == "attn":
            #     #     sent_hs = rel_sent_hs
            #     #     sent_mask = sent_mask_attn
            #     #     sent_labels = sent_labels[:, :max_sent_len]

            #     logits, sent_ner_loss = theta.ner_model(sent_hs, labels=sent_labels, mask=sent_mask)

            # if self.config.use_rel_state_hs:
            #     sub_pos_ids = rel_positional_ids[mask_pos[0], mask_pos[1]+1]
            #     obj_pos_ids = rel_positional_ids[mask_pos[0], mask_pos[1]+2]
            #     sub_rel_hs = rel_stage_hs[mask_pos[0], sub_pos_ids]
            #     obj_rel_hs = rel_stage_hs[mask_pos[0], obj_pos_ids]
            #     sub_hs = sub_rel_hs + sub_tag_hs
            #     obj_hs = obj_rel_hs + obj_tag_hs
            # else:
            #     sub_ent_hs = ent_hidden_states[:, 0, :]
            #     obj_ent_hs = ent_hidden_states[:, 1, :]
            #     sub_hs = sub_ent_hs + sub_tag_hs
            #     obj_hs = obj_ent_hs + obj_tag_hs

            sub_hs = ent_hidden_states[:, 0, :] + sub_tag_hs
            obj_hs = ent_hidden_states[:, 1, :] + obj_tag_hs

            if self.config.remove_ner_info:  # default false
                sub_hs = sub_tag_hs
                obj_hs = obj_tag_hs

            rel_hidden_states = torch.cat([rel_hidden_states, sub_hs, obj_hs], dim=-1)
            if self.config.use_rel_mention and False:
                rel_hidden_states = torch.cat([rel_hidden_states, mentions], dim=-1)

            if self.config.use_filter_in_rel:
                sub_pair_proj = theta.filter.sub_proj(ent_hidden_states[:, 0, :])
                obj_pair_proj = theta.filter.obj_proj(ent_hidden_states[:, 1, :])
                ent_pair_hs = torch.cat([sub_pair_proj, obj_pair_proj], dim=-1)

                if self.config.use_filter_in_rel == "span_mention":
                    span_mentions = torch.stack(span_mentions)
                    ent_pair_hs = torch.cat([ent_pair_hs, span_mentions], dim=-1)

                ent_pair_hs = theta.filter.filter_entity_pair_net(ent_pair_hs)[1]
                rel_hidden_states = torch.cat([rel_hidden_states, ent_pair_hs], dim=-1)

            # if self.config.use_rel_opt3 == "tag":
            #     sub_hs = sub_tag_hs
            #     obj_hs = obj_tag_hs

            # elif self.config.use_rel_opt3 == "embed":
            #     sub_pos_ids = rel_positional_ids[mask_pos[0], mask_pos[1]+1]
            #     obj_pos_ids = rel_positional_ids[mask_pos[0], mask_pos[1]+2]
            #     sub_hs = rel_stage_hs[mask_pos[0], sub_pos_ids]
            #     obj_hs = rel_stage_hs[mask_pos[0], obj_pos_ids]
            #     sub_hs += sub_tag_hs
            #     obj_hs += obj_tag_hs

            # elif self.config.use_rel_opt3 == "embed2":
            #     sub_hs = ent_hidden_states[:, 0, :] + sub_tag_hs
            #     obj_hs = ent_hidden_states[:, 1, :] + obj_tag_hs

            # elif self.config.use_rel_opt3 == "embed3":
            #     ss_pos_ids = rel_positional_ids[mask_pos[0], mask_pos[1]+1]
            #     os_pos_ids = rel_positional_ids[mask_pos[0], mask_pos[1]+2]
            #     se_pos_ids = rel_positional_id2[mask_pos[0], mask_pos[1]+1]
            #     oe_pos_ids = rel_positional_id2[mask_pos[0], mask_pos[1]+2]
            #     sub_hs = rel_stage_hs[mask_pos[0], ss_pos_ids] + rel_stage_hs[mask_pos[0], se_pos_ids]
            #     obj_hs = rel_stage_hs[mask_pos[0], os_pos_ids] + rel_stage_hs[mask_pos[0], oe_pos_ids]
            #     sub_hs += sub_tag_hs
            #     obj_hs += obj_tag_hs

            # else:
            #     raise NotImplementedError(f"rel_opt3: {self.config.use_rel_opt3} not implemented")

            # if self.config.use_rel == 'mlp':
            #     rel_hidden_states = torch.cat([rel_hidden_states, sub_hs, obj_hs], dim=-1)
            # else:
            #     rel_hidden_states += sub_hs - obj_hs

            # if self.config.use_rel_tag_cross_attn:
            #     tag_embeddings = theta.get_rel_tag_embeddings(with_na=True, device=device).unsqueeze(0) # [1, 1, ent_num, hidden_size]
            #     rel_hidden_states = self.downscale(rel_hidden_states)
            #     attn_out = self.cross_attn(rel_hidden_states.unsqueeze(0), tag_embeddings, tag_embeddings)[0]
            #     rel_hidden_states = self.layer_norm(rel_hidden_states + attn_out).squeeze(0)

        if mode != 'predict' and not self.config.dry_test:
            triple_labels = self.get_triples_label(triples, device, ent_groups, mode=mode, epoch=theta.current_epoch)
            assert len(ent_groups) == len(rel_hidden_states) == len(triple_labels)

            if mode == "train":
                for li in range(self.rel_type_num+1):
                    self.statistic[f"train_gold_label_{li}_filtered_count"] += (triple_labels == li).sum().item()
                self.statistic["train_gold_label_filtered_count"] += len(triple_labels)
            elif return_loss:
                for li in range(self.rel_type_num+1):
                    self.statistic[f"val_gold_label_{li}_filtered_count"] += (triple_labels == li).sum().item()
                self.statistic["val_gold_label_filtered_count"] += len(triple_labels)
            else:
                for li in range(self.rel_type_num+1):
                    self.statistic[f"val_pred_label_{li}_filtered_count"] += (triple_labels == li).sum().item()
                self.statistic["val_pred_label_filtered_count"] += len(triple_labels)

        else:
            triple_labels = None

        # semantic_loss = torch.tensor(0.0).to(hidden_state.device)
        # if self.config.use_semantic_loss and len(ent_groups) != 0:
        #     if self.config.model.hidden_size != rel_hidden_states.shape[-1]:
        #         rel_hidden_states = self.downscale(rel_hidden_states)
        #         assert self.config.model.hidden_size == rel_hidden_states.shape[-1]

        #     if self.config.use_semantic_loss == "cos":
        #         rel_embeddings = theta.plm_model.get_input_embeddings().weight[triple_labels + theta.rel_ids[0]]
        #         semantic_loss = nn.CosineEmbeddingLoss(reduction='mean')
        #         semantic_loss = semantic_loss(rel_hidden_states, rel_embeddings, torch.ones(len(ent_groups), device=device))
        #     elif self.config.use_semantic_loss == "ce":
        #         rel_embeddings = theta.get_rel_tag_embeddings(with_na=True, device=device).unsqueeze(0).repeat(len(rel_hidden_states), 1, 1)
        #         semantic_logits = torch.einsum("bd,bed->be", rel_hidden_states, rel_embeddings)
        #         semantic_loss = nn.CrossEntropyLoss(reduction='mean')(semantic_logits, triple_labels)
        #     else:
        #         raise NotImplementedError(f"semantic loss {self.config.use_semantic_loss} not implemented")

        return ent_groups, rel_hidden_states, triple_labels, filter_loss, sent_ner_loss

    def forward(self, theta, batch, hidden_state, gold_entities=None, pred_entities=None, return_loss=False, mode="train", with_score=False):

        output = self.prepare(
                    theta=theta,
                    batch=batch,
                    hidden_state=hidden_state,
                    gold_entities=gold_entities,
                    pred_entities=pred_entities,
                    mode=mode,
                    return_loss=return_loss)


        '''
        这里的 hit_rate 是指过滤后，正确的候选实体对与真实实体对的数量之比
        所体现的是关系抽取和实体过滤工作作用下的结果
        最终的 F1 在 PR 相差不大的情况下，可以理解为 hit_rate * rel_f1
        '''
        # if mode != 'train' and mode != 'predict' and (mode != self.pre_mode or mode == 'test'):
        #     self.filter_score["precision"] = self.hit_count / (self.rel_count + 1e-8)
        #     self.filter_score["recall"] = self.hit_count / (self.grt_count + 1e-8)
        #     self.filter_score["f1"] = 2 * self.hit_count / (self.grt_count + self.rel_count + 1e-8)
        #     theta.log("info/pair_precision", self.filter_score["precision"])
        #     theta.log("info/pair_recall", self.filter_score["recall"])
        #     theta.log("info/pair_f1", self.filter_score["f1"])
        #     self.hit_count = 0
        #     self.grt_count = 0
        #     self.rel_count = 0

        self.pre_mode = mode

        ent_groups, hidden_output, triple_labels, filter_loss = output[:4]
        sent_ner_loss = output[4] if len(output) > 4 else None

        if len(hidden_output) == 0:
            rel_loss = torch.tensor(0.0).to(hidden_state.device)
            return ([],) if not return_loss else ([], rel_loss, filter_loss, sent_ner_loss)

        # if theta.graph is not None and mode != 'predict':
        #     hidden_output = theta.graph.query_rels(hidden_output)

        if self.config.use_rel == 'lmhead':
            assert self.lmhead is not None
            logits = self.lmhead(hidden_output)
            logits = logits[..., self.rel_ids] # python Ellipsis operator # BUG
        else:
            logits = self.classifier(hidden_output)

        triples_pred = []
        relation_logits = logits.argmax(dim=-1)
        for i in range(len(ent_groups)):
            rel = relation_logits[i].item()
            if with_score:
                triples_pred.append(ent_groups[i] + [rel, logits[i]])
            else:
                triples_pred.append(ent_groups[i] + [rel])

            # if relation_logits[i] > 0 and theta.graph:
            #     rel_embeddings = hidden_output[i].detach().clone()
            #     theta.graph.add_edge(sub=ent_groups[i][0], obj=ent_groups[i][1], rel_type=rel-1, embedding=rel_embeddings)

        # triples_pred = [ent_groups[i] + [rel] for i in range(len(ent_groups))]

        rel_loss = torch.tensor(0.0).to(logits.device)
        if triple_labels is not None and return_loss:
            new_logits = logits.view(-1, len(self.rel_ids))
            new_labels = triple_labels.view(-1)

            if self.config.use_rel_na_warmup: # 5

                # fre previous version
                if "_" not in str(self.config.use_rel_na_warmup):
                    self.config.use_rel_na_warmup = "sin_" + str(self.config.use_rel_na_warmup)

                func, warmup_epochs = self.config.use_rel_na_warmup.split("_")
                if func == "sin":
                    sin_warm = lambda x: np.sin((min(1, x + 0.001) * np.pi / 2))
                    warmup_rate = sin_warm(theta.current_epoch / int(warmup_epochs))
                elif func == "cos":
                    cos_warm = lambda x: 1 - np.cos((min(1, x + 0.001) * np.pi / 2))
                    warmup_rate = cos_warm(theta.current_epoch / int(warmup_epochs))
                else:
                    raise NotImplementedError(f"func: {func} not implemented")

                self.loss_weight[0] = float(self.config.get("na_rel_weight", 1)) * warmup_rate

            if self.config.use_rel_focal_loss: # False
                loss_fct = focal_loss(alpha=None, gamma=2, num_classes=len(self.rel_ids))
                rel_loss = loss_fct(new_logits, new_labels)

            elif self.config.use_rel_loss_sum:

                if self.config.use_dynamic_loss_sum:
                    scale_rate = int(self.config.use_rel_loss_sum) * (self.statistic["last_filter_rate"] or 1)
                else:
                    scale_rate = int(self.config.use_rel_loss_sum)

                assert scale_rate > 0, "use_rel_loss_sum 参数错误"
                loss_fct = nn.CrossEntropyLoss(reduction='sum', weight=self.loss_weight.to(logits.device))
                rel_loss = loss_fct(new_logits, new_labels) / scale_rate / self.config.batch_size * 16
            else:
                loss_fct = nn.CrossEntropyLoss(reduction='mean', weight=self.loss_weight.to(logits.device))
                rel_loss = loss_fct(new_logits, new_labels)

            return (triples_pred, rel_loss, filter_loss, sent_ner_loss)

        if mode == "predict":
            return (triples_pred, ent_groups)

        return (triples_pred,)

    def log_statistic_train(self):
        self.log("statistic/train_gold_count", self.statistic["train_gold_count"])
        self.log("statistic/train_gold_filtered_count", self.statistic["train_gold_filtered_count"])
        self.log("statistic/train_gold_use_count", self.statistic["train_gold_use_count"])
        self.log("statistic/train_pred_count", self.statistic["train_pred_count"])
        self.log("statistic/train_filter_rate", self.statistic["train_gold_filtered_count"] / (self.statistic["train_gold_count"] + 1e-8))
        self.log("statistic/train_filter_rate_plus", self.statistic["train_gold_use_count"] / (self.statistic["train_gold_count"] + 1e-8))
        self.log("statistic/train_filter_rate_pred", self.statistic["train_pred_count"] / (self.statistic["train_gold_count"] + 1e-8))
        self.statistic["last_filter_rate"] = self.statistic["train_gold_filtered_count"] / (self.statistic["train_gold_count"] + 1e-8)
        self.statistic["train_gold_count"] = 0.0
        self.statistic["train_gold_filtered_count"] = 0.0
        self.statistic["train_gold_use_count"] = 0.0
        self.statistic["train_pred_count"] = 0.0

        self.log("statistic/train_gold_label_count", self.statistic["train_gold_label_count"])
        self.log("statistic/train_gold_label_filtered_count", self.statistic["train_gold_label_filtered_count"])
        for i in range(self.rel_type_num+1):
            self.log(f"statistic/train_gold_label_{i}_count", self.statistic[f"train_gold_label_{i}_count"])
            self.log(f"statistic/train_gold_label_{i}_filtered_count", self.statistic[f"train_gold_label_{i}_filtered_count"])
            self.log(f"statistic/train_gold_label_{i}_rate", self.statistic[f"train_gold_label_{i}_count"] / (self.statistic["train_gold_label_count"] + 1e-8))
            self.log(f"statistic/train_gold_label_{i}_filtered_rate", self.statistic[f"train_gold_label_{i}_filtered_count"] / (self.statistic["train_gold_label_filtered_count"] + 1e-8))
            self.statistic[f"train_gold_label_{i}_filtered_count"] = 0.0
            self.statistic[f"train_gold_label_{i}_count"] = 0.0

        self.statistic["train_gold_label_count"] = 0.0
        self.statistic["train_gold_label_filtered_count"] = 0.0

    def log_statistic_val(self):
        self.log("statistic/val_gold_count", self.statistic["val_gold_count"])
        self.log("statistic/val_gold_filtered_count", self.statistic["val_gold_filtered_count"])
        self.log("statistic/val_gold_use_count", self.statistic["val_gold_use_count"])
        self.log("statistic/val_pred_count", self.statistic["val_pred_count"])
        self.log("statistic/val_pred_filtered_count", self.statistic["val_pred_filtered_count"])
        self.log("statistic/val_pred_use_count", self.statistic["val_pred_use_count"])
        self.log("statistic/val_gold_filter_rate", self.statistic["val_gold_filtered_count"] / (self.statistic["val_gold_count"] + 1e-8))
        self.log("statistic/val_gold_filter_rate_plus", self.statistic["val_gold_use_count"] / (self.statistic["val_gold_count"] + 1e-8))
        self.log("statistic/val_pred_filter_rate", self.statistic["val_pred_filtered_count"] / (self.statistic["val_pred_count"] + 1e-8))
        self.log("statistic/val_pred_filter_rate_plus", self.statistic["val_pred_use_count"] / (self.statistic["val_pred_count"] + 1e-8))
        self.statistic["val_gold_count"] = 0.0
        self.statistic["val_gold_filtered_count"] = 0.0
        self.statistic["val_gold_use_count"] = 0.0
        self.statistic["val_pred_count"] = 0.0
        self.statistic["val_pred_filtered_count"] = 0.0
        self.statistic["val_pred_use_count"] = 0.0

        self.log("statistic/val_gold_label_count", self.statistic["val_gold_label_count"])
        self.log("statistic/val_gold_label_filtered_count", self.statistic["val_gold_label_filtered_count"])
        self.log("statistic/val_pred_label_count", self.statistic["val_pred_label_count"])
        self.log("statistic/val_pred_label_filtered_count", self.statistic["val_pred_label_filtered_count"])
        for i in range(self.rel_type_num+1):
            self.log(f"statistic/val_gold_label_{i}_count", self.statistic[f"val_gold_label_{i}_count"])
            self.log(f"statistic/val_gold_label_{i}_filtered_count", self.statistic[f"val_gold_label_{i}_filtered_count"])
            self.log(f"statistic/val_pred_label_{i}_count", self.statistic[f"val_pred_label_{i}_count"])
            self.log(f"statistic/val_pred_label_{i}_filtered_count", self.statistic[f"val_pred_label_{i}_filtered_count"])
            self.log(f"statistic/val_gold_label_{i}_rate", self.statistic[f"val_gold_label_{i}_count"] / (self.statistic["val_gold_label_count"] + 1e-8))
            self.log(f"statistic/val_gold_label_{i}_filtered_rate", self.statistic[f"val_gold_label_{i}_filtered_count"] / (self.statistic["val_gold_label_filtered_count"] + 1e-8))
            self.log(f"statistic/val_pred_label_{i}_rate", self.statistic[f"val_pred_label_{i}_count"] / (self.statistic["val_pred_label_count"] + 1e-8))
            self.log(f"statistic/val_pred_label_{i}_filtered_rate", self.statistic[f"val_pred_label_{i}_filtered_count"] / (self.statistic["val_pred_label_filtered_count"] + 1e-8))
            self.statistic[f"val_gold_label_{i}_count"] = 0.0
            self.statistic[f"val_gold_label_{i}_filtered_count"] = 0.0
            self.statistic[f"val_pred_label_{i}_count"] = 0.0
            self.statistic[f"val_pred_label_{i}_filtered_count"] = 0.0

        self.statistic["val_gold_label_count"] = 0.0
        self.statistic["val_gold_label_filtered_count"] = 0.0
        self.statistic["val_pred_label_count"] = 0.0
        self.statistic["val_pred_label_filtered_count"] = 0.0


    def log_ent_pair_info(self):
        self.filter_score["precision"] = self.hit_count / (self.rel_count + 1e-8)
        self.filter_score["recall"] = self.hit_count / (self.grt_count + 1e-8)
        self.filter_score["f1"] = 2 * self.hit_count / (self.grt_count + self.rel_count + 1e-8)
        self.log("pair/f1", self.filter_score["f1"])
        self.log("pair/precision", self.filter_score["precision"])
        self.log("pair/recall", self.filter_score["recall"])
        self.hit_count = 0
        self.grt_count = 0
        self.rel_count = 0

    def log_filter_rate(self):
        self.log("filter/rate", self.remain / (self.total + 1e-8))
        self.remain = 0
        self.total = 0

    def log_filter_rate_val(self):
        self.log("filter/rate_val", self.remain_val / (self.total_val + 1e-8))
        self.log("filter/rate_gold_ent", self.remain_gold / (self.total_gold + 1e-8))
        self.remain_val = 0
        self.total_val = 0
        self.remain_gold = 0
        self.total_gold = 0

    # def prepare_one_sent(self, theta, batch, hidden_state, entities, mode):

    #     ent_ids = theta.ent_ids
    #     device = hidden_state.device
    #     input_ids, _, pos, triples, _, _, _ = batch
    #     bsz, seq_len = input_ids.shape

    #     ent_groups = []
    #     rel_hidden_states = []

    #     cls_token = theta.tokenizer.cls_token_id
    #     sep_token = theta.tokenizer.sep_token_id
    #     pad_token = theta.tokenizer.pad_token_id
    #     mask_token = theta.tokenizer.mask_token_id

    #     # 先构建一个初始的实体组，然后按照置信度排序，能排多少是多少
    #     logits, filter_loss, map_dict = theta.filter(hidden_state, entities, triples, mode)
    #     logits = logits.sigmoid()

    #     if mode == "train" or self.config.use_gold_filter_val or self.config.filter_rate == 0:
    #         logits = theta.filter.get_filter_label(entities, triples, logits, map_dict)

    #     plm_model = theta.plm_model_for_re if self.config.use_two_plm else theta.plm_model

    #     batches = []
    #     for b, entity in enumerate(entities):

    #         sent_s, sent_e = pos[b, 0], pos[b, 1]
    #         sent_len = sent_e - sent_s

    #         ids = [cls_token] + input_ids[b, sent_s:sent_e].tolist() + [sep_token]
    #         pos_ids = [b for b in range(sent_len+2)]
    #         masks = [1 for b in range(sent_len+2)]

    #         marker_mask = 1
    #         draft_ent_groups = theta.filter.get_draft_ent_groups(entities, b, map_dict, logits, mode)
    #         for i, ent_pair in enumerate(draft_ent_groups):
    #             (sub_s, sub_e, sub_t), (obj_s, obj_e, obj_t), score = ent_pair

    #             ss_tid = theta.tag_ids[sub_t]
    #             os_tid = theta.tag_ids[obj_t]
    #             ss_pid = sub_s - sent_s
    #             os_pid = obj_s - sent_s

    #             if len(ids) + 3 > 512:
    #                 batches.append((ids, pos_ids, masks))
    #                 if len(batches) == 4:
    #                     mask_hs = self.calc_mask_hs(theta, plm_model, batches) # copilot NewBee
    #                     rel_hidden_states = torch.cat([rel_hidden_states, mask_hs], dim=0) if len(rel_hidden_states) > 0 else mask_hs
    #                     batches = []

    #                 ids = [cls_token] + input_ids[b, sent_s:sent_e].tolist() + [sep_token]
    #                 pos_ids = [b for b in range(sent_len+2)]
    #                 masks = [1 for b in range(sent_len+2)]
    #                 marker_mask = 1

    #             marker_mask += 1
    #             ids += [mask_token, ss_tid, os_tid]
    #             pos_ids += [os_pid, ss_pid, os_pid]
    #             masks += [marker_mask] * 3

    #             if i == len(draft_ent_groups) - 1:
    #                 batches.append((ids, pos_ids, masks))

    #             ent_g = [b, sub_s, sub_e, obj_s, obj_e, sub_t, obj_t]
    #             ent_groups.append(ent_g)

    #     if len(batches) != 0:
    #         mask_hs = self.calc_mask_hs(theta, plm_model, batches)
    #         rel_hidden_states = torch.cat([rel_hidden_states, mask_hs], dim=0) if len(rel_hidden_states) > 0 else mask_hs # type: ignore

    #     # 从 triples 中构建标签
    #     triple_labels = self.get_triples_label(triples, device, ent_groups, mode=mode, epoch=theta.current_epoch)

    #     assert len(ent_groups) == len(rel_hidden_states) == len(triple_labels)
    #     return ent_groups, rel_hidden_states, triple_labels, filter_loss


    # def calc_mask_hs(self, theta, plm_model, batches):
    #     max_len = max([len(b[0]) for b in batches])
    #     bid, bpos, bmasks = [], [], []
    #     rel_attention_mask_matrix = torch.zeros([len(batches), max_len, max_len])
    #     for b, (b_ids, b_pos_ids, b_masks) in enumerate(batches):
    #         b_ids += [theta.tokenizer.pad_token_id] * (max_len - len(b_ids))
    #         b_pos_ids += [0] * (max_len - len(b_pos_ids))
    #         cur_len = len(b_masks)
    #         matrix = []
    #         # 这里的 m.tolist() 会比之前要好，在计算上面
    #         for from_mask in b_masks:
    #             matrix_i = []
    #             for to_mask in b_masks:
    #                 # 每组实体只能看到自己的标记和句子中的文本
    #                 if to_mask == 1 or from_mask == to_mask:
    #                     matrix_i.append(1)
    #                 else:
    #                     matrix_i.append(0)

    #             matrix.append(matrix_i)
    #         rel_attention_mask_matrix[b, :cur_len, :cur_len] = torch.tensor(matrix)

    #         bid.append(b_ids)
    #         bpos.append(b_pos_ids)
    #         bmasks.append(b_masks)

    #     bid = torch.tensor(bid).to(theta.device)
    #     bpos = torch.tensor(bpos).to(theta.device)
    #     rel_attention_mask_matrix = rel_attention_mask_matrix.to(theta.device)
    #     # bmasks = torch.tensor(bmasks).to(theta.device)

    #     outputs = plm_model(
    #                     bid,
    #                     attention_mask=rel_attention_mask_matrix,
    #                     position_ids=bpos,
    #                     output_hidden_states=True)

    #     rel_stage_hs = outputs.hidden_states[-1]
    #     mask_pos = torch.where(bid == theta.tokenizer.mask_token_id)
    #     mask_hs = rel_stage_hs[mask_pos[0], mask_pos[1]]
    #     return mask_hs


    def bert_embeddings_with_bio_tag_embedding(self,
        word_embeddings = None,
        bio_tags = None,
        input_ids = None,
    ) -> torch.Tensor:

        assert word_embeddings is not None
        assert input_ids is not None

        # [batch_size, seq_len, hidden_size]
        bert_embeddings = word_embeddings(input_ids)
        bio_tags_embeddings = word_embeddings(bio_tags)

        # [batch_size, seq_len, hidden_size]
        bert_embeddings = bert_embeddings + bio_tags_embeddings

        return bert_embeddings



def is_overlap(entity):
    for i in range(len(entity)):
        for j in range(i+1, len(entity)):
            if entity[i][1] > entity[j][0] and entity[i][0] < entity[j][1]:
                return True
    return False