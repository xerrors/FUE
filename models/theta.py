from collections import defaultdict
import pytorch_lightning as pl
import torch
import torch.nn as nn
from models.components import PrefixEncoder
# from models.batch_filter import batch_filter
from models.entity_pair_filter import FilterModel
# from models.pre_relation import PreREModel
from models.re_model import REModel
from models.ner_model import NERModel
# from models.runtime_graph import RuntimeGraph
from models.functions import getBertForMaskedLMClass

from data.utils import get_language_map_dict
# from models.span_ner_model import SpanEntityModel
from utils.metrics import f1_score
from utils.optimizers import get_optimizer


class Theta(pl.LightningModule):
    """https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html#pytorch_lightning.core.LightningModule"""

    def __init__(self, config, data):
        super().__init__()
        self.tag_ids = None
        self.ent_ids = None
        self.rel_ids = None
        self.config = config
        self.tokenizer = data.tokenizer

        # 常用参数
        self.lr = config.lr
        self.batch_size = config.batch_size
        self.hidden_size = config.model.hidden_size

        ModelClass = getBertForMaskedLMClass(config.model)
        self.plm_model = ModelClass.from_pretrained(config.model.model_name_or_path) # type: ignore

        self.length_embedding = nn.Embedding(512, config.model.hidden_size)
        self.dropout = torch.nn.Dropout(config.model.hidden_dropout_prob)

        if self.config.use_ner_prompt or self.config.use_rel_prompt:
            self.prompt_len = config.prompt_len or 16
            self.n_layer = config.model.num_hidden_layers
            self.n_head = config.model.num_attention_heads
            self.n_embd = config.model.hidden_size // config.model.num_attention_heads
            self.prefix_tokens = torch.arange(self.prompt_len).long()

            projection = config.use_prefix_prompt_projection

            if self.config.use_ner_prompt:
                self.ner_prefix_encoder = PrefixEncoder(num_hidden_layers=self.n_layer,
                                                        prompt_len=self.prompt_len,
                                                        prefix_hidden_size=512,
                                                        hidden_size=self.hidden_size,
                                                        prefix_projection=projection)

            if self.config.use_rel_prompt:
                self.rel_prefix_encoder = PrefixEncoder(num_hidden_layers=self.n_layer,
                                                        prompt_len=self.prompt_len,
                                                        prefix_hidden_size=512,
                                                        hidden_size=self.hidden_size,
                                                        prefix_projection=projection)

        self.bce_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

        # Others
        self.rel_num = len(config.dataset.rels)
        self.ent_num = len(config.dataset.ents)
        self.na_idx = data.rel2id.get(config.dataset.na_label, None)
        self.cur_doc_id = 0
        self.cur_mode = 'train'
        self.info_dict = defaultdict(int)

        # 模型评估
        self.best_f1 = 0
        self.test_f1 = 0
        self.test_f1_plus = 0
        self.test_p = None
        self.test_r = None
        self.ner_f1 = None
        self.rel_f1 = None
        self.extend_and_init_additional_tokens()
        self.register_components()

    def get_prompt(self, stage, batch_size):

        if stage == "ner":
            prefix_encoder = self.ner_prefix_encoder
        elif stage == "rel":
            prefix_encoder = self.rel_prefix_encoder
        else:
            raise ValueError("stage must be one of [ner, rel]")

        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.plm_model.device)
        past_key_values = prefix_encoder(prefix_tokens)
        past_key_values = past_key_values.view(
            batch_size,
            self.prompt_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def extend_and_init_additional_tokens(self):
        """扩展并初始化额外的 token"""
        config = self.config

        if self.na_idx is None:
            rels = ['NA'] + config.dataset.rels
        else:
            rels = config.dataset.rels

        ents = config.dataset.ents

        rel_tokens = [f"[R{i}]" for i in range(len(rels))]
        tag_tokens = [f"[S-{e}]" for e in ents] + [f"[E-{e}]" for e in ents]
        ent_tokens = ["[O]"] + [f"[B-{e}]" for e in ents] + [f"[I-{e}]" for e in ents]
        special_tokens = ["[RC]"]

        if config.use_normal_tag:
            tag_tokens += ["[SS]", "[OS]", "[SE]", "[OE]"]

        # 扩展词表
        special_tokens_dict = {'additional_special_tokens': rel_tokens + ent_tokens + tag_tokens + special_tokens}

        # PAD to multiple of 16
        cur_size = len(self.tokenizer) + len(special_tokens_dict["additional_special_tokens"])
        pad_size = 16 - cur_size % 16
        special_tokens_dict["additional_special_tokens"] += [f"[PADME{i}]" for i in range(pad_size)]

        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.plm_model.resize_token_embeddings(len(self.tokenizer)) # type: ignore
        # if config.use_two_plm:
        #     self.plm_model_for_re.resize_token_embeddings(len(self.tokenizer)) # type: ignore

        self.rel_ids = self.tokenizer.convert_tokens_to_ids(rel_tokens)
        self.ent_ids = self.tokenizer.convert_tokens_to_ids(ent_tokens)
        self.tag_ids = self.tokenizer.convert_tokens_to_ids(tag_tokens)

        # 初始化关系的词向量
        with torch.no_grad():
            embeds = self.plm_model.get_input_embeddings().weight # type: ignore
            # if self.config.use_two_plm:
            #     embeds_re = self.plm_model_for_re.get_input_embeddings().weight # type: ignore

            ace_rel_map, ace_ent_map, tag_map = get_language_map_dict()

            # 下面的代码是有点丑陋，甚至于恶心的，但是也没有办法，先这么写着吧
            # Rel
            ace_rel_ids = [self.tokenizer.encode(ace_rel_map[rel], add_special_tokens=False) for rel in rels]
            for i, rel_id in enumerate(self.rel_ids):
                embeds[rel_id] = embeds[ace_rel_ids[i]].mean(dim=-2) # type: ignore
                # if self.config.use_two_plm:
                #     embeds_re[rel_id] = embeds_re[ace_rel_ids[i]].mean(dim=-2) # type: ignore

            # Entity
            ace_ent_ids = [self.tokenizer.encode("outside", add_special_tokens=False)]
            for (idx, ent) in enumerate(ents):
                ace_ent_ids.insert(idx+1, self.tokenizer.encode("begin " + ace_ent_map[ent], add_special_tokens=False))
                ace_ent_ids.append(self.tokenizer.encode("inside " + ace_ent_map[ent], add_special_tokens=False))

            for i, ent_id in enumerate(self.ent_ids):
                embeds[ent_id] = embeds[ace_ent_ids[i]].mean(dim=-2) # type: ignore
                # if self.config.use_two_plm:
                #     embeds_re[ent_id] = embeds_re[ace_ent_ids[i]].mean(dim=-2) # type: ignore

            # Tag
            ace_tag_ids = []
            for (idx, ent) in enumerate(ents):
                ace_tag_ids.insert(idx, self.tokenizer.encode("start " + ace_ent_map[ent], add_special_tokens=False))
                ace_tag_ids.append(self.tokenizer.encode("end " + ace_ent_map[ent], add_special_tokens=False))

            if config.use_normal_tag:
                ace_tag_ids.append(self.tokenizer.encode("subject start", add_special_tokens=False))
                ace_tag_ids.append(self.tokenizer.encode("object start", add_special_tokens=False))
                ace_tag_ids.append(self.tokenizer.encode("subject end", add_special_tokens=False))
                ace_tag_ids.append(self.tokenizer.encode("object end", add_special_tokens=False))

            for i, tag_id in enumerate(self.tag_ids):
                embeds[tag_id] = embeds[ace_tag_ids[i]].mean(dim=-2) # type: ignore
                # if self.config.use_two_plm:
                #     embeds_re[tag_id] = embeds_re[ace_tag_ids[i]].mean(dim=-2) # type: ignore

            assert (self.plm_model.get_input_embeddings().weight == embeds).all() # type: ignore

    def register_components(self):
        """ 用于构建除了预训练语言模型之外的所有模型组件
        很多时候，使用哪些模型组件，使用哪些模块都是需要根据 config 来决定的
        """
        config = self.config

        self.rel_model = REModel(self)
        self.ner_model = NERModel(self)
        self.filter = FilterModel(self)
        # self.span_ner = SpanEntityModel(self)
        # self.graph = RuntimeGraph(self) if config.use_graph_layers > 0 else None

        # if self.config.use_pre_rel:
        #     self.pre_rel_model = PreREModel(config)
        self.graph = None

    def forward(self, batch, mode="train"):

        # Forward
        input_ids, attention_mask, pos, triples, ent_maps, sent_mask, span_mask = batch

        if self.config.use_ner_prompt:
            batch_size = input_ids.shape[0]
            past_key_values = self.get_prompt(stage="ner", batch_size=batch_size)
            prefix_attention_mask = torch.ones(batch_size, self.prompt_len).to(self.plm_model.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        else:
            past_key_values = None

        outputs = self.plm_model(input_ids, attention_mask=attention_mask, output_hidden_states=True, past_key_values=past_key_values) # type: ignore

        # 一些参数
        hidden_state = outputs.hidden_states[-1]

        output = {}
        output["hidden_state"] = hidden_state

        ner_out = self.ner_model(hidden_state, labels=ent_maps, graph=self.graph, mask=sent_mask, mode=mode)
        ner_logits, ner_loss = ner_out[0], ner_out[1]
        if self.config.use_ner_hs:
            raise NotImplementedError("use_ner_hs is not implemented")
            hidden_state = ner_out[2][-1]
        gold_entities = self.ner_model.decode_entities(ent_maps, pos=pos, mode=mode)  # gold entities
        pred_entities = self.ner_model.decode_entities(ner_logits, pos=pos, mode=mode)

        output["gold_entities"] = gold_entities
        output["pred_entities"] = pred_entities
        output["ner_logits"] = ner_logits
        output["pos"] = pos

        if not (mode == "test" and self.config.dry_test):
            if sum([len(e) for e in gold_entities]) > 0:
                rel_output = self.rel_model(
                                    theta=self,
                                    batch=batch,
                                    hidden_state=hidden_state,
                                    gold_entities=gold_entities, # gold entities
                                    pred_entities=pred_entities,
                                    return_loss=True,
                                    mode=mode)

                triples_pred, rel_loss, filter_loss, sent_ner_loss = rel_output
                output["triples_pred_with_gold"] = triples_pred

            else:
                rel_loss = torch.tensor(0.0, device=input_ids.device)
                filter_loss = torch.tensor(0.0, device=input_ids.device)
                sent_ner_loss = torch.tensor(0.0, device=input_ids.device)
                output["triples_pred_with_gold"] = []

        # 如果是测试阶段，使用预测的 triples
        if mode != "train":
            if sum([len(e) for e in pred_entities]) > 0:
                rel_output = self.rel_model(
                                    theta=self,
                                    batch=batch,
                                    hidden_state=hidden_state,
                                    gold_entities=gold_entities, # gold entities
                                    pred_entities=pred_entities,
                                    return_loss=False,
                                    mode=mode)
                output["triples_pred"] = rel_output[0]
            else:
                output["triples_pred"] = []

        # 计算损失
        if mode == "train":

            ner_rate = self.config.ner_rate
            rel_rate = self.config.rel_rate
            filter_rate = self.config.filter_rate
            rel_ner_rate = self.config.rel_ner_rate

            def rate_func(x, max_rate=1):
                if x is None or x == 0 or x == "0":
                    cur_rate = 1
                elif self.config.use_convert_warmup_to_steps:
                    cur_rate = (self.global_step + 1) / (int(x) / self.config.max_epochs * self.num_training_steps)
                else:
                    cur_rate = (self.current_epoch + 1) / int(x)

                return min(max_rate, cur_rate)

            rel_rate = rate_func(self.config.use_warmup_rel) * rel_rate
            ner_rate = rate_func(self.config.use_warmup_ner) * ner_rate
            filter_rate = rate_func(self.config.use_warmup_filter) * filter_rate

            # stop_grad
            # if self.config.loss_stop_grad:
            #     ner_rate = 1 / (ner_loss.detach().item() + 1e-8) * ner_rate
            #     rel_rate = 1 / (rel_loss.detach().item() + 1e-8) * rel_rate
            #     filter_rate = 1 / (filter_loss.detach().item() + 1e-8) * filter_rate
            #     rel_ner_rate = 1 / (sent_ner_loss.detach().item() + 1e-8) * rel_ner_rate
            #     loss = ner_loss * ner_rate + rel_loss * rel_rate + filter_loss * filter_rate + sent_ner_loss * rel_ner_rate

            self.log("loss/ner_loss", ner_loss)
            self.log("loss/rel_loss", rel_loss)
            self.log("loss/filter_loss", filter_loss)
            self.log("loss/sent_ner_loss", sent_ner_loss)

            norm_gamma = 1 / self.config.global_batch_size * self.config.batch_size * 2
            if self.config.norm_loss_bs:
                ner_loss = ner_loss * norm_gamma
                rel_loss = rel_loss * norm_gamma
                filter_loss = filter_loss if self.config.use_filter_loss_sum else (filter_loss * norm_gamma)
                sent_ner_loss = sent_ner_loss * norm_gamma

            loss = ner_loss * ner_rate + rel_loss * rel_rate + filter_loss * filter_rate + sent_ner_loss * rel_ner_rate

            output["loss"] = loss

        return output

    # Train https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html#pytorch_lightning.core.LightningModule.training_step
    def training_step(self, batch, batch_idx):

        loss = self(batch, mode="train")["loss"]
        self.log('loss/train_loss', loss)

        lr_step = {}
        for i, pg in enumerate(self.optimizers().param_groups): # type: ignore
            lr_step[f"info/lr_{i}"] = pg["lr"]
        self.log_dict(lr_step)

        return loss

    def training_epoch_end(self, outputs):
        self.filter.log_filter_train_metrics()
        self.rel_model.log_filter_rate()
        self.rel_model.log_statistic_train()

    def validation_step(self, batch, batch_idx):
        output = self(batch, mode="dev")
        return self.eval_step_output(batch, output)

    def validation_epoch_end(self, outputs):
        f1, p, r = f1_score(outputs, 'pred_triples', 'gold_triples', slice=3)
        ner_f1, ner_p, ner_r = f1_score(outputs, 'pred_entities', 'gold_entities')
        rel_f1, rel_p, rel_r = f1_score(outputs, 'pred_triples_with_gold', 'gold_triples', slice=3)

        self.best_f1 = max(f1, self.best_f1)
        self.log_dict_values({'val_f1': f1, 'val/precision': p, 'val/recall': r})
        self.log_dict_values({'best_f1': self.best_f1}, on_epoch=True, prog_bar=True)
        self.log_dict_values({'val/ner_f1': ner_f1, 'val/ner_p': ner_p, 'val/ner_r': ner_r})
        self.log_dict_values({'val/rel_f1': rel_f1, 'val/rel_p': rel_p, 'val/rel_r': rel_r})
        self.filter.log_filter_val_metrics()
        self.rel_model.log_ent_pair_info()
        self.rel_model.log_filter_rate_val()
        self.rel_model.log_statistic_val()

    def test_step(self, batch, batch_idx):
        output = self(batch, mode="test")
        return self.eval_step_output(batch, output)

    def test_epoch_end(self, outputs):
        f1, p, r = f1_score(outputs, 'pred_triples', 'gold_triples', slice=4)
        f1_plus, p_plus, r_plus = f1_score(outputs, 'pred_triples', 'gold_triples')
        ner_f1, ner_p, ner_r = f1_score(outputs, 'pred_entities', 'gold_entities')
        rel_f1, rel_p, rel_r = f1_score(outputs, 'pred_triples_with_gold', 'gold_triples', slice=4)

        # 保存 output
        with open(f"{self.config.output_dir}/output.json", "w") as f:
            import json
            json.dump(outputs, f)

        # returned
        self.test_f1 = f1
        self.test_p = p
        self.test_r = r
        self.test_f1_plus = f1_plus
        self.test_p_plus = p_plus
        self.test_r_plus = r_plus
        self.ner_f1 = ner_f1
        self.ner_p = ner_p
        self.ner_r = ner_r
        self.rel_f1 = rel_f1
        self.rel_p = rel_p
        self.rel_r = rel_r
        self.log_dict_values({'test/f1': f1, 'test/p': p, 'test/r': r})
        self.log_dict_values({'test/ner_f1': ner_f1, 'test/ner_p': ner_p, 'test/ner_r': ner_r})
        self.log_dict_values({'test/rel_f1': rel_f1, 'test/rel_p': rel_p, 'test/rel_r': rel_r})
        self.log_dict_values({'test/f1_plus': f1_plus, 'test/p_plus': p_plus, 'test/r_plus': r_plus})
        # self.filter.log_filter_val_metrics()
        # self.rel_model.log_ent_pair_info()
        # self.rel_model.log_filter_rate_val()
        # self.rel_model.log_statistic_val()

    def eval_step_output(self, batch, output):
        # batch = batch_filter(batch, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id)
        input_ids, _, pos, triples, ent_maps, sent_mask, _  = batch # type: ignore

        pred_entities = self.get_span_set(input_ids, output["pred_entities"])
        gold_entities = self.get_span_set(input_ids, output["gold_entities"])

        # if self.config.debug:
        #     wrong = set(pred_entities) - set(gold_entities)
        #     lost = set(gold_entities) - set(pred_entities)
        #     if len(wrong) > 0 or len(lost) > 0:
        #         print(f"Wrong: {wrong}\nLost: {lost}")
        #         pass

        pred_triples, gold_triples = self.get_triple_set(input_ids, triples, output, "triples_pred")

        if not self.config.dry_test:
            pred_triples_with_gold = self.get_triple_set(input_ids, triples, output, "triples_pred_with_gold", pred_only=True)
        else:
            pred_triples_with_gold = []

        return {
            'pred_entities': pred_entities,
            'gold_entities': gold_entities,
            'pred_triples': pred_triples,
            'gold_triples': gold_triples,
            'pred_triples_with_gold': pred_triples_with_gold,
        }


    # Optimizer https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html#pytorch_lightning.core.LightningModule.configure_optimizers
    def configure_optimizers(self):
        return get_optimizer(self, self.config)

    def get_triple_set(self, input_ids, triples, output, name, pred_only=False):
        pred_triples = []
        # start, end 是左闭右开区间
        # [batch_idx, sub_start, sub_end, obj_start, obj_end, sub_type, obj_type, score, rel_idx]
        #  0          1          2        3          4        5         6         7      8 (include NA)
        for t in output[name]:
            if t[8] != 0:
                sub_token = self.tokenizer.decode(input_ids[t[0], t[1]:t[2]])
                obj_token = self.tokenizer.decode(input_ids[t[0], t[3]:t[4]])
                rel_type = self.config.dataset.rels[t[8]-1]
                sub_type = self.config.dataset.ents[t[5]]
                obj_type = self.config.dataset.ents[t[6]]
                triple = (t[0], sub_token, obj_token, rel_type, sub_type, obj_type)
                pred_triples.append(triple)

        if pred_only:
            return pred_triples

        gold_triples = []
        # [sub_start, sub_end, obj_start, obj_end, rel_idx, sub_type, obj_type]
        #  0          1        2          3        4(No NA) 5         6
        for b in range(input_ids.shape[0]):
            for t in triples[b]:
                if t[4] != -1:
                    sub_token = self.tokenizer.decode(input_ids[b, t[0]:t[1]])
                    obj_token = self.tokenizer.decode(input_ids[b, t[2]:t[3]])
                    rel_type = self.config.dataset.rels[t[4]]
                    sub_type = self.config.dataset.ents[t[5]]
                    obj_type = self.config.dataset.ents[t[6]]
                    triple = (b, sub_token, obj_token, rel_type, sub_type, obj_type)
                    gold_triples.append(triple)
                else:
                    break
        return pred_triples, gold_triples

    def get_span_set(self, input_ids, entities):
        entities_token = []
        for b in range(input_ids.shape[0]):
            for e in entities[b]:
                ent_token = self.tokenizer.decode(input_ids[b, e[0]:e[1]])
                ent_type = self.config.dataset.ents[e[2]]
                entities_token.append((b, ent_token, ent_type))
        return entities_token

    def log_dict_values(self, d, **kwargs):
        for k, v in d.items():
            self.log(k, v, **kwargs)

    def get_rel_tag_embeddings(self, with_na=False, with_grad=True, device=None):
        device = device or self.device
        rel_tag_embeddings = self.plm_model.get_input_embeddings().weight[torch.tensor(self.rel_ids, device=self.device)]
        if not with_na:
            rel_tag_embeddings = rel_tag_embeddings[1:]
        if not with_grad:
            rel_tag_embeddings = rel_tag_embeddings.detach()
        return rel_tag_embeddings