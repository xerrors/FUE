import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
import os
import torch
from data.data_structures import Dataset
from data.utils import convert_dataset_to_samples
from xerrors import cprint as cp


class DataModule(pl.LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.model_name_or_path)

        self.data_train = None
        self.data_val = None
        self.data_test = None

        self.rel_num = len(config.dataset.rels)
        self.id2rel = config.dataset.rels
        self.rel2id = {name: idx for idx, name in enumerate(config.dataset.rels)}

        self.ner_num = len(config.dataset.ents)
        self.id2ner = config.dataset.ents
        self.ner2id = {name: idx for idx, name in enumerate(config.dataset.ents)}

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.data_train = self.__get_dataset("train")
            self.data_val = self.__get_dataset("val")

        if stage == "test" or stage is None:
            self.data_test = self.__get_dataset("test")

        if stage == "predict":
            self.data_test = self.get_dataset_ace_for_predict()

    def __get_dataset(self, mode):
        """根据不同的任务类型以及数据集类型使用不同的数据加载方法"""
        print(cp.green(f"Loading {mode} data..."), end=" ")
        if self.config.dataset.name in ["ace2005", "ace2004", 'scierc']:

            if self.config.dataset.name == "ace2005":
                filename = self.config.dataset[mode]
            elif self.config.dataset.name == "ace2004":
                piece = self.config.data_piece or 0
                assert piece in [0, 1, 2, 3, 4], "data_piece must be in [0, 1, 2, 3, 4]"
                filename = self.config.dataset[mode][piece]
            elif self.config.dataset.name == 'scierc':
                filename = self.config.dataset[mode]
            else:
                raise NotImplementedError

            cache_tag = ""
            cache_tag += f".max{self.config.max_seq_len}"
            cache_tag += f".ctw{self.config.context_window}"
            cache_tag += ".cross_ner" if self.config.use_cross_ner else ""

            cache_path = filename + cache_tag + ".cache"
            if os.path.exists(cache_path) and self.config.use_cache:
                print(cp.green(f"Loading {mode} data from {cache_path}"))
                datasets = torch.load(cache_path)
            else:
                datasets = self.get_dataset_ace(mode, filename)
                if self.config.use_cache:
                    torch.save(datasets, cache_path)
        else:
            raise NotImplementedError(
                f"Dataset {self.config.dataset.name} not implemented!")

        return datasets

    def get_dataset_ace(self, mode, filename=None):

        dataset = Dataset(filename)

        features = convert_dataset_to_samples(
            dataset, self.config, self.tokenizer, is_test=(mode == "test"))

        # 由于这里是已经将 rel 转化为 map 的形式，如果出现主机内存爆掉的情况，就自己写一个 dataloader，等到加载的时候再转化为 map
        dataset = TensorDataset(
            features["input_ids"],
            features["attention_mask"],
            features["pos"],
            features["triples"],
            features["ent_maps"],
            features["sent_mask"],
            features["span_mask"],
            )

        return dataset

    def get_dataset_ace_for_predict(self):

        dataset = Dataset(self.config.dataset["test"])
        items = []
        for doc_id, doc in enumerate(dataset): # type: ignore

            doc_text = " ".join([" ".join(sent.text) for sent in doc.sentences])

            for sent in doc.sentences:
                item = {
                    "doc": doc_text,
                    "sent": " ".join(sent.text),
                }

                entities = []
                for ent in sent.ner:
                    entities.append({
                        "entity text": " ".join(ent.span.text),
                        "entity type": ent.label,
                    })

                relations = []
                for rel in sent.relations:
                    relations.append({
                        "subject": " ".join(rel.pair[0].text),
                        "object": " ".join(rel.pair[1].text),
                        "relation": rel.label,
                    })

                item["entities"] = entities # type: ignore
                item["relations"] = relations # type: ignore

                if len(relations) > 0:
                    items.append(item)


        return items

    def train_dataloader(self):
        return DataLoader(self.data_train, shuffle=True, batch_size=self.config.batch_size, num_workers=self.config.num_worker, pin_memory=True) # type: ignore

    def val_dataloader(self):
        return DataLoader(self.data_val, shuffle=False, batch_size=self.config.batch_size, num_workers=self.config.num_worker, pin_memory=True) # type: ignore

    def test_dataloader(self):
        return DataLoader(self.data_test, shuffle=False, batch_size=self.config.test_batch_size, num_workers=self.config.num_worker, pin_memory=False) # type: ignore
