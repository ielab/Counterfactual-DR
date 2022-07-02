import torch
from transformers import PreTrainedModel, RobertaConfig, RobertaModel, RobertaTokenizer, BertModel, BertTokenizer
from typing import List, Optional
import numpy as np


class AnceEncoder(PreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = 'ance_encoder'
    load_tf_weights = None
    _keys_to_ignore_on_load_missing = [r'position_ids']
    _keys_to_ignore_on_load_unexpected = [r'pooler', r'classifier']

    def __init__(self, config: RobertaConfig):
        super().__init__(config)
        self.config = config
        self.roberta = RobertaModel(config)
        self.embeddingHead = torch.nn.Linear(config.hidden_size, 768)
        self.norm = torch.nn.LayerNorm(768)
        self.init_weights()

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def init_weights(self):
        self.roberta.init_weights()
        self.embeddingHead.apply(self._init_weights)
        self.norm.apply(self._init_weights)

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
    ):
        input_shape = input_ids.size()
        device = input_ids.device
        if attention_mask is None:
            attention_mask = (
                torch.ones(input_shape, device=device)
                if input_ids is None
                else (input_ids != self.roberta.config.pad_token_id)
            )
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        all_hidden_states, _ = outputs[-2:]
        pooled_output = all_hidden_states[-1][:, 0, :]
        pooled_output = self.norm(self.embeddingHead(pooled_output))
        return pooled_output


class AnceQueryEncoder:
    def __init__(self, encoder_dir: str = None, device: str = 'cpu'):
        if encoder_dir:
            self.device = device
            self.model = AnceEncoder.from_pretrained(encoder_dir, output_hidden_states=True, output_attentions=True)
            self.model.to(self.device)
            self.tokenizer = RobertaTokenizer.from_pretrained(encoder_dir)

    def encode(self, query: str):
        inputs = self.tokenizer(
            [query],
            max_length=64,
            padding='longest',
            truncation=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        inputs.to(self.device)
        embeddings = self.model(inputs["input_ids"]).detach().cpu().numpy().astype('float32')
        return embeddings.flatten()


    def batch_encode(self, query: List[str]):
        inputs = self.tokenizer(
            query,
            max_length=512,
            padding='longest',
            truncation=True,
            add_special_tokens=False,
            return_tensors='pt'
        )
        inputs.to(self.device)
        embeddings = self.model(inputs["input_ids"]).detach().cpu().numpy()
        return embeddings


class TctColBertQueryEncoder:

    def __init__(self, encoder_dir: str = None, tokenizer_name: str = None,
                 encoded_query_dir: str = None, device: str = 'cpu'):
        if encoder_dir:
            self.device = device
            self.model = BertModel.from_pretrained(encoder_dir)
            self.model.to(self.device)
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name or encoder_dir)
            self.has_model = True
        if (not self.has_model) and (not self.has_encoded_query):
            raise Exception('Neither query encoder model nor encoded queries provided. Please provide at least one')

    def encode(self, query: str):
        if self.has_model:
            max_length = 36  # hardcode for now
            inputs = self.tokenizer(
                '[CLS] [Q] ' + query + '[MASK]' * max_length,
                max_length=max_length,
                truncation=True,
                add_special_tokens=False,
                return_tensors='pt'
            )
            inputs.to(self.device)
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.detach().cpu().numpy()
            return np.average(embeddings[:, 4:, :], axis=-2).flatten()


class PositionBasedClickModel:
    def __init__(self, pc=None, eta=1):
        self.eta = eta
        self.pc = np.array(pc)
        self.propensities = np.power(np.divide(1, np.arange(1.0, 11)), self.eta)

    def simulate(self, rels):
        click_probs = self.pc[rels]
        click_probs = click_probs * self.propensities
        rand = np.random.rand(len(rels))
        clicks = rand < click_probs
        clicks = clicks.astype(int)

        return clicks

    def batch_simulate(self, batch_rels):
        click_probs = self.pc[batch_rels]
        click_probs = click_probs * self.propensities
        rand = np.random.rand(batch_rels.shape[0], batch_rels.shape[1])
        clicks = rand < click_probs
        clicks = clicks.astype(int)

        return clicks