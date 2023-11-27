# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/01-gpt2-with-value-head.ipynb (unless otherwise specified).

__all__ = ['CausalLMOutputWithCrossAttentions', 'ValueHead', 'GPT2HeadWithValueModel', 'respond_to_batch']

# Cell
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Model, GPT2PreTrainedModel
from transformers import top_k_top_p_filtering
from transformers.modeling_outputs import ModelOutput
from torch import nn
from torch.nn import Identity
import torch.nn.functional as F
import torch
from dataclasses import dataclass
from typing import Optional, Tuple

# Cell
@dataclass
class CausalLMOutputWithCrossAttentions(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    value: Optional[torch.FloatTensor] = None

# Cell

class ValueHead(nn.Module):
    """The ValueHead class implements a head for GPT2 that returns a scalar for each output token."""
    def __init__(self, config):
        super().__init__()
        self.detach_head = False
        self.summary_type = config.summary_type if hasattr(config, "summary_type") else "last"
        if self.summary_type == "attn":
            raise NotImplementedError

        self.summary = Identity()
        if hasattr(config, "summary_use_proj") and config.summary_use_proj:
            if hasattr(config, "summary_proj_to_labels") and config.summary_proj_to_labels and config.num_labels > 0:
                num_classes = config.num_labels
            else:
                num_classes = config.hidden_size
            self.summary = nn.Linear(config.hidden_size, num_classes)

        self.activation = Identity()
        if hasattr(config, "summary_activation") and config.summary_activation == "tanh":
            self.activation = nn.Tanh()

        self.first_dropout = Identity()
        if hasattr(config, "summary_first_dropout") and config.summary_first_dropout > 0:
            self.first_dropout = nn.Dropout(config.summary_first_dropout)

        self.last_dropout = Identity()
        if hasattr(config, "summary_last_dropout") and config.summary_last_dropout > 0:
            self.last_dropout = nn.Dropout(config.summary_last_dropout)

        self.flatten = nn.Flatten()

    def forward(self, hidden_states, cls_index=None):
        if self.detach_head:
            output = hidden_states.detach()
        else:
            output = hidden_states
        output = self.first_dropout(output)
        output = self.summary(output)
        output = self.activation(output)
        output = self.last_dropout(output)

        return output

# Cell

class GPT2HeadWithValueModel(GPT2LMHeadModel):
    """The GPT2HeadWithValueModel class implements a GPT2 language model with a secondary, scalar head."""
    def __init__(self, config):
        print('config',config)
        super().__init__(config)
        config.num_labels = 1
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.v_head = ValueHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }

    def detach_value_head(self):
        self.v_head.detach_head = True

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mc_token_ids=None,
        lm_labels=None,
        mc_labels=None,
        use_cache=None,
        return_dict=False,
        output_attentions=False,
        output_hidden_states=False,
    ):
        loss=None
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            use_cache=None,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)
        value = self.v_head(hidden_states).squeeze(-1)

        if not return_dict:
            outputs = (lm_logits,) + transformer_outputs[1:] + (value,)
    
            return outputs

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
            value=value,
        )
        return outputs

## CriticControl
def sentiment_generation(model, queries, txt_len=25, top_vocab=1, top_p=1.0, no_repeat_ngram=-1):
    """Sample text from language model."""
    input_ids = queries
    ngram_list = dict()
    next_token_id = 0
    for i in range(txt_len):
        # Get Logits
        outputs = model(input_ids)
        # print("======================")
        # print(outputs[0].size())
        #
        # print(outputs[2].size())

        next_token_logits = outputs[0][:, -1, :]
        probs = F.softmax(next_token_logits, dim=-1)
        # V_value = outputs[2].unsqueeze(-1)[:, -1, :]
        # Sample
        # _, candidate_tokens = torch.topk(probs, top_vocab, dim=-1)
        # for _, Q_token in enumerate(candidate_tokens[0]):
        #     # print('Q_token',Q_token)
        #     # print('probs[0][Q_token.item()] ',probs[0][Q_token.item()] )
        #     Q_value = model(torch.cat([input_ids, Q_token.view([1,1])], dim=-1))[2].unsqueeze(-1)[:, -1, :]
        #     # print(Q_value.size())
        #     probs[0][Q_token.item()] = probs[0][Q_token.item()] * (torch.nn.Sigmoid()(Q_value) / torch.nn.Sigmoid()(V_value))
            # print('probs[0][Q_token.item()] ', probs[0][Q_token.item()])
        # if tuple(input_ids[0][-no_repeat_ngram+1:].tolist()) in ngram_list.keys():
        #     banned_token_list = ngram_list[tuple(input_ids[0][-no_repeat_ngram+1:].tolist())]
        #     for _, banned_token in enumerate(banned_token_list):
        #         probs[0][banned_token] = -float("inf")

        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
        # if tuple(input_ids[0][-no_repeat_ngram:-1].tolist()) in ngram_list.keys():
        #     ngram_list[tuple(input_ids[0][-no_repeat_ngram:-1].tolist())].append(next_token.item())
        # else:
        #     ngram_list[tuple(input_ids[0][-no_repeat_ngram:-1].tolist())] = [next_token.item()]
        # print("ngram_list",ngram_list)
    return input_ids

## CriticControl
def topic_generation(model, queries, txt_len=80, top_k=10, top_p=1.0, no_repeat_ngram=4):
    """Sample text from language model."""
    input_ids = queries
    ngram_list = dict()
    next_token_id = 0
    for i in range(txt_len):
        # Get Logits
        outputs = model(input_ids)
        next_token_logits = outputs[0][:, -1, :]
        V_value = outputs[2].unsqueeze(-1)[:, -1, :]
        probs = F.softmax(next_token_logits, dim=-1)
        _, candidate_tokens = torch.topk(probs, top_k, dim=-1)
        # Distribution Shift!
        for _, Q_token in enumerate(candidate_tokens[0]):
            Q_value = model(torch.cat([input_ids, Q_token.view([1,1])], dim=-1))[2].unsqueeze(-1)[:, -1, :]
            probs[0][Q_token.item()] = probs[0][Q_token.item()] * (torch.nn.Sigmoid()(Q_value) / torch.nn.Sigmoid()(V_value))**(1/1)
        if tuple(input_ids[0][-no_repeat_ngram+1:].tolist()) in ngram_list.keys():
            banned_token_list = ngram_list[tuple(input_ids[0][-no_repeat_ngram+1:].tolist())]
            for _, banned_token in enumerate(banned_token_list):
                probs[0][banned_token] = 0
        _, next_token = torch.topk(probs, 1, dim=-1) 
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        if tuple(input_ids[0][-no_repeat_ngram:-1].tolist()) in ngram_list.keys():
            ngram_list[tuple(input_ids[0][-no_repeat_ngram:-1].tolist())].append(next_token.item())
        else:
            ngram_list[tuple(input_ids[0][-no_repeat_ngram:-1].tolist())] = [next_token.item()]
    return input_ids