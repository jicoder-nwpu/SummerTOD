"""
   MTTOD: model.py

   implements MTTOD model, with huggingface transformers module.

   Copyright 2021 ETRI LIRS, Yohan Lee
   Copyright 2018- The Hugging Face team. All rights reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""


import copy

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import T5ForConditionalGeneration, T5EncoderModel
from transformers.modeling_outputs import Seq2SeqLMOutput

from utils import definitions
from adaptor import Adaptor


class T5ForSummaryGeneration(T5ForConditionalGeneration):
    def __init__(self, config):
        super(T5ForSummaryGeneration, self).__init__(config)
    
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.is_encoder_decoder = False
        encoder_config.use_cache = False
        encoder_config.num_layers = config.summary_encoder_num_layers
        
        encoder_config.add_summary_cross_attention = True
        # encoder_config.add_summary_cross_attention = False

        self.summary_encoder = type(self.encoder)(encoder_config, self.shared)

    def initialize_additional(self):
        encoder_config = copy.deepcopy(self.config)
        encoder_config.is_decoder = False
        encoder_config.is_encoder_decoder = False
        # encoder_config.add_summary_cross_attention = False
        encoder_config.use_cache = False
        encoder_config.num_layers = self.config.summary_encoder_num_layers
        
        encoder_config.add_summary_cross_attention = True

        self.summary_encoder = type(self.encoder)(encoder_config, self.shared)
        
        self.summary_encoder.load_state_dict(self.encoder.state_dict(), strict=False)

        t5_state_dict=torch.load('./model_path/pytorch_model.bin')
        for n,p in self.decoder.named_parameters():
            if "layer.2" in n:
                name_split = n.split('layer.2')
                name = 'decoder.' + name_split[0] + 'layer.1' + name_split[1]
                p.data.copy_(t5_state_dict[name].data)
            if "layer.3" in n:
                name_split = n.split('layer.3')
                name = 'decoder.' + name_split[0] + 'layer.2' + name_split[1]
                p.data.copy_(t5_state_dict[name].data)
                
        for n,p in self.summary_encoder.named_parameters():
            if "layer.1" in n:
                name_split = n.split('layer.1')
                name = 'decoder.' + name_split[0] + 'layer.1' + name_split[1]
                p.data.copy_(t5_state_dict[name].data)
            if "layer.2" in n:
                name_split = n.split('layer.2')
                name = 'encoder.' + name_split[0] + 'layer.1' + name_split[1]
                p.data.copy_(t5_state_dict[name].data)

    def initialize_weights(self, modules):
        for module in modules:
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def prepare_inputs_for_generation(self, input_ids,
                                      past=None, attention_mask=None,
                                      use_cache=None, encoder_outputs=None,
                                      **kwargs):
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {"decoder_input_ids": input_ids,
                "past_key_values": past,
                "encoder_outputs": encoder_outputs,
                "attention_mask": attention_mask,
                "use_cache": use_cache,
                "summary_encoder_outputs": kwargs.get('summary_encoder_outputs'),
                "summary_attention_mask": kwargs.get('summary_attention_mask'),
                "decoder_type": kwargs.get('decoder_type')}

    def forward(self,
                input_ids=None,
                attention_mask=None,
                decoder_input_ids=None,
                encoder_outputs=None,
                summary_attention_mask=None,
                summary_encoder_outputs=None,
                past_key_values=None,
                inputs_embeds=None,
                decoder_inputs_embeds=None,
                lm_labels=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                encoder_only=None,
                encoder_type=None,
                decoder_type=None):

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        
        if encoder_outputs is not None:
            if isinstance(encoder_outputs, tuple):
                encoder_hidden_states = encoder_outputs[0]
            else:
                encoder_hidden_states = encoder_outputs.last_hidden_state

        #对话历史部分编码
        if encoder_type == 'history' and encoder_outputs is None:
            
            encoder_outputs = self.encoder(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           inputs_embeds=inputs_embeds,
                                           return_dict=return_dict)

            if return_dict:
                encoder_hidden_states = encoder_outputs.last_hidden_state
            else:
                encoder_hidden_states = encoder_outputs[0]

            hs = encoder_hidden_states * (self.model_dim ** -0.5)

        #last_summary 筛选
        if summary_encoder_outputs is not None:
            if isinstance(summary_encoder_outputs, tuple):
                summary_hidden_states = summary_encoder_outputs[0]
            else:
                summary_hidden_states = summary_encoder_outputs.last_hidden_state
            
        elif encoder_type == 'summary':
            
            if summary_attention_mask is None:
                summary_encoder_outputs = self.summary_encoder(input_ids=input_ids,
                                                               attention_mask=attention_mask,
                                                               return_dict=return_dict)
            else:
                summary_encoder_outputs = self.summary_encoder(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            encoder_hidden_states=encoder_hidden_states,
                                            encoder_attention_mask=summary_attention_mask,
                                            return_dict=return_dict)
            
            if return_dict:
                summary_hidden_states = summary_encoder_outputs.last_hidden_state
            else:
                summary_hidden_states = summary_encoder_outputs[0]

        if encoder_only:
            if encoder_type == 'history':
                return encoder_outputs
            else:
                return summary_encoder_outputs        
            
        if lm_labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = self._shift_right(lm_labels)

        # if decoder_type == "resp":
        #     decoder = self.resp_decoder
        #     lm_head = self.resp_lm_head
        # else:
        decoder = self.decoder
        lm_head = self.lm_head

        if past_key_values is not None:
            assert lm_labels is None, "Decoder should not use cached key value states when training"
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        decoder_outputs = decoder(input_ids=decoder_input_ids,
                                inputs_embeds=decoder_inputs_embeds,
                                past_key_values=past_key_values,
                                encoder_hidden_states=encoder_hidden_states,
                                encoder_attention_mask=attention_mask,
                                summary_hidden_states=summary_hidden_states,
                                summary_attention_mask=summary_attention_mask,
                                use_cache=use_cache,
                                output_attentions=output_attentions,
                                return_dict=return_dict)

        sequence_output = decoder_outputs[0]

        sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = lm_head(sequence_output)

        lm_loss = None
        if lm_labels is not None:
            lm_loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            lm_loss = lm_loss_fct(
                lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))

        # for training
        if not return_dict:
            pred_lm = torch.argmax(lm_logits, dim=-1)
            outputs = (lm_loss, pred_lm,) + \
                (encoder_hidden_states, lm_logits)

        # for prediction
        else:
            outputs = Seq2SeqLMOutput(
                loss=lm_loss,
                logits=lm_logits,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.last_hidden_state,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs[1] if len(
                    encoder_outputs) > 1 else None,
                encoder_attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None)

        return outputs
