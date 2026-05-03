from dataclasses import dataclass

from transformers.models.t5.modeling_t5 import (
    T5Stack, T5Block, T5LayerNorm, T5LayerSelfAttention, T5LayerFF, T5LayerCrossAttention,
    T5PreTrainedModel, T5ForConditionalGeneration
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import copy

from transformers.modeling_outputs import ModelOutput, BaseModelOutput, BaseModelOutputWithPast, BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput, Seq2SeqModelOutput
from transformers.modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import logging
from transformers import BeamScorer, BeamSearchScorer


# from utils import *

logger = logging.get_logger(__name__)


# ================= [新增模块 Start] =================
# 全局变量，用于在 Forward 过程中传递生成的参数
CURRENT_HYPER_PARAMS = {}

class HyperLinear(nn.Linear):
    """
    一个能够接收 HyperNetwork 参数的 Linear 层。
    它在原有 Linear 的基础上，加上 LoRA 项： output = Linear(x) + x @ A @ B
    """
    def __init__(self, original_linear, layer_idx, param_type):
        super().__init__(original_linear.in_features, original_linear.out_features, bias=original_linear.bias is not None)
        with torch.no_grad():
            self.weight.copy_(original_linear.weight)
            if original_linear.bias is not None:
                self.bias.copy_(original_linear.bias)
        
        self.layer_idx = layer_idx
        self.param_type = param_type # 'q' 或 'v'

    def forward(self, input):
        output = F.linear(input, self.weight, self.bias)

        if self.layer_idx in CURRENT_HYPER_PARAMS:
            matrices = CURRENT_HYPER_PARAMS[self.layer_idx].get(self.param_type)
            if matrices is not None:
                matrix_a, matrix_b = matrices
                lora_h = torch.matmul(input, matrix_a)
                lora_out = torch.matmul(lora_h, matrix_b)
                output = output + lora_out
        
        return output

class HyperLoRAGenerator(nn.Module):
    """
    HyperNetwork: 输入 Prototype，输出所有层的 LoRA 参数 (A 和 B)
    """
    def __init__(self, prototype_dim, hidden_dim, adapter_dim, rank=8, num_layers=12):
        super().__init__()
        self.rank = rank
        self.adapter_dim = adapter_dim
        self.num_layers = num_layers
        
        self.generator = nn.Sequential(
            nn.Linear(prototype_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.head_a = nn.Linear(hidden_dim, num_layers * 2 * adapter_dim * rank)
        self.head_b = nn.Linear(hidden_dim, num_layers * 2 * rank * adapter_dim)
        
        nn.init.zeros_(self.head_a.weight)
        nn.init.zeros_(self.head_a.bias)
        nn.init.zeros_(self.head_b.weight)
        nn.init.zeros_(self.head_b.bias)

    def forward(self, prototype_vector):
        batch_size = prototype_vector.size(0)
        features = self.generator(prototype_vector)
        
        flat_a = self.head_a(features)
        all_a = flat_a.view(batch_size, self.num_layers, 2, self.adapter_dim, self.rank)
        
        flat_b = self.head_b(features)
        all_b = flat_b.view(batch_size, self.num_layers, 2, self.rank, self.adapter_dim)
        
        return all_a, all_b
# ================= [新增模块 End] =================


class VisualEmbedding(nn.Module):
    def __init__(self, config, obj_order_embedding):
        super().__init__()
        self.config = config
        feat_dim = config.feat_dim
        pos_dim = config.pos_dim
        n_images = config.n_images

        if self.config.individual_vis_layer_norm:
            feat_embedding = [nn.Linear(feat_dim, config.d_model)]
            if self.config.use_vis_layer_norm:
                feat_embedding.append(T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon))
            self.feat_embedding = nn.Sequential(*feat_embedding)

            absolute_vis_pos_embedding = [nn.Linear(pos_dim + 1, config.d_model)]
            if self.config.use_vis_layer_norm:
                absolute_vis_pos_embedding.append(T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon))
            self.absolute_vis_pos_embedding = nn.Sequential(*absolute_vis_pos_embedding)

            if self.config.use_vis_order_embedding:
                self.obj_order_embedding = obj_order_embedding
                self.img_order_embedding = nn.Embedding(n_images, config.d_model)

        else:
            feat_embedding = [nn.Linear(feat_dim, config.d_model)]
            self.feat_embedding = nn.Sequential(*feat_embedding)

            absolute_vis_pos_embedding = [nn.Linear(pos_dim + 1, config.d_model)]
            self.absolute_vis_pos_embedding = nn.Sequential(*absolute_vis_pos_embedding)

            if self.config.use_vis_order_embedding:
                self.obj_order_embedding = obj_order_embedding
                self.img_order_embedding = nn.Embedding(n_images, config.d_model)

            if self.config.use_vis_layer_norm:
                self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def get_area(self, pos):
        height = pos[:, :, 3] - pos[:, :, 2]
        width = pos[:, :, 1] - pos[:, :, 0]
        area = height * width
        return area


    def forward(self, feats, pos, img_order_ids=None, obj_order_ids=None):
        B, N, _ = feats.size()
        assert pos.size() == (B, N, 4)

        feat_embedding = self.feat_embedding(feats)

        device = feats.device
        dtype = feats.dtype

        area = self.get_area(pos).unsqueeze(2) # [B, N, 1]
        pos = torch.cat([pos, area], dim=2) # [B, N, 5]

        absolute_vis_pos_embedding = self.absolute_vis_pos_embedding(pos)

        if self.config.use_vis_order_embedding:
            if img_order_ids is None:
                img_order_ids = torch.zeros(N, dtype=torch.long, device=device)
                img_order_ids = img_order_ids.unsqueeze(0) 
            img_order_embedding = self.img_order_embedding(img_order_ids)

            if obj_order_ids is None:
                obj_order_ids = torch.arange(N, dtype=torch.long, device=device)
                obj_order_ids = obj_order_ids.unsqueeze(0) 
            obj_order_ids = self.obj_order_embedding.num_embeddings - obj_order_ids - 1
            obj_order_embedding = self.obj_order_embedding(obj_order_ids)

            vis_embedding = feat_embedding + absolute_vis_pos_embedding + \
                img_order_embedding + obj_order_embedding

        else:
            vis_embedding = feat_embedding + absolute_vis_pos_embedding

        if not self.config.individual_vis_layer_norm:
            if self.config.use_vis_layer_norm:
                vis_embedding = self.layer_norm(vis_embedding)

        return vis_embedding

class JointEncoder(T5Stack):
    def __init__(self, config, embed_tokens=None):
        super(T5Stack, self).__init__(config)
        self.config = config

        self.embed_tokens = embed_tokens
        self.is_decoder = self.config.is_decoder
        assert self.config.is_decoder is False

        self.visual_embedding = VisualEmbedding(self.config, embed_tokens)

        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=(i == 0))
                for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        self.init_weights()
        self.model_parallel = False
        self.device_map = None
        print("========== Original Joint Encoder ========== ")
        
        for i, block in enumerate(self.block):
            try:
                attn_module = block.layer[0].SelfAttention
                if not isinstance(attn_module.q, HyperLinear):
                    attn_module.q = HyperLinear(attn_module.q, layer_idx=i, param_type='q')
                if not isinstance(attn_module.v, HyperLinear):
                    attn_module.v = HyperLinear(attn_module.v, layer_idx=i, param_type='v')
            except AttributeError:
                print(f"Warning: Could not replace layer {i} with HyperLinear. Check T5 structure.")


    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings
        self.visual_embedding.obj_order_embedding = new_embeddings


    def forward(
        self,
        input_ids=None,
        attention_mask=None,

        vis_inputs=None,
        vis_attention_mask=None,

        inputs_embeds=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        B, L = inputs_embeds.size()[:-1]

        vis_feats = vis_inputs[0]
        boxes = vis_inputs[1]
        img_order_ids = None
        obj_order_ids = None
        if len(vis_inputs) >= 3:
            img_order_ids = vis_inputs[2]
        if len(vis_inputs) == 4:
            obj_order_ids = vis_inputs[3]

        vis_embeds = self.visual_embedding(
            vis_feats, boxes, img_order_ids, obj_order_ids)

        V_L = vis_embeds.size(1)

        inputs_embeds = torch.cat([inputs_embeds, vis_embeds], dim=1)

        if past_key_values is None:
            past_key_values = [None] * len(self.block)

            if attention_mask is None:
                attention_mask = input_ids.ne(self.config.pad_token_id).to(dtype=inputs_embeds.dtype,
                                                                           device=inputs_embeds.device)

            if vis_attention_mask is None:
                vis_attention_mask = attention_mask.new_ones(B, V_L)

            attention_mask = torch.cat([attention_mask, vis_attention_mask], dim=1)

            extended_attention_mask = self.get_extended_attention_mask(attention_mask, (B, L + V_L),
                                                                       inputs_embeds.device)

        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None

        hidden_states = self.dropout(inputs_embeds)

        if self.config.num_layers > 0:
            assert self.block[0].layer[0].SelfAttention.has_relative_attention_bias

            seq_length = L + V_L

            q_len = seq_length
            k_len = seq_length

            text_position_bias = self.block[0].layer[0].SelfAttention.compute_bias(L, L)
            num_heads = text_position_bias.size(1)
            position_bias = text_position_bias.new_zeros(
                1, num_heads, seq_length, seq_length)
            position_bias[:, :, :L, :L] = text_position_bias

            position_bias = position_bias + extended_attention_mask

            for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    encoder_decoder_position_bias=None,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
                hidden_states, present_key_value_state = layer_outputs[:2]

                if len(layer_outputs) > 2:
                    position_bias = layer_outputs[2]

                if use_cache:
                    present_key_value_states = present_key_value_states + \
                        (present_key_value_state,)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


class VLT5(T5ForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config):
        super(T5ForConditionalGeneration, self).__init__(config)

        self.config = config

        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False

        self.encoder = JointEncoder(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False

        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.prototype_fc1 = nn.Linear(config.d_model, config.d_model)
        self.prototype_fc2 = nn.Linear(config.d_model, config.d_model)
        
        self.gate_layer = nn.Linear(3 * config.d_model, 2)
        
        self.L = 20
        self.V_L = 36

        self.init_weights()

        self.model_parallel = False
        self.device_map = None

        self.Q_task_mem_proto = {}
        self.V_task_mem_proto = {}
        self.Q_task_cur_proto = {}
        self.V_task_cur_proto = {}
        self.Q_prototype_num = {}
        self.V_prototype_num = {}
        print("Q_task_mem_proto and Q_task_cur_proto")
        
        self.hyper_generator = HyperLoRAGenerator(
            prototype_dim=config.d_model, 
            hidden_dim=64,
            adapter_dim=config.d_model,
            rank=8,
            num_layers=config.num_layers
        )
        

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def extend_vocab(self, vocab_size):

        new_shared = nn.Embedding(vocab_size, self.config.d_model)
        old_weight = self.shared.weight.data.detach().clone()
        old_vocab_size = old_weight.size(0)
        new_shared.weight.data[:old_vocab_size, :] = old_weight
        self.shared = new_shared

        new_lm_head = nn.Linear(self.config.d_model, vocab_size, bias=False)
        old_weight = self.lm_head.weight.data.detach().clone()
        old_vocab_size = old_weight.size(0)
        new_lm_head.weight.data[:old_vocab_size, :] = old_weight
        self.lm_head = new_lm_head

        self.vis_encoder.visual_embedding.obj_order_embedding = self.shared

        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

        self.lm_head.weight = self.shared.weight

        self.config.vocab_size = vocab_size
        self.encoder.config.vocab_size = vocab_size
        self.vis_encoder.config.vocab_size = vocab_size
        self.decoder.config.vocab_size = vocab_size


    def cosine_similarity_multi(self, a, b, labels=None, rep="real"):
        sim_act = nn.Tanh()
        a_normalized = F.normalize(sim_act(a), dim=1)
        b_normalized = F.normalize(sim_act(b), dim=1)
        similiarity = F.linear(a_normalized, b_normalized).transpose(1,0)
        max_idx = torch.argmax(similiarity, dim=1)
        selected_prototype = a[max_idx]

        if labels is not None:
            labels = torch.topk(labels, 1)[1].squeeze(1)
            acc = (max_idx == labels).sum()//labels.shape[0]
        else:
            acc = -1

        return selected_prototype, max_idx, acc


    def update_prototype(self, current_Q_prototype, current_V_prototype, current_num_Q, current_num_V, current_task_id, proto_alpha, proto_beta):

        if current_task_id not in self.Q_task_cur_proto:
            self.Q_task_cur_proto[current_task_id] = current_Q_prototype
            self.Q_prototype_num = current_num_Q
            self.V_prototype_num = current_num_V
            self.V_prototype = current_V_prototype
            if current_task_id == 0:
                self.Q_prototype = current_Q_prototype
            else:
                self.Q_prototype[current_task_id] = current_Q_prototype[current_task_id]
        else:

            self.Q_task_cur_proto[current_task_id] = current_Q_prototype

            if current_task_id != 0:
                if current_task_id not in self.Q_task_mem_proto:
                    current_Q_prototype_mem = current_Q_prototype.clone()
                    current_Q_prototype_mem[current_task_id] = 0
                    self.Q_task_mem_proto[current_task_id] = current_Q_prototype_mem
                else:
                    current_Q_prototype_mem = current_Q_prototype.clone()
                    current_Q_prototype_mem[current_task_id] = 0
                    self.Q_task_mem_proto[current_task_id] = proto_alpha*self.Q_task_mem_proto[current_task_id] + (1-proto_alpha)*current_Q_prototype_mem.detach()

                self.Q_prototype = self.Q_task_mem_proto[current_task_id].detach()
                self.Q_prototype[current_task_id] = self.Q_task_cur_proto[current_task_id][current_task_id].detach()
            else:
                self.Q_prototype = self.Q_task_cur_proto[current_task_id]


            self.V_prototype = proto_beta*self.V_prototype + (1-proto_beta)*current_V_prototype
            self.Q_prototype_num = self.Q_prototype_num.detach() + current_num_Q
            self.V_prototype_num = self.V_prototype_num.detach() + current_num_V

    def calculate_current_prototype(self, fc_hidden_Q, labels):
        fc_hidden_Q = torch.mean(fc_hidden_Q, dim=1)

        div_item_ = torch.sum(labels, dim=0).unsqueeze(1).repeat(1, 768)
        ones = torch.ones((labels.shape[1], fc_hidden_Q.shape[-1])).to(torch.device('cuda'))
        div_item = torch.where(div_item_ <= 0, ones, div_item_)

        current_prototype_Q = torch.matmul(torch.transpose(labels, 0, 1),
                                           fc_hidden_Q) / div_item

        current_num = torch.sum(labels, dim=0)
        return current_prototype_Q, current_num

    # ================= [新增] 计算 Semantic-Functional Alignment Loss =================
    def compute_alignment_loss(self, semantic_vectors, functional_params_a, functional_params_b):
        """
        计算对齐 Loss：
        1. Semantic Distance: 原型向量之间的余弦相似度矩阵
        2. Functional Distance: 生成参数之间的余弦相似度矩阵
        3. Loss: 两个矩阵的 MSE Loss
        """
        batch_size = semantic_vectors.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=semantic_vectors.device)

        # 1. Semantic Similarity Matrix
        # semantic_vectors: [B, Dim]
        sem_norm = F.normalize(semantic_vectors, dim=1)
        sem_sim_matrix = torch.mm(sem_norm, sem_norm.t()) # [B, B]

        # 2. Functional Similarity Matrix
        # 我们需要把生成的参数展平：[B, Layers, 2, Dim, Rank] -> [B, -1]
        flat_a = functional_params_a.view(batch_size, -1)
        flat_b = functional_params_b.view(batch_size, -1)
        # 拼接 A 和 B
        flat_params = torch.cat([flat_a, flat_b], dim=1) # [B, Huge_Dim]
        
        func_norm = F.normalize(flat_params, dim=1)
        func_sim_matrix = torch.mm(func_norm, func_norm.t()) # [B, B]

        # 3. Alignment Loss (MSE)
        # 强迫功能空间的拓扑结构 与 语义空间的拓扑结构 一致
        loss_align = F.mse_loss(func_sim_matrix, sem_sim_matrix)
        
        return loss_align
    # =============================================================================


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_outputs=None,

        vis_inputs=None,
        vis_attention_mask=None,

        decoder_input_ids=None,
        decoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        labels=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        reduce_loss=False,

        return_hidden_state=False,

        **kwargs,
    ):

        CURRENT_HYPER_PARAMS.clear()
        
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:

            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,

                vis_inputs=vis_inputs,
                vis_attention_mask=vis_attention_mask,

                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(
                    encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(
                    encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if 'cate_labels' in kwargs:
            cate_labels = kwargs['cate_labels']
        if 'ques_labels' in kwargs:
            ques_labels = kwargs['ques_labels']

        if 'proto_alpha' in kwargs:
            proto_alpha = kwargs['proto_alpha']
        if 'proto_beta' in kwargs:
            proto_beta = kwargs['proto_beta']

        if 'current_task_id' in kwargs:
            current_task_id = kwargs['current_task_id']

        if 'proto_update' in kwargs and kwargs['proto_update']:

            current_prototype_Q, current_num_Q = self.calculate_current_prototype(hidden_states[:, :self.L, :],
                                                                                  ques_labels)
            current_prototype_V, current_num_V = self.calculate_current_prototype(hidden_states[:, self.L:, :],
                                                                                  cate_labels)

            if 'memory' in kwargs and kwargs['memory'] == True:
                loss_memory_Q, loss_memory_V = self.memory_loss(hidden_states[:, :self.L, :],
                                                                hidden_states[:, self.L:, :], ques_labels, cate_labels)
            else:
                loss_memory_Q, loss_memory_V = 0, 0

            self.update_prototype(current_prototype_Q, current_prototype_V, current_num_Q, current_num_V,
                                  current_task_id, proto_alpha, proto_beta)

            retrievaled_Q_proto, max_idx_Q, acc_Q = self.cosine_similarity_multi(self.Q_prototype, torch.mean(
                hidden_states[:, :self.L, :], dim=1), ques_labels)
            retrievaled_Q_proto = retrievaled_Q_proto.unsqueeze(1)
            retrievaled_V_proto, max_idx_V, acc_V = self.cosine_similarity_multi(self.V_prototype, torch.mean(
                hidden_states[:, self.L:, :], dim=1), cate_labels)
            retrievaled_V_proto = retrievaled_V_proto.unsqueeze(1)
        else:
            retrievaled_Q_proto, max_idx_Q, acc_Q = self.cosine_similarity_multi(self.Q_prototype, torch.mean(hidden_states[:, :self.L, :], dim=1))
            retrievaled_Q_proto = retrievaled_Q_proto.unsqueeze(1)
            retrievaled_V_proto, max_idx_V, acc_V = self.cosine_similarity_multi(self.V_prototype, torch.mean(hidden_states[:, self.L:, :], dim=1))
            retrievaled_V_proto = retrievaled_V_proto.unsqueeze(1)
            loss_memory_Q, loss_memory_V = 0, 0


        Q_proto = retrievaled_Q_proto.detach()
        V_proto = retrievaled_V_proto.detach()
        
        hyper_input = Q_proto.squeeze(1) 
        
        # 生成参数
        gen_a, gen_b = self.hyper_generator(hyper_input)
        
        # === [新增] 计算 Alignment Loss ===
        # 我们希望生成的参数距离 与 输入的原型距离 保持一致
        loss_alignment = self.compute_alignment_loss(hyper_input, gen_a, gen_b)
        # ================================

        CURRENT_HYPER_PARAMS.clear()
        batch_size_gen = gen_a.size(0)
        
        for i in range(self.config.num_layers):
            q_a = gen_a[:, i, 0, :, :]
            q_b = gen_b[:, i, 0, :, :]
            v_a = gen_a[:, i, 1, :, :]
            v_b = gen_b[:, i, 1, :, :]
            
            CURRENT_HYPER_PARAMS[i] = {
                'q': (q_a, q_b),
                'v': (v_a, v_b)
            }

        pooled_hidden = torch.mean(hidden_states, dim=1)
        gate_input = torch.cat([pooled_hidden, 
                               Q_proto.squeeze(1), 
                               V_proto.squeeze(1)], dim=1)
        gate = torch.sigmoid(self.gate_layer(gate_input))
        gate_Q = gate[:, 0].unsqueeze(-1).unsqueeze(-1)
        gate_V = gate[:, 1].unsqueeze(-1).unsqueeze(-1)

        Q_proto_expanded = Q_proto.expand_as(hidden_states)
        V_proto_expanded = V_proto.expand_as(hidden_states)
        hidden_states = hidden_states + gate_Q * Q_proto_expanded + gate_V * V_proto_expanded
        


        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = self._shift_right(labels)

        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id).to(dtype=hidden_states.dtype, device=hidden_states.device)

        if vis_attention_mask is None:
            B, L = attention_mask.size()
            V_L = hidden_states.size(1) - L
            vis_attention_mask = attention_mask.new_ones(B, V_L)
        encoder_attention_mask = torch.cat([attention_mask, vis_attention_mask], dim=1)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,

            encoder_hidden_states=hidden_states,
            encoder_attention_mask=encoder_attention_mask,

            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        assert self.config.tie_word_embeddings is True

        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        if return_hidden_state:
            return sequence_output

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            if reduce_loss:
                loss_fct = CrossEntropyLoss(ignore_index=-100)
            else:
                loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
            loss = loss_fct(
                lm_logits.view(-1, lm_logits.size(-1)),
                labels.view(-1))

        return VLSeq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_last_hidden_state=decoder_outputs.last_hidden_state,
            decoder_hidden_states=decoder_outputs.hidden_states,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=encoder_attention_mask,
            loss_memory_Q = loss_memory_Q,
            loss_memory_V = loss_memory_V,
            loss_alignment = loss_alignment # 新增字段
            # ================================
        )
    
    def generate(self, input_ids=None, **kwargs):
        CURRENT_HYPER_PARAMS.clear()
        
        vis_inputs = kwargs.get('vis_inputs', None)
        vis_attention_mask = kwargs.get('vis_attention_mask', None)
        attention_mask = kwargs.get('attention_mask', None)
        
        encoder = self.get_encoder()
        
        if attention_mask is None and input_ids is not None:
            attention_mask = input_ids.ne(self.config.pad_token_id).to(input_ids.device)
            
        encoder_outputs = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            vis_inputs=vis_inputs,
            vis_attention_mask=vis_attention_mask,
            return_dict=True
        )
        
        hidden_states = encoder_outputs.last_hidden_state

        retrievaled_Q_proto, max_idx_Q, acc_Q = self.cosine_similarity_multi(
            self.Q_prototype, 
            torch.mean(hidden_states[:, :self.L, :], dim=1)
        )
        retrievaled_Q_proto = retrievaled_Q_proto.unsqueeze(1)

        retrievaled_V_proto, max_idx_V, acc_V = self.cosine_similarity_multi(
            self.V_prototype, 
            torch.mean(hidden_states[:, self.L:, :], dim=1)
        )
        retrievaled_V_proto = retrievaled_V_proto.unsqueeze(1)

        Q_proto = retrievaled_Q_proto.detach()
        V_proto = retrievaled_V_proto.detach()
        
        hyper_input = Q_proto.squeeze(1)
        gen_a, gen_b = self.hyper_generator(hyper_input)
        
        batch_size_gen = gen_a.size(0)
        for i in range(self.config.num_layers):
            q_a = gen_a[:, i, 0, :, :]
            q_b = gen_b[:, i, 0, :, :]
            v_a = gen_a[:, i, 1, :, :]
            v_b = gen_b[:, i, 1, :, :]
            
            CURRENT_HYPER_PARAMS[i] = {
                'q': (q_a, q_b),
                'v': (v_a, v_b)
            }
            
        pooled_hidden = torch.mean(hidden_states, dim=1)
        gate_input = torch.cat([pooled_hidden, 
                               Q_proto.squeeze(1), 
                               V_proto.squeeze(1)], dim=1)
        gate = torch.sigmoid(self.gate_layer(gate_input))
        gate_Q = gate[:, 0].unsqueeze(-1).unsqueeze(-1)
        gate_V = gate[:, 1].unsqueeze(-1).unsqueeze(-1)

        Q_proto_expanded = Q_proto.expand_as(hidden_states)
        V_proto_expanded = V_proto.expand_as(hidden_states)
        
        hidden_states_fused = hidden_states + gate_Q * Q_proto_expanded + gate_V * V_proto_expanded
        
        from transformers.modeling_outputs import BaseModelOutput
        new_encoder_outputs = BaseModelOutput(
            last_hidden_state=hidden_states_fused,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions
        )
        
        return super().generate(
            encoder_outputs=new_encoder_outputs,
            attention_mask=attention_mask,
            **kwargs
        )

    def prepare_inputs_for_generation(
        self, input_ids, past=None, attention_mask=None, use_cache=None,
        encoder_outputs=None,
        **kwargs):

        if past is not None:
            input_ids = input_ids[:, -1:]

        output = {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }

        if 'vis_attention_mask' in kwargs:
            output['vis_attention_mask'] = kwargs['vis_attention_mask']

        return output

    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: torch.LongTensor = None,
        encoder_outputs: ModelOutput = None,
        **model_kwargs
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1,
                                                                expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(
                0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(
                0, expanded_return_idx)

        if model_kwargs.get("vis_attention_mask", None) is not None:
            model_kwargs['vis_attention_mask'] = model_kwargs['vis_attention_mask'].index_select(
                0, expanded_return_idx)

        if is_encoder_decoder:
            assert encoder_outputs is not None
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx
            )
            model_kwargs["encoder_outputs"] = encoder_outputs

        return input_ids, model_kwargs

@dataclass
class VLSeq2SeqLMOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    decoder_last_hidden_state: Optional[Tuple[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None

    vis_encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    vis_encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    vis_encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None

    encoder_attention_mask: Optional[Tuple[torch.FloatTensor]] = None
    loss_memory_Q: torch.FloatTensor = None
    loss_memory_V: torch.FloatTensor = None
    
    # === [新增] ===
    loss_alignment: Optional[torch.FloatTensor] = None
