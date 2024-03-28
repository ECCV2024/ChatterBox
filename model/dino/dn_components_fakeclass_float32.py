# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# DN-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]


import torch
from torch import tensor

from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
# from .DABDETR import sigmoid_focal_loss
from util import box_ops
import torch.nn.functional as F


def prepare_for_cdn(dn_args, training, num_queries, num_classes, hidden_dim, label_enc, sparse_embedding=None, compress_text=None):
    """
        A major difference of DINO from DN-DETR is that the author process pattern embedding pattern embedding in its detector
        forward function and use learnable tgt embedding, so we change this function a little bit.
        :param dn_args: targets, dn_number, label_noise_ratio, box_noise_scale
        :param training: if it is training or inference
        :param num_queries: number of queires
        :param num_classes: number of classes
        :param hidden_dim: transformer hidden dim
        :param label_enc: encode labels in dn
        :return:
        """
    if training:
        print('dn_args:{}'.format(dn_args))
        targets, dn_number, label_noise_ratio, box_noise_scale = dn_args
        #####################################
        num_category = label_enc.weight.shape[0]-1
        print(num_category)
        print('targets_before_add_label:{}'.format(targets))
        for t in targets:
            print('t:{}'.format(t))
            if 'labels' not in t.keys():
                print('labels not in targets.keys()')
            t['labels'] = torch.randint(0,num_category, (t['boxes'].shape[0],)).to(t['boxes'].device)
            print("t['labels'].dtype:{}".format(t['labels'].dtype))
        print('targets_after_add_label:{}'.format(targets))
        #####################################
        # positive and negative dn queries
        dn_number = dn_number * 2
        known = [(torch.ones_like(t['labels'])).cuda() for t in targets]
        batch_size = len(known)
        known_num = [sum(k) for k in known]
        if int(max(known_num)) == 0:
            dn_number = 1
        else:
            if dn_number >= 100:
                dn_number = dn_number // (int(max(known_num) * 2))
            elif dn_number < 1:
                dn_number = 1
        if dn_number == 0:
            dn_number = 1
        unmask_bbox = unmask_label = torch.cat(known)
        labels = torch.cat([t['labels'] for t in targets])
        # print('labels.dtype:{}'.format(labels.dtype))
        boxes = torch.cat([t['boxes'] for t in targets])
        batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])

        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)

        known_indice = known_indice.repeat(2 * dn_number, 1).view(-1)
        known_labels = labels.repeat(2 * dn_number, 1).view(-1)
        known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)
        known_bboxs = boxes.repeat(2 * dn_number, 1)
        known_labels_expaned = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        ###########################
        #sparse_embedding_expand=torch.zeros(known_labels_expaned.shape[0],label_enc.weight.shape[1])
        # print('sparse_embedding.shape:{}'.format(sparse_embedding.shape))
        # print('sparse_embedding.dtype:{}'.format(sparse_embedding.dtype))
        # sparse_embedding=tensor(sparse_embedding,dtype=torch.float32)
        # print('sparse_embedding.dtype:{}'.format(sparse_embedding.dtype))

        sparse_embedding=compress_text(sparse_embedding)
        print('compressed_sparse_embedding.dtype:{}'.format(sparse_embedding.dtype))
        print('compressed_sparse_embedding.shape:{}'.format(sparse_embedding.shape))
        new_sparse_embedding_repeat_list=[]
        for cnt, t in enumerate(targets):
            print('sparse_embedding.dtype:{}'.format(sparse_embedding.dtype))
            new_sparse_embedding = sparse_embedding
            print('new_sparse_embedding.dtype:{}'.format(new_sparse_embedding.dtype))
            new_sparse_embedding_repeat = new_sparse_embedding.unsqueeze(0).repeat(t['labels'].shape[0], 1)
            print('new_sparse_embedding_repeat.dtype:{}'.format(new_sparse_embedding_repeat.dtype))
            print('new_sparse_embedding_repeat.shape:{}'.format(new_sparse_embedding_repeat.shape))
            new_sparse_embedding_repeat_list.append(new_sparse_embedding_repeat)
        print('new_sparse_embedding_repeat_list:{}'.format(new_sparse_embedding_repeat_list))
        new_sparse_embedding_repeat=torch.cat([i for i in new_sparse_embedding_repeat_list],dim=0)
        print('new_sparse_embedding_repeat.shape:{}'.format(new_sparse_embedding_repeat.shape))
        print('new_sparse_embedding_repeat.dtype:{}'.format(new_sparse_embedding_repeat.dtype))
        new_sparse_embedding_expand=new_sparse_embedding_repeat.repeat(2*dn_number,1)
        print('new_sparse_embedding_expand.shape:{}'.format(new_sparse_embedding_expand.shape))
        print('new_sparse_embedding_expand.device:{}'.format(new_sparse_embedding_expand.device))
        print('new_sparse_embedding_repeat.dtype:{}'.format(new_sparse_embedding_repeat.dtype))
        ###########################
        if label_noise_ratio > 0:
            p = torch.rand_like(known_labels_expaned.float())
            chosen_indice = torch.nonzero(p < (label_noise_ratio * 0.5)).view(-1)  # half of bbox prob
            new_label = torch.randint_like(chosen_indice, 0, num_classes)  # randomly put a new one here
            known_labels_expaned.scatter_(0, chosen_indice, new_label)
            #########################
            unknown_label=label_enc.weight.shape[0]-1
            print('unknown_label:{}'.format(unknown_label))
            unknown_label_embedding = label_enc(tensor([unknown_label]).to(new_sparse_embedding_expand.device))
            print('unknown_label_embedding.shape:{}'.format(unknown_label_embedding.shape))
            print('chosen_indice.shape:{}'.format(chosen_indice.shape))
            #print('unknown_label_embedding.unsqueeze(0).shape{}'.format(unknown_label_embedding.unsqueeze(0).shape))
            #print('unknown_label_embedding.unsqueeze(0).repeat(chosen_indice.shape[0], 1).shape:{}'.format(unknown_label_embedding.unsqueeze(0).repeat(chosen_indice.shape[0], 1).shape))
            print('unknown_label_embedding.repeat(chosen_indice.shape[0], 1).shape:{}'.format(unknown_label_embedding.repeat(chosen_indice.shape[0], 1).shape))
            print('chosen_indice.unsqueeze(1).repeat(1,unknown_label_embedding.shape[1].shape:{}'.format(chosen_indice.unsqueeze(1).repeat(1,unknown_label_embedding.shape[1]).shape))
            
            new_sparse_embedding_expand.scatter_(0, chosen_indice.unsqueeze(1).repeat(1,unknown_label_embedding.shape[1]),
                                                 unknown_label_embedding.repeat(chosen_indice.shape[0], 1))
            print('new_sparse_embedding_expand.shape:{}'.format(new_sparse_embedding_expand.shape))
            # unknown_label2=label_enc.weight.shape[0]
            # unknown_label_embedding2=label_enc(tensor([unknown_label2]))
            # new_sparse_embedding_expand.scatter_(0,
            #                                      chosen_indice.unsqueeze(1).repeat(1, unknown_label_embedding.shape[1]),
            #                                      unknown_label_embedding.repeat(chosen_indice.shape[0], 1))

            #sparse_embedding_expand.scatter_(0,chosen_indice,sparse_embedding.repeat(new_label.shape[0]))
            #########################
        single_pad = int(max(known_num))

        pad_size = int(single_pad * 2 * dn_number)
        positive_idx = torch.tensor(range(len(boxes))).long().cuda().unsqueeze(0).repeat(dn_number, 1)
        positive_idx += (torch.tensor(range(dn_number)) * len(boxes) * 2).long().cuda().unsqueeze(1)
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(boxes)
        if box_noise_scale > 0:
            known_bbox_ = torch.zeros_like(known_bboxs)
            known_bbox_[:, :2] = known_bboxs[:, :2] - known_bboxs[:, 2:] / 2
            known_bbox_[:, 2:] = known_bboxs[:, :2] + known_bboxs[:, 2:] / 2

            diff = torch.zeros_like(known_bboxs)
            diff[:, :2] = known_bboxs[:, 2:] / 2
            diff[:, 2:] = known_bboxs[:, 2:] / 2

            rand_sign = torch.randint_like(known_bboxs, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
            rand_part = torch.rand_like(known_bboxs)
            rand_part[negative_idx] += 1.0
            rand_part *= rand_sign
            known_bbox_ = known_bbox_ + torch.mul(rand_part,
                                                  diff).cuda() * box_noise_scale
            known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)
            known_bbox_expand[:, :2] = (known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2
            known_bbox_expand[:, 2:] = known_bbox_[:, 2:] - known_bbox_[:, :2]

        m = known_labels_expaned.long().to('cuda')
        print('m.dtype:{}'.format(m.dtype))
        print('known_labels_expaned.dtype:{}'.format(known_labels_expaned.dtype))
        print('known_labels_expaned.dtype:{}'.format(known_labels_expaned.dtype))
        input_label_embed = label_enc(m)
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)

        padding_label = torch.zeros(pad_size, hidden_dim).cuda()
        padding_bbox = torch.zeros(pad_size, 4).cuda()

        input_query_label = padding_label.repeat(batch_size, 1, 1)
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

        map_known_indice = torch.tensor([]).to('cuda')
        if len(known_num):
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
            map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(2 * dn_number)]).long()
        if len(known_bid):
            print('input_query_label.dtype:{}'.format(input_query_label.dtype))
            print('map_known_indice.dtype:{}'.format(map_known_indice.dtype))
            print('input_label_embed.dtype:{}'.format(input_label_embed.dtype))
            print('input_bbox_embed.dtype:{}'.format(input_bbox_embed.dtype))
            print('new_sparse_embedding_expand.dtype:{}'.format(new_sparse_embedding_expand.dtype))
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed
            ######################
            print('new_sparse_embedding_expand.shape:{}'.format(new_sparse_embedding_expand.shape))
            print('input_label_embed.shape:{}'.format(input_label_embed.shape))

            assert new_sparse_embedding_expand.shape == input_label_embed.shape
            assert new_sparse_embedding_expand.device == input_label_embed.device
            assert new_sparse_embedding_expand.dtype == input_label_embed.dtype
            input_query_label[(known_bid.long(), map_known_indice)] = new_sparse_embedding_expand
            #####################

        tgt_size = pad_size + num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(dn_number):
            if i == 0:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
            if i == dn_number - 1:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * i * 2] = True
            else:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * 2 * i] = True

        dn_meta = {
            'pad_size': pad_size,
            'num_dn_group': dn_number,
        }
        idx=[positive_idx,negative_idx]
    else:

        input_query_label = None
        input_query_bbox = None
        attn_mask = None
        dn_meta = None
        idx=None

    return input_query_label, input_query_bbox, attn_mask, dn_meta,idx


def dn_post_process(outputs_class, outputs_coord, dn_meta, aux_loss, _set_aux_loss):
    """
        post process of dn after output from the transformer
        put the dn part in the dn_meta
    """
    if dn_meta and dn_meta['pad_size'] > 0:
        output_known_class = outputs_class[:, :, :dn_meta['pad_size'], :]
        output_known_coord = outputs_coord[:, :, :dn_meta['pad_size'], :]
        outputs_class = outputs_class[:, :, dn_meta['pad_size']:, :]
        outputs_coord = outputs_coord[:, :, dn_meta['pad_size']:, :]
        out = {'pred_logits': output_known_class[-1], 'pred_boxes': output_known_coord[-1]}
        if aux_loss:
            out['aux_outputs'] = _set_aux_loss(output_known_class, output_known_coord)
        dn_meta['output_known_lbs_bboxes'] = out
    return outputs_class, outputs_coord


