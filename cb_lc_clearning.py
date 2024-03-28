import json
import os
import shutil

def divide_qa(conversation):
    #"conversation": [
  # {
  #  "from": "human",
  #  "value": "What is the object on top of the building? <building:[191, 336, 766, 853]>"
  # },
    #   {
    #    "from": "gpt",
    #    "value": "The objects on top of the building are a belfry and a green dome. <belfry:[239, 369, 370, 536], green dome:[258, 369, 347, 430]> "
    #   },
    # ] 
    if len(conversation)==0:
        return []
    assert len(conversation)%2 ==0
    round_qa=[]
    for cnt,c in enumerate(conversation):
        if c['from']=='human':
            question_anno_list = []
            question=c['value']
            if '<' in question and '>' in question and '[' in question and ']' in question:
                q_sentence,anno=question.split('<')
                anno=anno.split('>')[0]
                anno_list=anno.split('],')
                for a in anno_list:
                    a=a.strip(' ')
                    question_anno_list.append(a)
            else:
                return None
        elif c['from']=='gpt':
            answer_anno_list = []
            answer=c['value']
            if '<' in answer and '>' in answer and '[' in answer and ']' in answer:
                a_sentence, anno = answer.split('<')
                anno = anno.split('>')[0]
                anno_list = anno.split('],')
                for a in anno_list:
                    a = a.strip(' ')
                    answer_anno_list.append(a)
            else:
                return None
        if cnt % 2 !=0:
            round_qa.append([q_sentence,question_anno_list,a_sentence,answer_anno_list])
    return round_qa

def qa_relation_object(round_qa,relation_chain_2_path,id):
    #for single obj
    with open(relation_chain_2_path,'r') as fr:
        file=json.load(fr)
        relation_info=file[str(id)]
        relations=relation_info["relation"]
        for cnt,r in enumerate(relations):
            object_attribute_name=r["object_attribute_name"]
            object_bbox=r['object_bbox']
            subject_attribute_name=r['subject_attribute_name']
            subject_bbox=r['subject_bbox']
            qa=round_qa[cnt]
            q_sentence, question_anno_list, a_sentence, answer_anno_list=qa
            # assert len(question_anno_list)==1
            # assert len(answer_anno_list)==1
            if len(question_anno_list)==1 and len(answer_anno_list)==1:
                object_name_bbox=object_attribute_name+':'+str(object_bbox)
                subject_name_bbox=subject_attribute_name+':'+str(subject_bbox)
                if question_anno_list[0]==subject_name_bbox or question_anno_list[0] in subject_name_bbox or subject_name_bbox in question_anno_list[0]:
                    if answer_anno_list[0]==object_name_bbox or answer_anno_list[0] in object_name_bbox or object_name_bbox in answer_anno_list[0]:
                        pass
                    else:
                        return 'delete'
                else:
                    return 'delete'
            else:
                return 'delete'
            # question_name,question_bbox=question_anno_list.split(':')
            # answer_name,answer_bbox=answer_anno_list.split(':')
            # if question_name==subject_attribute_name or question_name in subject_attribute_name or subject_attribute_name in question_name:
            #     if answer_name==object_attribute_name or answer_name in object_attribute_name or object_attribute_name in answer_name:
            #         pass
            # else:
            #     return 'delete'
        return round_qa

def qa_relation_chain(round_qa):
    question_obj_list=[]
    answer_obj_list=[]
    for r in round_qa:
        assert len(r[1])==1
        assert len(r[3])==1
        question_obj_list.append(r[1][0])
        answer_obj_list.append(r[3][0])
    assert len(question_obj_list)==len(answer_obj_list)
    for i in range(len(question_obj_list)-1):
        if answer_obj_list[i] != question_obj_list[i+1]:
            return 'delete'
    return round_qa

def object_not_in_qa(round_qa):
    for r in round_qa:
        if ':' in r[1][0]:
            name1=r[1][0].split(':')[0]
            if ' ' in name1:
                name1=name1.split(' ')[-1]
        else:
            return 'delete'
        if ':' in r[3][0]:
            name2=r[3][0].split(':')[0]
            if ' ' in name2:
                name2=name2.split(' ')[-1]
        else:
            return 'delete'
        if name1 not in r[0]:
            return 'delete'
        if name2 not in r[2]:
            return 'delete'
    return round_qa

def filter_relation(path,relation_chain_2_path):
    with open(path,'r') as fr:
        file=json.load(fr)
        id=file['id']
        conversation=file['conversation']
        assert len(conversation)%2==0
        round_qa = divide_qa(conversation)
        if round_qa is not None and round_qa != []:
            type1=qa_relation_object(round_qa, relation_chain_2_path, id)
            if type1 != 'delete':
                type2=qa_relation_chain(round_qa)
                if type2 != 'delete':
                    type3=object_not_in_qa(round_qa)
                    return type3
                else:
                    return type2
            else:
                return type1
        else:
            return 'delete'

def single_path(base_path,relation_chain_2_path,save_path):
    sum_save=0
    sum_all=0
    dirs=os.listdir(base_path)
    for d in dirs:
        path2=os.path.join(base_path,d)
        files=os.listdir(path2)
        for f in files:
            sum_all+=1
            file_path=os.path.join(path2,f)
            a=filter_relation(file_path,relation_chain_2_path)
            if a != 'delete':
                sum_save+=1
                old_path=file_path
                new_path=os.path.join(save_path,d)
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                new_path=os.path.join(new_path,f)
                shutil.copy(old_path,new_path)
    print('remain ratio:{}'.format(sum_save/sum_all))


#path='/home/ubuntu/Workspace/ZhangYuan/code/visual_genome_batch/chain/object/batch/response_4/0/10.json'
#path='/home/ubuntu/Workspace/ZhangYuan/code/visual_genome_batch/chain/object/batch/response_4/0/8.json'
#path='/home/ubuntu/Workspace/ZhangYuan/code/visual_genome_batch/chain/object/batch/response_4/1/2402028.json'
relation_chain_2_path='/home/ubuntu/Workspace/ZhangYuan/code/visual_genome_batch/chain/object/filter_batch/relation_chain_2_len3.json'
base_path='/home/ubuntu/Workspace/ZhangYuan/code/visual_genome_batch/chain/object_v2/batch/response_3'
save_path='/home/ubuntu/Workspace/ZhangYuan/code/visual_genome_batch/chain/object_v2/filter_batch/filter_response_step1'
if not os.path.exists(save_path):
    os.makedirs(save_path)
single_path(base_path,relation_chain_2_path,save_path)


