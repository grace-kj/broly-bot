from dense_retriever import *
import subprocess
import numpy as np
import pandas as pd
import json

def run_retriever():

  # 질문 text에 대해서 inference
  subprocess.call("python dense_retriever.py \
	model_file='/content/drive/MyDrive/Colab Notebooks/DPR/model_checkpoint/dpr_pdf_notice_wt_whole2/dpr_biencoder.4' \
	qa_dataset=pdf_validation \
	ctx_datatsets=[pdf_wt_passage] \
	encoded_ctx_files=['/content/drive/MyDrive/Colab Notebooks/DPR/outputs/pdf_wt_passage_4_0'] \
	out_file='/content/drive/MyDrive/Colab Notebooks/DPR/outputs/pdf_validation_4_0.json'", shell=True)

  # 질문 subtitle에 대해서 inference
  subprocess.call("python dense_retriever.py \
	model_file='/content/drive/MyDrive/Colab Notebooks/DPR/model_checkpoint/dpr_pdf_notice_wt_whole2/dpr_biencoder.4' \
	qa_dataset=pdf_validation \
	ctx_datatsets=[pdf_notice_subtitle_passage] \
	encoded_ctx_files=['/content/drive/MyDrive/Colab Notebooks/DPR/outputs/pdf_notice_subtitle_passage_4_0'] \
	out_file='/content/drive/MyDrive/Colab Notebooks/DPR/outputs/pdf_notice_subtitle_validation_4_0.json'", shell=True)


def ensemble():

  # text 점수
  json_file_path = "/content/drive/MyDrive/Colab Notebooks/DPR/outputs/pdf_validation_4_0.json"
  with open(json_file_path, 'r') as j:
    contents = json.loads(j.read()) 
  content = pd.DataFrame(contents)

  # subtitle 점수
  json_file_path = "/content/drive/MyDrive/Colab Notebooks/DPR/outputs/pdf_notice_subtitle_validation_4_0.json"
  with open(json_file_path, 'r') as j:
    subtitle_contents = json.loads(j.read()) 
  subtitle_content = pd.DataFrame(subtitle_contents)

  # 점수 합산
  k = 7 # retriever에서 뽑은 문서 개수
  score_list = []
  
  for i in range(len(contents)):
    # 한 질문에서 나오는 topk 리스트
    content_ctx = content['ctxs'][i]
    subtitle_ctx = subtitle_content['ctxs'][i]

    content_score = {}
    subtitle_score = {}
    content_id = []
    subtitle_id = []
    content_title = {}
    content_text = {}
    subtitle_title = {}
    subtitle_text = {}

    for j in range(k):
      cid = int(content_ctx[j]['id'])
      content_id.append(cid)
      content_score[cid] = content_ctx[j]['score']
      content_title[cid] = content_ctx[j]['title']
      content_text[cid] = content_ctx[j]['text']

      sid = int(subtitle_ctx[j]['id'])
      subtitle_id.append(sid)
      subtitle_score[sid] = subtitle_ctx[j]['score']
      subtitle_title[sid] = subtitle_ctx[j]['title']
      subtitle_text[sid] = subtitle_ctx[j]['text']

    # 교집합 먼저 계산
    intersection = list(set(content_id) & set(subtitle_id))
    complement_content = list(set(content_id) - set(subtitle_id))
    complement_subtitle = list(set(subtitle_id) - set(content_id))
    
    # 합산 점수 계산
    final_score = {}
    for doc_id in intersection:
      final_score[doc_id] = 1.5 * float(content_score[doc_id]) + 1.5 * float(subtitle_score[doc_id])
    for doc_id in complement_content:
      final_score[doc_id] = float(content_score[doc_id])
    for doc_id in complement_subtitle:
      final_score[doc_id] = 1.7 * float(subtitle_score[doc_id])
      
    # 합산 점수 소팅
    sorted_score = sorted(final_score.items(), key=lambda x: x[1], reverse=True)
    sorted_score = sorted_score[:5]
    
    # 딕셔너리 합쳐주기
#    ret_list = []
#    for id, score in sorted_score:
#      ret_dict = {}
#      ret_dict['id'] = id
#      ret_dict['score'] = score
#      if id in complement_subtitle:
#        ret_dict['text'] = subtitle_text[id]
#        ret_dict['title'] = subtitle_title[id]
#      else:
#        ret_dict['text'] = content_text[id]
#        ret_dict['title'] = content_title[id]
        
#      ret_list.append(ret_dict)

    score_list.append(sorted_score)
  
  return score_list


if __name__ == '__main__':

  query = input("질문을 입력해주세요")

  # 질문 모델 인풋 형식으로 전처리
  new_question = [[query, "['" + 'retriever_new_question' + "']"]]
  validation_data = pd.DataFrame(new_question)
  validation_data.to_csv("pdf_validation.csv", mode='w', sep='\t', index=False, header=None)

  # retriever로 inference
  run_retriever()

  # 점수 합산해서 앙상블
  ids = ensemble()
  print(ids)