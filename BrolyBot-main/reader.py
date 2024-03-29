from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
from rouge import Rouge
import torch
import pandas as pd
import numpy as np
import pickle

from tkinter import *
from PIL import ImageTk,Image
 
from PIL import Image
import requests
import matplotlib.pyplot as plt


class Reader():
    def __init__(
        self,
        threshold = 10000,
        pretrained_model = "skt/kogpt2-base-v2",
        checkpoint = "model/model_checkpoint/reader_checkpoint.pt"):


        self.threshold = 10000

        ## model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(pretrained_model)
        self.model = GPT2LMHeadModel.from_pretrained(pretrained_model)
        self.model.to(self.device)
        self.model.eval()
        checkpoint = torch.load(checkpoint)
        self.model.load_state_dict(checkpoint)

        ## data
        self.notice_data = pd.read_csv("data/notice_data_0_1000.csv", index_col='id')
        self.notice_meta_data = pd.read_csv("data/notice_meta_data_0_1000.csv")
        self.pdf_data = pd.read_csv("data/pdf_data.csv", index_col='id')
        with open("data/pdf_meta_data2.pkl", 'rb') as f:
            self.pdf_meta_data = pickle.load(f)

    def read(self, query, id_list):
        for id in id_list:
            if id < self.threshold:
                self.notice(query, id)
            
            else:
                self.pdf(query, id)
    
    def only_generate(self, query, id):
        if id < self.threshold:
            return ""
            # notice = self.notice_data.loc[id]
            # content = notice['content']
            # if not pd.isnull(content):
            #     return self.generate(context = content, query=query)
            # else:
            #     return notice['title']
        else:
            pdf = self.pdf_data.loc[id]
            content = pdf['content']
            if not pd.isnull(content):
                return self.generate(context = content, query=query)
            elif pdf['meta_data'] == -1:
                return " ".join([pdf['title'], pdf['subtitle']])

    def notice(self, query, id):
        notice = self.notice_data.loc[id]

        ## 1. generate text
        content = notice['content']
        if not pd.isnull(content):
            generated = self.generate(context = content, query=query)
            print(generated)
        else:
            print(notice['title'])
        
        ## 2. link
        print(f"공지사항 링크 >> {notice['url']}")

        ## 3. meta data
        if notice['meta_data'] > 0:
            meta_data = self.notice_meta_data[self.notice_meta_data['id'] == id]
            for _, md in meta_data.iterrows():
                if md['data_type'] == 'image':
                    try:
                        im = Image.open(requests.get(md['data'], stream=True).raw)

                        ###### 1
                        # im.show()

                        ###### 2
                        # plt.imshow(im)
                        # plt.show()

                        ##### 3
                        # root = Tk()
                        # root.title(notice['title'])
                        # root.geometry('800x600')
                        # img = ImageTk.PhotoImage(im)
                        # label = Label(image=img)
                        # label.pack()
                        # quit = Button(root, text='종료하기', command=root.quit)
                        # quit.pack()
                        # root.mainloop()
                    except:
                        print("")


                elif md['data_type'] == 'attachment':
                    print(f"첨부 파일 링크 >> {md['data']}")
    
    def pdf(self, query, id):
        pdf = self.pdf_data.loc[id]

        ## 1. generate text
        content = pdf['content']
        if not pd.isnull(content):
            generated = self.generate(context = content, query=query)
            print(generated)
        elif pdf['meta_data'] == -1:
            print(pdf['title'], '-' , pdf['subtitle'])
        
        ## 2. link
        print(f"학사제도 e-book >> {pdf['url']}")

        ## 3. metadata
        if pdf['meta_data']:
            print(pdf['title'], '-' , pdf['subtitle'])
            print(self.pdf_meta_data[id])
    
    def generate(self, query, context):
        q_token = '<unused0>'
        a_token = '<unused1>'
        c_token = '<unused2>'
        bos_token = '</s>'
        eos_token = '</s>'
        mask_token = '<mask>'
        pad_token = '<pad>'

        with torch.no_grad():
            total_len = len(self.tokenizer.encode(context)) + len(self.tokenizer.encode(query)) + 3
            if total_len > 1024:
                context = context[:1024-total_len]
            input = c_token + context + q_token + query + bos_token
            input_ids = self.tokenizer.encode(input, return_tensors='pt').to(self.device)
            assert len(input_ids) < 1024, input_ids
            output = self.model.generate(
                inputs= input_ids,
                max_length=512,
                no_repeat_ngram_size=5)
            output = self.tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
        return output
    
    def evaluate(self, y, pred):
        rouge = Rouge()
        score = rouge.get_scores(pred, y, avg = True)
        return score

if __name__ == '__main__':
    reader = Reader()
    print(reader.only_generate("학사 일정", 10000))