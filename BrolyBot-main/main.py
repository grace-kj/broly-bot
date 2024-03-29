from retriever import run_retriever, ensemble
import pandas as pd
from reader import Reader

def main():
    reader = Reader()

    query = input("질문을 입력해주세요")

    # 질문 모델 인풋 형식으로 전처리
    new_question = [[query, "['" + 'retriever_new_question' + "']"]]
    validation_data = pd.DataFrame(new_question)
    validation_data.to_csv("pdf_validation.csv", mode='w', sep='\t', index=False, header=None)

    # retriever로 inference
    run_retriever()

    # 점수 합산해서 앙상블
    ids = ensemble()

    for i in ids:
        reader.read(i)


if __name__ == '__main__':
    main()