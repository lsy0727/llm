# pdf 
import pdfplumber

import pandas as pd # .csv, .xlsx 처리
import os, shutil, socket, subprocess, time
import re
import time

# from glob import glob
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import SentenceTransformerEmbeddings  # 더이상 사용되지 않음
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import Runnable
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

llm_model = "gemma3:12b"
embedding_model = "text-embedding-3-large"
# embedding_model = "jhgan/ko-sroberta-multitask"
# embedding_model = "all-MiniLM-L6-v2"
# embedding_model = "BAAI/bge-m3"

c_size = 200    # 청크 사이즈
c_overlab = 50  # 청크 오버랩

# 텍스트 후처리
def clean_text(text):
    # 줄바꿈 제거 및 불필요한 공백 정리
    text = re.sub(r"\s+", " ", text)
    # 특수 문자 제거 (필요시)
    text = re.sub(r"[^\w\s.,:;()가-힣A-Za-z]", "", text)
    return text.strip()

# Ollama 실행 여부 확인
def is_ollama_running(host='localhost', port=11434):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        try:
            sock.connect((host, port))
            return True
        except socket.error:
            return False

# Ollama 자동 실행
def start_ollama_if_needed(model_name="gemma3:12b"):
    if not is_ollama_running():
        subprocess.Popen(["ollama", "run", model_name], creationflags=subprocess.CREATE_NO_WINDOW)
        for _ in range(20):
            if is_ollama_running():
                break
            time.sleep(0.5)

# ✅ 여러 PDF 불러오기
def load_files_from_directory(directory_path: str):
    supported_exts = [".pdf", ".csv", ".xlsx", ".xls"]
    documents = []

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        ext = os.path.splitext(filename)[-1].lower()

        if ext not in supported_exts:
            print(f"⚠️ 지원되지 않는 형식의 파일 무시됨: {filename}")
            continue

        try:
            if ext == ".pdf":   # PDF: 페이지 단위
                with pdfplumber.open(file_path) as pdf:
                    for i, page in enumerate(pdf.pages):
                        text = clean_text(page.extract_text())
                        if text:
                            documents.append(Document(
                                page_content=text,
                                metadata={
                                    "source": filename,
                                    "page": i + 1,
                                    "type": "pdf",
                                    "timestamp": time.time()
                                }
                            ))

            elif ext == ".csv": # CSV: row 단위
                df = pd.read_csv(file_path)
                for i, row in df.iterrows():
                    row_text = ", ".join(f"{col}: {row[col]}" for col in df.columns)
                    documents.append(Document(
                        page_content=row_text,
                        metadata={"source": filename, "row": i + 1}
                    ))

            elif ext in [".xlsx", ".xls"]:  # 엑셀: row 단위
                df = pd.read_excel(file_path)
                for i, row in df.iterrows():
                    row_text = ", ".join(f"{col}: {row[col]}" for col in df.columns)
                    documents.append(Document(
                        page_content=row_text,
                        metadata={"source": filename, "row": i + 1}
                    ))

        except Exception as e:
            print(f"❌ 파일 처리 중 오류 발생: {filename} → {e}")

    return documents

# 벡터 DB 구축
def setup_vector_db_from_pdfs(folder_path):
    start_ollama_if_needed("gemma3:12b")
    all_documents = load_files_from_directory(folder_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=c_size, chunk_overlap=c_overlab)
    chunks = splitter.split_documents(all_documents)

    embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cuda"}
    )
    # 임베딩
    if os.path.exists("./file_db"):
        # shutil.rmtree("./file_db")
        vectordb = Chroma(persist_directory="./file_db", embedding_function=embedding)
    else:
        vectordb = Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory="./file_db")

    llm = Ollama(model="gemma3:12b")
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    # qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())

    return qa_chain

# 메인 실행
if __name__ == "__main__":
    folder_path = "C:/Users/Intel/Desktop/llm_ws/pdf_dir"  # ✅ 여기에 PDF들이 들어있는 폴더 경로 지정

    if not os.path.exists(folder_path):
        print(f"❌ 폴더가 존재하지 않습니다: {folder_path}")
        exit()

    try:
        qa_chain = setup_vector_db_from_pdfs(folder_path)
    except Exception as e:
        print(f"❌ 문서 불러오기 실패: {e}")
        exit()

    while True:
        question = input("\n❓ 질문을 입력하세요 (종료하려면 'exit'): ").strip()
        if question.lower() == "exit":
            print("🔚 종료합니다.")
            break

        try:
            prompt = f"""
                질문: {question}

                아래 문서들을 기반으로 정확하고 구체적인 한국어 답변을 생성하세요.
                - 가능한 모든 문서 정보를 반영
                - 중복 없이 요약
                - 반드시 한국어로 답변
                """
            start_time = time.time()
            answer = qa_chain.run(prompt)
            end_time = time.time()
            print(f"\n💬 답변:\n{answer}")
            print(f"응답 시간: {end_time - start_time:.2f}초")
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
