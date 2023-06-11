import argparse
import json
import os
import shutil
from typing import List, Optional
import urllib

import nltk
import pydantic
import uvicorn
from fastapi import Body, FastAPI, File, Form, Query, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing_extensions import Annotated
from starlette.responses import RedirectResponse

from chains.local_doc_qa import LocalDocQA
from configs.model_config import (VS_ROOT_PATH, UPLOAD_ROOT_PATH, EMBEDDING_DEVICE,
                                  EMBEDDING_MODEL, NLTK_DATA_PATH,
                                  VECTOR_SEARCH_TOP_K, OPEN_CROSS_DOMAIN)
import models.shared as shared
from models.loader.args import parser
from models.loader import LoaderCheckPoint

nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path


class BaseResponse(BaseModel):
    code: int = pydantic.Field(200, description="HTTP status code")
    msg: str = pydantic.Field("success", description="HTTP status message")

    class Config:
        schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
            }
        }

class ListKnowledgesResponse(BaseResponse):
    data: List[str] = pydantic.Field(..., description="List of knowledge ids")

    class Config:
        schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
                "data": ["hibox问答助手", "前端组件库"],
            }
        }


class ListDocsResponse(BaseResponse):
    data: List[str] = pydantic.Field(..., description="List of document names")

    class Config:
        schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
                "data": ["doc1.docx", "doc2.pdf", "doc3.txt"],
            }
        }

class SourceDocument(BaseModel):
    score: int
    source: str
    page_number: int
    content: str


class ChatMessage(BaseResponse):
    question: str = pydantic.Field(..., description="Question text")
    response: str = pydantic.Field(..., description="Response text")
    source_documents: List[SourceDocument] = pydantic.Field(
        ..., description="List of source documents and their scores"
    )

    class Config:
        schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
                "question": "工伤保险如何办理？",
                "response": "参保单位为员工缴纳工伤保险费。",
                "source_documents": [
                    {"source": "doc1.docx", "content": "广州市单位从业的特定人员参加工伤保险办事指引.docx：\n\n\t( 一)  从业单位  (组织)  按“自愿参保”原则，  为未建 立劳动关系的特定从业人员单项参加工伤保险 、缴纳工伤保 险费。.", "page_num": 1, "score": 321 },
                    {"source": "doc2.pdf", "content": "doc2 content...", "page_num": 1, "score": 397 },
                    {"source": "doc3.txt", "content": "doc3 content...", "page_num": 1, "score": 476 },
                ],
            }
        }


def get_folder_path(local_doc_id: str):
    return os.path.join(UPLOAD_ROOT_PATH, local_doc_id)


def get_vs_path(local_doc_id: str):
    return os.path.join(VS_ROOT_PATH, local_doc_id)


def get_file_path(local_doc_id: str, doc_name: str):
    return os.path.join(UPLOAD_ROOT_PATH, local_doc_id, doc_name)

def list_knowledges():
    all_knowledge_ids = []
    if not os.path.exists(VS_ROOT_PATH):
        return ListKnowledgesResponse(data=all_knowledge_ids)
    all_knowledge_ids = os.listdir(VS_ROOT_PATH)
    if not all_knowledge_ids:
        return ListKnowledgesResponse(data=all_knowledge_ids)
    all_knowledge_ids.sort()
    return ListKnowledgesResponse(data=all_knowledge_ids)


async def upload_doc(
        file: UploadFile = File(description="A single binary file"),
        knowledge_base_id: str = Form(..., description="Knowledge Base Name", example="kb1"),
):
    saved_path = get_folder_path(knowledge_base_id)
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)

    file_content = await file.read()  # 读取上传文件的内容

    file_path = os.path.join(saved_path, file.filename)
    if os.path.exists(file_path) and os.path.getsize(file_path) == len(file_content):
        file_status = f"文件 {file.filename} 已存在。"
        return BaseResponse(code=200, msg=file_status)

    with open(file_path, "wb") as f:
        f.write(file_content)

    vs_path = get_vs_path(knowledge_base_id)
    vs_path, loaded_files = local_doc_qa.init_knowledge_vector_store([file_path], vs_path)
    if len(loaded_files) > 0:
        file_status = f"文件 {file.filename} 已上传至新的知识库，并已加载知识库，请开始提问。"
        return BaseResponse(code=200, msg=file_status)
    else:
        file_status = "文件上传失败，请重新上传"
        return BaseResponse(code=500, msg=file_status)


async def upload_docs(
        files: Annotated[
            List[UploadFile], File(description="Multiple files as UploadFile")
        ],
        knowledge_base_id: str = Form(..., description="Knowledge Base Name", example="kb1"),
):
    saved_path = get_folder_path(knowledge_base_id)
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    filelist = []
    for file in files:
        file_content = ''
        file_path = os.path.join(saved_path, file.filename)
        file_content = file.file.read()
        if os.path.exists(file_path) and os.path.getsize(file_path) == len(file_content):
            continue
        with open(file_path, "ab+") as f:
            f.write(file_content)
        filelist.append(file_path)
    if filelist:
        vs_path, loaded_files = local_doc_qa.init_knowledge_vector_store(filelist, get_vs_path(knowledge_base_id))
        if len(loaded_files):
            file_status = f"已上传 {'、'.join([os.path.split(i)[-1] for i in loaded_files])} 至知识库，并已加载知识库，请开始提问"
            return BaseResponse(code=200, msg=file_status)
    file_status = "文件未成功加载，请重新上传文件"
    return BaseResponse(code=500, msg=file_status)


async def list_docs(
        knowledge_base_id: Optional[str] = Query(default=None, description="Knowledge Base Name", example="kb1")
):
    if knowledge_base_id:
        local_doc_folder = get_folder_path(knowledge_base_id)
        if not os.path.exists(local_doc_folder):
            return ListDocsResponse(code=1, msg=f"Knowledge base {knowledge_base_id} not found", data=[])
        all_doc_names = [
            doc
            for doc in os.listdir(local_doc_folder)
            if os.path.isfile(os.path.join(local_doc_folder, doc))
        ]
        return ListDocsResponse(data=all_doc_names)
    else:
        if not os.path.exists(UPLOAD_ROOT_PATH):
            all_doc_ids = []
        else:
            all_doc_ids = [
                folder
                for folder in os.listdir(UPLOAD_ROOT_PATH)
                if os.path.isdir(os.path.join(UPLOAD_ROOT_PATH, folder))
            ]

        return ListDocsResponse(data=all_doc_ids)


async def delete_docs(
        knowledge_base_id: str = Query(...,
                                       description="Knowledge Base Name(注意此方法仅删除上传的文件并不会删除知识库)",
                                       example="kb1"),
        doc_name: Optional[str] = Query(
            None, description="doc name", example="doc_name_1.pdf"
        ),
):
    knowledge_base_id = urllib.parse.unquote(knowledge_base_id)
    if not os.path.exists(os.path.join(UPLOAD_ROOT_PATH, knowledge_base_id)):
        return {"code": 1, "msg": f"Knowledge base {knowledge_base_id} not found"}
    if doc_name:
        doc_path = get_file_path(knowledge_base_id, doc_name)
        if os.path.exists(doc_path):
            os.remove(doc_path)
            return BaseResponse(code=200, msg=f"document {doc_name} delete success")
        else:
            return BaseResponse(code=1, msg=f"document {doc_name} not found")
    else:
        shutil.rmtree(get_folder_path(knowledge_base_id))
        return BaseResponse(code=200, msg=f"Knowledge Base {knowledge_base_id} delete success")

async def add_one_doc(
        knowledge_base_id: str = Form(..., description="Knowledge Base Name", example="kb1"),
        one_doc_title: str = Form(..., description="One Doc Title", example="hibox最新版本"),
        one_doc_content: str = Form(..., description="One Doc Content", example="hibox的最新版本是2.6.2"),
        chunk_size: Optional[int] = Body(2000, description="Vector Store Chunk Size", example="2000"),
        chunk_conent: Optional[bool] = Body(False, description="Vector Store Chunk Conent", example="false"),

):
    vs_path = os.path.join(VS_ROOT_PATH, knowledge_base_id)
    if not os.path.exists(vs_path):
        # return BaseResponse(code=1, msg=f"Knowledge base {knowledge_base_id} not found")
        return BaseResponse(
            code=404,
            msg=f"Knowledge base {knowledge_base_id} not found",
        )
    else:
        vs_path, loaded_files = local_doc_qa.one_knowledge_add(vs_path, one_doc_title, one_doc_content, chunk_conent, chunk_size)

        if len(loaded_files):
            file_status = f"已添加 {one_doc_title} 至知识库 {knowledge_base_id}"
            return BaseResponse(code=200, msg=file_status)
        else:
            file_status = "文档入库失败，请联系管理员"
            return BaseResponse(code=500, msg=file_status)


async def search_doc(
        knowledge_base_id: str = Body(..., description="Knowledge Base Name", example="kb1"),
        question: str = Body(..., description="Question", example="工伤保险是什么？"),
        score_threshold: Optional[int] = Body(700, description="Score Threshold", example="500"),
        top_k: Optional[int] = Body(5, description="Vector Search top_k", example="3"),
        chunk_size: Optional[int] = Body(500, description="Vector Search Chunk Size", example="2000"),
        chunk_conent: Optional[bool] = Body(True, description="Vector Search Chunk Conent", example="true"),
):
    vs_path = os.path.join(VS_ROOT_PATH, knowledge_base_id)
    if not os.path.exists(vs_path):
        # return BaseResponse(code=1, msg=f"Knowledge base {knowledge_base_id} not found")
        return ChatMessage(
            code=404,
            msg=f"Knowledge base {knowledge_base_id} not found",
            question=question,
            response="",
            source_documents=[],
        )
    else:
        resp, _ = local_doc_qa.get_knowledge_based_conent_test(
                query=question, vs_path=vs_path, score_threshold=score_threshold, vector_search_top_k=top_k, chunk_size=chunk_size, chunk_conent=chunk_conent
        )
        source_documents = [
            SourceDocument(source=os.path.split(doc.metadata.get("source", "unknown"))[-1], content=doc.page_content, score=doc.metadata['score'], page_number=doc.metadata.get("page_number", 1))
            for _, doc in enumerate(resp["source_documents"])
        ]
        # 按score从小到大排序
        source_documents.sort(key=lambda x: x.score)

        return ChatMessage(
            msg="success",
            question=question,
            response="",
            source_documents=source_documents,
        )



async def stream_local_doc_chat(websocket: WebSocket, knowledge_base_id: str):
    await websocket.accept()
    turn = 1
    while True:
        input_json = await websocket.receive_json()
        question, knowledge_base_id = input_json[""], input_json["knowledge_base_id"]
        vs_path = os.path.join(VS_ROOT_PATH, knowledge_base_id)

        if not os.path.exists(vs_path):
            await websocket.send_json({"error": f"Knowledge base {knowledge_base_id} not found"})
            await websocket.close()
            return

        await websocket.send_json({"question": question, "turn": turn, "flag": "start"})

        last_print_len = 0
        for resp in local_doc_qa.get_knowledge_based_answer(
                query=question, vs_path=vs_path, streaming=True
        ):
            await websocket.send_text(resp["result"][last_print_len:])
            last_print_len = len(resp["result"])

        source_documents = [
            f"""出处 [{inum + 1}] {os.path.split(doc.metadata['source'])[-1]}：\n\n{doc.page_content}\n\n"""
            f"""相关度：{doc.metadata['score']}\n\n"""
            for inum, doc in enumerate(resp["source_documents"])
        ]

        await websocket.send_text(
            json.dumps(
                {
                    "question": question,
                    "turn": turn,
                    "flag": "end",
                    "sources_documents": source_documents,
                },
                ensure_ascii=False,
            )
        )
        turn += 1


async def document():
    return RedirectResponse(url="/docs")





def api_start(host, port):
    global app
    global local_doc_qa

    llm_model_ins = shared.loaderLLM()

    app = FastAPI(title="向量知识库-Beta环境", description="`联系咚咚群`: 1026086734",)
    # Add CORS middleware to allow all origins
    # 在config.py中设置OPEN_DOMAIN=True，允许跨域
    # set OPEN_DOMAIN=True in config.py to allow cross-domain
    if OPEN_CROSS_DOMAIN:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    app.get("/", response_model=BaseResponse)(document)

    app.get("/vs/list_knowledges", response_model=ListKnowledgesResponse)(list_knowledges)
    app.get("/vs/list_docs", response_model=ListDocsResponse)(list_docs)
    app.post("/vs/upload_knowledge_doc", response_model=BaseResponse)(upload_doc)
    app.post("/vs/upload_knowledge_docs", response_model=BaseResponse)(upload_docs)
    app.delete("/vs/delete_doc", response_model=BaseResponse)(delete_docs)
    app.post("/vs/add_one_doc", response_model=BaseResponse)(add_one_doc)
    app.post("/vs/search_doc", response_model=ChatMessage)(search_doc)
    app.websocket("/vs/stream-chat/{knowledge_base_id}")(stream_local_doc_chat)

    local_doc_qa = LocalDocQA()
    local_doc_qa.init_cfg(
        llm_model=llm_model_ins,
        embedding_model=EMBEDDING_MODEL,
        embedding_device=EMBEDDING_DEVICE,
        top_k=VECTOR_SEARCH_TOP_K,
    )
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7866)
    # 初始化消息
    args = None
    args = parser.parse_args()
    args_dict = vars(args)
    shared.loaderCheckPoint = LoaderCheckPoint(args_dict)
    api_start(args.host, args.port)
