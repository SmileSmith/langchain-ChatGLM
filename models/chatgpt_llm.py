from abc import ABC
from langchain.llms.base import LLM
from langchain import LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from typing import Optional, List, Any
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.base import BaseCallbackHandler
from models.loader import LoaderCheckPoint
from models.base import (BaseAnswer,
                         AnswerResult,
                         AnswerResultStream)
from configs.model_config import OPENAI_API_KEY, OPENAI_API_BASE

template="""AI是一个前端开发工程师：
{history}
Human: {human_input}
AI:"""

promptTemp = PromptTemplate(
    input_variables=["history", "human_input"],
    template=template
)


class StreamingCallbackHandler(BaseCallbackHandler):
    prompt: str = "",
    answer: str = "",
    history: List[List[str]] = [],
    generate_with_callback: AnswerResultStream = None

    def __init__(self, prompt: str, history: List[List[str]] = [], generate_with_callback: AnswerResultStream = None):
        super().__init__()
        self.history = history
        self.prompt = prompt
        self.answer = ""
        self.generate_with_callback = generate_with_callback


    def on_llm_new_token(self, token: str,  **kwargs: Any) -> None:
        self.answer = "".join(self.answer) + token
        self.history[-1] = [self.prompt, self.answer]
        answer_result = AnswerResult()
        answer_result.history = self.history
        answer_result.llm_output = {"answer": self.answer}
        self.generate_with_callback(answer_result)


class CHATGPTLLM(BaseAnswer, LLM, ABC):
    max_token: int = 2048
    temperature: float = 0.1
    top_p = 0.8
    # history = []
    checkPoint: LoaderCheckPoint = None
    llmChain: LLMChain = None
    history_len: int = 10

    def __init__(self, checkPoint: LoaderCheckPoint = None):
        super().__init__()
        self.checkPoint = checkPoint

    @property
    def _llm_type(self) -> str:
        return "ChatGPT"

    @property
    def _check_point(self) -> LoaderCheckPoint:
        return self.checkPoint

    @property
    def _history_len(self) -> int:
        return self.history_len

    def set_history_len(self, history_len: int = 10) -> None:
        self.history_len = history_len

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        pass

    def _generate_answer(self, prompt: str,
                         history: List[List[str]] = [],
                         streaming: bool = False,
                         generate_with_callback: AnswerResultStream = None) -> None:

        history += [[]]
        llm = ChatOpenAI(temperature=self.temperature, openai_api_key=OPENAI_API_KEY, openai_api_base=OPENAI_API_BASE, streaming=streaming, verbose=True, callback_manager=CallbackManager([StreamingCallbackHandler(prompt, history, generate_with_callback)]))
        if (self.llmChain is None):
            self.llmChain = LLMChain(
                llm=llm,
                prompt=promptTemp,
                verbose=True,
                memory=ConversationBufferWindowMemory(k=2),
            )
        self.llmChain.llm = llm
        result = self.llmChain.predict(human_input=prompt, history=history)
        print("result")
        print(result)



